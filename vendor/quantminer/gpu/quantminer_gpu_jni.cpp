#include <hip/hip_runtime.h>
#include <jni.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;
using Clock = std::chrono::steady_clock;

struct ProfileCounters {
    std::uint64_t contexts = 0;
    std::uint64_t evaluations = 0;
    std::uint64_t nativeContextNs = 0;
    std::uint64_t jniTotalNs = 0;
    std::uint64_t allocationNs = 0;
    std::uint64_t h2dNs = 0;
    std::uint64_t kernelNs = 0;
    std::uint64_t d2hNs = 0;
    std::uint64_t candidateTotal = 0;
    std::uint64_t activeRowTotal = 0;
    std::uint64_t coverageUploads = 0;
    std::uint64_t coverageCacheHits = 0;
    int candidateMin = std::numeric_limits<int>::max();
    int candidateMax = 0;
    int activeRowMin = std::numeric_limits<int>::max();
    int activeRowMax = 0;
};

bool gProfiling = false;
ProfileCounters gProfile;

struct DeviceBuffers {
    float* values = nullptr;
    std::size_t valueCapacity = 0;
    int* activeRows = nullptr;
    std::size_t activeRowCapacity = 0;
    std::uint8_t* rightEligible = nullptr;
    std::uint8_t* leftCovered = nullptr;
    std::uint8_t* rightCovered = nullptr;
    std::size_t rightEligibleCapacity = 0;
    std::size_t leftCoverageCapacity = 0;
    std::size_t rightCoverageCapacity = 0;
    float* packedBounds = nullptr;
    std::size_t boundCapacity = 0;
    int* supports = nullptr;
    std::size_t supportCapacity = 0;

    std::vector<float> hostValues;
    std::vector<jint> hostActiveRows;
    std::vector<jbyte> hostRightEligible;
    std::vector<float> hostPackedBounds;
    std::vector<jint> hostSupports;
    std::vector<jbyte> hostLeftCovered;
    std::vector<jbyte> hostRightCovered;
    std::vector<jbyte> coverageScratch;
    bool leftCoverageValid = false;
    bool rightCoverageValid = false;

    hipEvent_t kernelStart = nullptr;
    hipEvent_t kernelStop = nullptr;
};

struct DeviceContext {
    DeviceBuffers* buffers = nullptr;
    int rowCount = 0;
    int dimensions = 0;
    int activeRowCount = 0;
};

std::mutex gBufferPoolMutex;
std::vector<DeviceBuffers*> gBufferPool;

Clock::time_point timerStart() {
    return gProfiling ? Clock::now() : Clock::time_point();
}

std::uint64_t timerElapsed(Clock::time_point start) {
    if (!gProfiling)
        return 0;
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

class ScopedTimer {
public:
    explicit ScopedTimer(std::uint64_t& destination)
        : destination_(destination), start_(timerStart()) {}
    ~ScopedTimer() { destination_ += timerElapsed(start_); }

private:
    std::uint64_t& destination_;
    Clock::time_point start_;
};

void throwJava(JNIEnv* env, const char* className, const std::string& message) {
    jclass exceptionClass = env->FindClass(className);
    if (exceptionClass != nullptr)
        env->ThrowNew(exceptionClass, message.c_str());
}

void checkHip(JNIEnv* env, hipError_t status, const char* operation) {
    if (status == hipSuccess)
        return;
    std::ostringstream message;
    message << operation << " failed: " << hipGetErrorString(status)
            << " (HIP error " << static_cast<int>(status) << ")";
    throwJava(env, "java/lang/IllegalStateException", message.str());
}

DeviceBuffers* acquireBuffers() {
    std::lock_guard<std::mutex> lock(gBufferPoolMutex);
    if (!gBufferPool.empty()) {
        DeviceBuffers* buffers = gBufferPool.back();
        gBufferPool.pop_back();
        buffers->leftCoverageValid = false;
        buffers->rightCoverageValid = false;
        return buffers;
    }
    return new DeviceBuffers();
}

void releaseBuffers(DeviceBuffers* buffers) {
    std::lock_guard<std::mutex> lock(gBufferPoolMutex);
    gBufferPool.push_back(buffers);
}

template <typename T>
bool ensureCapacity(JNIEnv* env, T** pointer, std::size_t& capacity,
                    std::size_t required, const char* operation) {
    if (required <= capacity && (*pointer != nullptr || required == 0))
        return true;

    T* replacement = nullptr;
    Clock::time_point allocationStart = timerStart();
    hipError_t status = hipMalloc(reinterpret_cast<void**>(&replacement), required * sizeof(T));
    gProfile.allocationNs += timerElapsed(allocationStart);
    checkHip(env, status, operation);
    if (env->ExceptionCheck())
        return false;

    if (*pointer != nullptr) {
        allocationStart = timerStart();
        status = hipFree(*pointer);
        gProfile.allocationNs += timerElapsed(allocationStart);
        checkHip(env, status, "HIP device-buffer release");
        if (env->ExceptionCheck()) {
            hipFree(replacement);
            return false;
        }
    }

    *pointer = replacement;
    capacity = required;
    return true;
}

bool ensureEvents(JNIEnv* env, DeviceBuffers* buffers) {
    if (!gProfiling || buffers->kernelStart != nullptr)
        return true;
    Clock::time_point allocationStart = timerStart();
    hipError_t status = hipEventCreate(&buffers->kernelStart);
    gProfile.allocationNs += timerElapsed(allocationStart);
    checkHip(env, status, "HIP profiling-event creation");
    if (env->ExceptionCheck())
        return false;

    allocationStart = timerStart();
    status = hipEventCreate(&buffers->kernelStop);
    gProfile.allocationNs += timerElapsed(allocationStart);
    checkHip(env, status, "HIP profiling-event creation");
    return !env->ExceptionCheck();
}

void h2d(JNIEnv* env, void* destination, const void* source, std::size_t bytes,
         const char* operation) {
    if (bytes == 0)
        return;
    Clock::time_point start = timerStart();
    hipError_t status = hipMemcpy(destination, source, bytes, hipMemcpyHostToDevice);
    gProfile.h2dNs += timerElapsed(start);
    checkHip(env, status, operation);
}

void uploadCoverage(JNIEnv* env, DeviceContext* context, jbyteArray javaCoverage,
                    bool leftSide) {
    DeviceBuffers* buffers = context->buffers;
    if (context->rowCount == 0)
        return;

    buffers->coverageScratch.resize(context->rowCount);
    env->GetByteArrayRegion(javaCoverage, 0, context->rowCount,
                            buffers->coverageScratch.data());
    if (env->ExceptionCheck())
        return;

    std::vector<jbyte>& cached = leftSide
        ? buffers->hostLeftCovered : buffers->hostRightCovered;
    bool& valid = leftSide ? buffers->leftCoverageValid : buffers->rightCoverageValid;
    if (valid && cached.size() == buffers->coverageScratch.size()
        && std::memcmp(cached.data(), buffers->coverageScratch.data(),
                       buffers->coverageScratch.size()) == 0) {
        if (gProfiling)
            ++gProfile.coverageCacheHits;
        return;
    }

    std::uint8_t* destination = leftSide
        ? buffers->leftCovered : buffers->rightCovered;
    h2d(env, destination, buffers->coverageScratch.data(), context->rowCount,
        leftSide ? "HIP left-coverage upload" : "HIP right-coverage upload");
    if (env->ExceptionCheck())
        return;
    cached = buffers->coverageScratch;
    valid = true;
    if (gProfiling)
        ++gProfile.coverageUploads;
}

__global__ void countSupportsKernel(
    const float* __restrict__ values,
    const int* __restrict__ activeRows,
    const std::uint8_t* __restrict__ rightEligible,
    const std::uint8_t* __restrict__ leftCovered,
    const std::uint8_t* __restrict__ rightCovered,
    const float* __restrict__ packedBounds,
    int rowCount,
    int activeRowCount,
    int dimensions,
    int leftDimensions,
    bool evaluateLeft,
    bool evaluateRight,
    int* __restrict__ supports) {
    const int candidate = blockIdx.x;
    const int lane = threadIdx.x;
    const float* candidateBounds = packedBounds + candidate * dimensions * 2;
    int conditionSupport = 0;
    int ruleSupport = 0;

    for (int activeIndex = lane; activeIndex < activeRowCount;
         activeIndex += kThreadsPerBlock) {
        const int row = activeRows[activeIndex];
        bool conditionCovered = true;

        if (evaluateLeft) {
            for (int dimension = 0; dimension < leftDimensions; ++dimension) {
                const float value = values[dimension * rowCount + row];
                conditionCovered = value >= candidateBounds[dimension * 2]
                    && value <= candidateBounds[dimension * 2 + 1];
                if (!conditionCovered)
                    break;
            }
        }
        else {
            conditionCovered = leftCovered[row] != 0;
        }

        if (!conditionCovered)
            continue;
        ++conditionSupport;

        bool ruleCovered;
        if (evaluateRight) {
            ruleCovered = rightEligible[row] != 0;
            for (int dimension = leftDimensions;
                 ruleCovered && dimension < dimensions; ++dimension) {
                const float value = values[dimension * rowCount + row];
                ruleCovered = value >= candidateBounds[dimension * 2]
                    && value <= candidateBounds[dimension * 2 + 1];
            }
        }
        else {
            ruleCovered = rightCovered[row] != 0;
        }

        if (ruleCovered)
            ++ruleSupport;
    }

    __shared__ int conditionReduction[kThreadsPerBlock];
    __shared__ int ruleReduction[kThreadsPerBlock];
    conditionReduction[lane] = conditionSupport;
    ruleReduction[lane] = ruleSupport;
    __syncthreads();

    for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            conditionReduction[lane] += conditionReduction[lane + stride];
            ruleReduction[lane] += ruleReduction[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        supports[candidate * 2] = conditionReduction[0];
        supports[candidate * 2 + 1] = ruleReduction[0];
    }
}

void recordGeometry(int candidateCount, int activeRowCount) {
    if (!gProfiling)
        return;
    ++gProfile.evaluations;
    gProfile.candidateTotal += candidateCount;
    gProfile.activeRowTotal += activeRowCount;
    gProfile.candidateMin = std::min(gProfile.candidateMin, candidateCount);
    gProfile.candidateMax = std::max(gProfile.candidateMax, candidateCount);
    gProfile.activeRowMin = std::min(gProfile.activeRowMin, activeRowCount);
    gProfile.activeRowMax = std::max(gProfile.activeRowMax, activeRowCount);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_src_solver_QuantMinerGpuSupportCounts_nativeCreateContext(
    JNIEnv* env, jclass, jfloatArray javaValues, jint rowCount, jint dimensions,
    jintArray javaActiveRows, jint activeRowCount, jbyteArray javaRightEligible,
    jboolean profiling) {
    gProfiling = profiling == JNI_TRUE;
    ScopedTimer contextTimer(gProfile.nativeContextNs);
    if (rowCount < 0 || dimensions <= 0 || activeRowCount < 0
        || activeRowCount > rowCount
        || env->GetArrayLength(javaValues) != rowCount * dimensions
        || env->GetArrayLength(javaActiveRows) < activeRowCount
        || env->GetArrayLength(javaRightEligible) != rowCount) {
        throwJava(env, "java/lang/IllegalArgumentException", "Invalid native QuantMiner GPU context geometry");
        return 0;
    }

    DeviceContext* context = new DeviceContext();
    context->buffers = acquireBuffers();
    context->rowCount = rowCount;
    context->dimensions = dimensions;
    context->activeRowCount = activeRowCount;
    DeviceBuffers* buffers = context->buffers;

    buffers->hostValues.resize(static_cast<std::size_t>(rowCount) * dimensions);
    buffers->hostActiveRows.resize(activeRowCount);
    buffers->hostRightEligible.resize(rowCount);
    if (rowCount * dimensions > 0)
        env->GetFloatArrayRegion(javaValues, 0, rowCount * dimensions,
                                 buffers->hostValues.data());
    if (activeRowCount > 0)
        env->GetIntArrayRegion(javaActiveRows, 0, activeRowCount,
                               buffers->hostActiveRows.data());
    if (rowCount > 0)
        env->GetByteArrayRegion(javaRightEligible, 0, rowCount,
                                buffers->hostRightEligible.data());
    if (env->ExceptionCheck()) {
        releaseBuffers(buffers);
        delete context;
        return 0;
    }

    for (int index = 0; index < activeRowCount; ++index) {
        if (buffers->hostActiveRows[index] < 0 || buffers->hostActiveRows[index] >= rowCount) {
            throwJava(env, "java/lang/IllegalArgumentException", "QuantMiner GPU active row is out of range");
            break;
        }
    }

    if (!env->ExceptionCheck())
        ensureCapacity(env, &buffers->values, buffers->valueCapacity,
                       buffers->hostValues.size(), "HIP value-buffer allocation");
    if (!env->ExceptionCheck())
        ensureCapacity(env, &buffers->activeRows, buffers->activeRowCapacity,
                       buffers->hostActiveRows.size(), "HIP active-row allocation");
    if (!env->ExceptionCheck())
        ensureCapacity(env, &buffers->rightEligible, buffers->rightEligibleCapacity,
                       rowCount, "HIP right-eligibility allocation");
    if (!env->ExceptionCheck())
        ensureCapacity(env, &buffers->leftCovered, buffers->leftCoverageCapacity,
                       rowCount, "HIP left-coverage allocation");
    if (!env->ExceptionCheck())
        ensureCapacity(env, &buffers->rightCovered, buffers->rightCoverageCapacity,
                       rowCount, "HIP right-coverage allocation");
    if (!env->ExceptionCheck())
        ensureEvents(env, buffers);

    if (!env->ExceptionCheck())
        h2d(env, buffers->values, buffers->hostValues.data(),
            buffers->hostValues.size() * sizeof(float), "HIP value-buffer upload");
    if (!env->ExceptionCheck())
        h2d(env, buffers->activeRows, buffers->hostActiveRows.data(),
            buffers->hostActiveRows.size() * sizeof(jint), "HIP active-row upload");
    if (!env->ExceptionCheck())
        h2d(env, buffers->rightEligible, buffers->hostRightEligible.data(),
            buffers->hostRightEligible.size(), "HIP right-eligibility upload");

    if (env->ExceptionCheck()) {
        releaseBuffers(buffers);
        delete context;
        return 0;
    }
    if (gProfiling)
        ++gProfile.contexts;
    return reinterpret_cast<jlong>(context);
}

extern "C" JNIEXPORT void JNICALL
Java_src_solver_QuantMinerGpuSupportCounts_nativeCount(
    JNIEnv* env, jclass, jlong contextHandle, jfloatArray javaPackedBounds,
    jint candidateCount, jint intervals, jint leftDimensions,
    jboolean evaluateLeft, jboolean evaluateRight,
    jbyteArray javaLeftCovered, jbyteArray javaRightCovered,
    jintArray javaResult) {
    ScopedTimer jniTimer(gProfile.jniTotalNs);
    DeviceContext* context = reinterpret_cast<DeviceContext*>(contextHandle);
    const std::size_t boundCount = context == nullptr ? 0
        : static_cast<std::size_t>(candidateCount) * intervals * 2;
    if (context == nullptr || candidateCount < 0 || intervals != context->dimensions
        || leftDimensions < 0 || leftDimensions > context->dimensions
        || env->GetArrayLength(javaPackedBounds) < boundCount
        || env->GetArrayLength(javaLeftCovered) != context->rowCount
        || env->GetArrayLength(javaRightCovered) != context->rowCount
        || env->GetArrayLength(javaResult) < candidateCount * 2) {
        throwJava(env, "java/lang/IllegalArgumentException", "Invalid native QuantMiner GPU population");
        return;
    }

    recordGeometry(candidateCount, context->activeRowCount);
    if (candidateCount == 0)
        return;
    DeviceBuffers* buffers = context->buffers;
    if (!ensureCapacity(env, &buffers->packedBounds, buffers->boundCapacity,
                        boundCount, "HIP packed-bound allocation")
        || !ensureCapacity(env, &buffers->supports, buffers->supportCapacity,
                           static_cast<std::size_t>(candidateCount) * 2,
                           "HIP support-count allocation")
        || !ensureEvents(env, buffers))
        return;

    buffers->hostPackedBounds.resize(boundCount);
    buffers->hostSupports.resize(static_cast<std::size_t>(candidateCount) * 2);
    env->GetFloatArrayRegion(javaPackedBounds, 0, boundCount,
                             buffers->hostPackedBounds.data());
    if (env->ExceptionCheck())
        return;
    h2d(env, buffers->packedBounds, buffers->hostPackedBounds.data(),
        boundCount * sizeof(float), "HIP packed-bound upload");
    if (env->ExceptionCheck())
        return;

    if (!evaluateLeft)
        uploadCoverage(env, context, javaLeftCovered, true);
    if (!env->ExceptionCheck() && !evaluateRight)
        uploadCoverage(env, context, javaRightCovered, false);
    if (env->ExceptionCheck())
        return;

    if (gProfiling) {
        checkHip(env, hipEventRecord(buffers->kernelStart), "HIP kernel-start event");
        if (env->ExceptionCheck())
            return;
    }
    hipLaunchKernelGGL(countSupportsKernel, dim3(candidateCount), dim3(kThreadsPerBlock), 0, 0,
                       buffers->values, buffers->activeRows, buffers->rightEligible,
                       buffers->leftCovered, buffers->rightCovered, buffers->packedBounds,
                       context->rowCount, context->activeRowCount, context->dimensions,
                       leftDimensions, evaluateLeft == JNI_TRUE, evaluateRight == JNI_TRUE,
                       buffers->supports);
    checkHip(env, hipGetLastError(), "HIP support-count kernel launch");
    if (env->ExceptionCheck())
        return;

    if (gProfiling) {
        checkHip(env, hipEventRecord(buffers->kernelStop), "HIP kernel-stop event");
        if (!env->ExceptionCheck())
            checkHip(env, hipEventSynchronize(buffers->kernelStop), "HIP kernel synchronization");
        if (env->ExceptionCheck())
            return;
        float kernelMs = 0.0f;
        checkHip(env, hipEventElapsedTime(&kernelMs, buffers->kernelStart, buffers->kernelStop),
                 "HIP kernel timing");
        if (env->ExceptionCheck())
            return;
        gProfile.kernelNs += static_cast<std::uint64_t>(kernelMs * 1000000.0f);
    }

    Clock::time_point downloadStart = timerStart();
    hipError_t downloadStatus = hipMemcpy(buffers->hostSupports.data(), buffers->supports,
                                          buffers->hostSupports.size() * sizeof(jint),
                                          hipMemcpyDeviceToHost);
    gProfile.d2hNs += timerElapsed(downloadStart);
    checkHip(env, downloadStatus, "HIP support-count download");
    if (env->ExceptionCheck())
        return;
    env->SetIntArrayRegion(javaResult, 0, candidateCount * 2, buffers->hostSupports.data());
}

extern "C" JNIEXPORT void JNICALL
Java_src_solver_QuantMinerGpuSupportCounts_nativeReleaseContext(
    JNIEnv*, jclass, jlong contextHandle) {
    DeviceContext* context = reinterpret_cast<DeviceContext*>(contextHandle);
    if (context == nullptr)
        return;
    releaseBuffers(context->buffers);
    delete context;
}

extern "C" JNIEXPORT void JNICALL
Java_src_solver_QuantMinerGpuSupportCounts_nativeDumpProfile(JNIEnv*, jclass) {
    if (!gProfiling)
        return;
    const int candidateMin = gProfile.candidateMin == std::numeric_limits<int>::max()
        ? 0 : gProfile.candidateMin;
    const int activeRowMin = gProfile.activeRowMin == std::numeric_limits<int>::max()
        ? 0 : gProfile.activeRowMin;
    std::cerr << std::fixed << std::setprecision(6)
              << "QUANTMINER_GPU_NATIVE_PROFILE"
              << " contexts=" << gProfile.contexts
              << " evaluations=" << gProfile.evaluations
              << " native_context_s=" << gProfile.nativeContextNs / 1.0e9
              << " jni_total_s=" << gProfile.jniTotalNs / 1.0e9
              << " allocation_s=" << gProfile.allocationNs / 1.0e9
              << " h2d_s=" << gProfile.h2dNs / 1.0e9
              << " kernel_s=" << gProfile.kernelNs / 1.0e9
              << " d2h_s=" << gProfile.d2hNs / 1.0e9
              << " candidate_total=" << gProfile.candidateTotal
              << " candidate_min=" << candidateMin
              << " candidate_max=" << gProfile.candidateMax
              << " active_row_mean_per_evaluation="
              << (gProfile.evaluations == 0 ? 0.0
                  : static_cast<double>(gProfile.activeRowTotal) / gProfile.evaluations)
              << " active_row_min=" << activeRowMin
              << " active_row_max=" << gProfile.activeRowMax
              << " coverage_uploads=" << gProfile.coverageUploads
              << " coverage_cache_hits=" << gProfile.coverageCacheHits
              << std::endl;
}
