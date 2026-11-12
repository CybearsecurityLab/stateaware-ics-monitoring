#include <hip/hip_runtime.h>
#include <jni.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct DeviceContext {
    float* values = nullptr;
    int* activeRows = nullptr;
    std::uint8_t* rightEligible = nullptr;
    std::uint8_t* leftCovered = nullptr;
    std::uint8_t* rightCovered = nullptr;
    float* lowerBounds = nullptr;
    float* upperBounds = nullptr;
    int* supports = nullptr;
    int candidateCapacity = 0;
    int rowCount = 0;
    int dimensions = 0;
    int activeRowCount = 0;
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

template <typename T>
void upload(JNIEnv* env, T** destination, const T* source, std::size_t count,
            const char* operation) {
    if (count == 0)
        return;
    checkHip(env, hipMalloc(reinterpret_cast<void**>(destination), count * sizeof(T)),
             "HIP device allocation");
    if (env->ExceptionCheck())
        return;
    checkHip(env, hipMemcpy(*destination, source, count * sizeof(T), hipMemcpyHostToDevice),
             operation);
}

void freeContext(DeviceContext* context) {
    if (context == nullptr)
        return;
    hipFree(context->values);
    hipFree(context->activeRows);
    hipFree(context->rightEligible);
    hipFree(context->leftCovered);
    hipFree(context->rightCovered);
    hipFree(context->lowerBounds);
    hipFree(context->upperBounds);
    hipFree(context->supports);
    delete context;
}

bool ensurePopulationCapacity(JNIEnv* env, DeviceContext* context, int candidateCount) {
    if (candidateCount <= context->candidateCapacity)
        return true;

    float* lowerBounds = nullptr;
    float* upperBounds = nullptr;
    int* supports = nullptr;
    const std::size_t boundCount = static_cast<std::size_t>(candidateCount) * context->dimensions;
    checkHip(env, hipMalloc(reinterpret_cast<void**>(&lowerBounds), boundCount * sizeof(float)),
             "HIP lower-bound allocation");
    if (env->ExceptionCheck()) goto cleanup;
    checkHip(env, hipMalloc(reinterpret_cast<void**>(&upperBounds), boundCount * sizeof(float)),
             "HIP upper-bound allocation");
    if (env->ExceptionCheck()) goto cleanup;
    checkHip(env, hipMalloc(reinterpret_cast<void**>(&supports),
                            static_cast<std::size_t>(candidateCount) * 2 * sizeof(int)),
             "HIP support-count allocation");
    if (env->ExceptionCheck()) goto cleanup;

    hipFree(context->lowerBounds);
    hipFree(context->upperBounds);
    hipFree(context->supports);
    context->lowerBounds = lowerBounds;
    context->upperBounds = upperBounds;
    context->supports = supports;
    context->candidateCapacity = candidateCount;
    return true;

cleanup:
    hipFree(lowerBounds);
    hipFree(upperBounds);
    hipFree(supports);
    return false;
}

__global__ void countSupportsKernel(
    const float* __restrict__ values,
    const int* __restrict__ activeRows,
    const std::uint8_t* __restrict__ rightEligible,
    const std::uint8_t* __restrict__ leftCovered,
    const std::uint8_t* __restrict__ rightCovered,
    const float* __restrict__ lowerBounds,
    const float* __restrict__ upperBounds,
    int rowCount,
    int activeRowCount,
    int dimensions,
    int candidateCount,
    int leftDimensions,
    bool evaluateLeft,
    bool evaluateRight,
    int* __restrict__ conditionSupports,
    int* __restrict__ ruleSupports) {
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidateCount)
        return;

    int conditionSupport = 0;
    int ruleSupport = 0;
    const float* candidateLower = lowerBounds + candidate * dimensions;
    const float* candidateUpper = upperBounds + candidate * dimensions;

    for (int activeIndex = 0; activeIndex < activeRowCount; ++activeIndex) {
        int row = activeRows[activeIndex];
        bool conditionCovered = true;

        if (evaluateLeft) {
            for (int dimension = 0; dimension < leftDimensions; ++dimension) {
                float value = values[dimension * rowCount + row];
                conditionCovered = value >= candidateLower[dimension]
                    && value <= candidateUpper[dimension];
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
                float value = values[dimension * rowCount + row];
                ruleCovered = value >= candidateLower[dimension]
                    && value <= candidateUpper[dimension];
            }
        }
        else {
            ruleCovered = rightCovered[row] != 0;
        }

        if (ruleCovered)
            ++ruleSupport;
    }

    conditionSupports[candidate] = conditionSupport;
    ruleSupports[candidate] = ruleSupport;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_src_solver_QuantMinerGpuSupportCounts_nativeCreateContext(
    JNIEnv* env, jclass, jfloatArray javaValues, jint rowCount, jint dimensions,
    jintArray javaActiveRows, jint activeRowCount, jbyteArray javaRightEligible) {
    if (rowCount < 0 || dimensions <= 0 || activeRowCount < 0
        || activeRowCount > rowCount
        || env->GetArrayLength(javaValues) != rowCount * dimensions
        || env->GetArrayLength(javaActiveRows) < activeRowCount
        || env->GetArrayLength(javaRightEligible) != rowCount) {
        throwJava(env, "java/lang/IllegalArgumentException", "Invalid native QuantMiner GPU context geometry");
        return 0;
    }

    DeviceContext* context = new DeviceContext();
    context->rowCount = rowCount;
    context->dimensions = dimensions;
    context->activeRowCount = activeRowCount;

    jfloat* values = env->GetFloatArrayElements(javaValues, nullptr);
    jint* activeRows = env->GetIntArrayElements(javaActiveRows, nullptr);
    jbyte* rightEligible = env->GetByteArrayElements(javaRightEligible, nullptr);
    if (values == nullptr || activeRows == nullptr || rightEligible == nullptr) {
        if (values != nullptr) env->ReleaseFloatArrayElements(javaValues, values, JNI_ABORT);
        if (activeRows != nullptr) env->ReleaseIntArrayElements(javaActiveRows, activeRows, JNI_ABORT);
        if (rightEligible != nullptr) env->ReleaseByteArrayElements(javaRightEligible, rightEligible, JNI_ABORT);
        delete context;
        return 0;
    }

    for (int index = 0; index < activeRowCount; ++index) {
        if (activeRows[index] < 0 || activeRows[index] >= rowCount) {
            throwJava(env, "java/lang/IllegalArgumentException", "QuantMiner GPU active row is out of range");
            break;
        }
    }

    if (!env->ExceptionCheck()) {
        upload(env, &context->values, values,
               static_cast<std::size_t>(rowCount) * dimensions,
               "HIP value-buffer upload");
    }
    if (!env->ExceptionCheck()) {
        upload(env, &context->activeRows, activeRows, activeRowCount,
               "HIP active-row upload");
    }
    if (!env->ExceptionCheck()) {
        upload(env, &context->rightEligible,
               reinterpret_cast<std::uint8_t*>(rightEligible), rowCount,
               "HIP right-eligibility upload");
    }
    if (!env->ExceptionCheck() && rowCount > 0) {
        checkHip(env, hipMalloc(reinterpret_cast<void**>(&context->leftCovered), rowCount),
                 "HIP left-coverage allocation");
    }
    if (!env->ExceptionCheck() && rowCount > 0) {
        checkHip(env, hipMalloc(reinterpret_cast<void**>(&context->rightCovered), rowCount),
                 "HIP right-coverage allocation");
    }

    env->ReleaseFloatArrayElements(javaValues, values, JNI_ABORT);
    env->ReleaseIntArrayElements(javaActiveRows, activeRows, JNI_ABORT);
    env->ReleaseByteArrayElements(javaRightEligible, rightEligible, JNI_ABORT);

    if (env->ExceptionCheck()) {
        freeContext(context);
        return 0;
    }
    return reinterpret_cast<jlong>(context);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_src_solver_QuantMinerGpuSupportCounts_nativeCount(
    JNIEnv* env, jclass, jlong contextHandle, jfloatArray javaLowerBounds,
    jfloatArray javaUpperBounds, jint candidateCount, jint intervals,
    jint leftDimensions, jboolean evaluateLeft, jboolean evaluateRight,
    jbyteArray javaLeftCovered, jbyteArray javaRightCovered) {
    DeviceContext* context = reinterpret_cast<DeviceContext*>(contextHandle);
    if (context == nullptr || candidateCount < 0 || intervals != context->dimensions
        || leftDimensions < 0 || leftDimensions > context->dimensions
        || env->GetArrayLength(javaLowerBounds) < candidateCount * intervals
        || env->GetArrayLength(javaUpperBounds) < candidateCount * intervals
        || env->GetArrayLength(javaLeftCovered) != context->rowCount
        || env->GetArrayLength(javaRightCovered) != context->rowCount) {
        throwJava(env, "java/lang/IllegalArgumentException", "Invalid native QuantMiner GPU population");
        return nullptr;
    }

    jintArray javaResult = env->NewIntArray(candidateCount * 2);
    if (javaResult == nullptr)
        return nullptr;
    if (candidateCount == 0)
        return javaResult;
    if (!ensurePopulationCapacity(env, context, candidateCount))
        return nullptr;

    jfloat* lowerBounds = env->GetFloatArrayElements(javaLowerBounds, nullptr);
    jfloat* upperBounds = env->GetFloatArrayElements(javaUpperBounds, nullptr);
    if (lowerBounds == nullptr || upperBounds == nullptr) {
        if (lowerBounds != nullptr)
            env->ReleaseFloatArrayElements(javaLowerBounds, lowerBounds, JNI_ABORT);
        if (upperBounds != nullptr)
            env->ReleaseFloatArrayElements(javaUpperBounds, upperBounds, JNI_ABORT);
        return nullptr;
    }

    checkHip(env, hipMemcpy(context->lowerBounds, lowerBounds,
                            static_cast<std::size_t>(candidateCount) * intervals * sizeof(float),
                            hipMemcpyHostToDevice),
             "HIP lower-bound upload");
    if (!env->ExceptionCheck()) {
        checkHip(env, hipMemcpy(context->upperBounds, upperBounds,
                                static_cast<std::size_t>(candidateCount) * intervals * sizeof(float),
                                hipMemcpyHostToDevice),
                 "HIP upper-bound upload");
    }
    env->ReleaseFloatArrayElements(javaLowerBounds, lowerBounds, JNI_ABORT);
    env->ReleaseFloatArrayElements(javaUpperBounds, upperBounds, JNI_ABORT);
    if (env->ExceptionCheck())
        return nullptr;

    if (!evaluateLeft && context->rowCount > 0) {
        jbyte* leftCovered = env->GetByteArrayElements(javaLeftCovered, nullptr);
        if (leftCovered == nullptr)
            return nullptr;
        checkHip(env, hipMemcpy(context->leftCovered, leftCovered, context->rowCount,
                                hipMemcpyHostToDevice),
                 "HIP left-coverage upload");
        env->ReleaseByteArrayElements(javaLeftCovered, leftCovered, JNI_ABORT);
    }
    if (env->ExceptionCheck())
        return nullptr;
    if (!evaluateRight && context->rowCount > 0) {
        jbyte* rightCovered = env->GetByteArrayElements(javaRightCovered, nullptr);
        if (rightCovered == nullptr)
            return nullptr;
        checkHip(env, hipMemcpy(context->rightCovered, rightCovered, context->rowCount,
                                hipMemcpyHostToDevice),
                 "HIP right-coverage upload");
        env->ReleaseByteArrayElements(javaRightCovered, rightCovered, JNI_ABORT);
    }
    if (env->ExceptionCheck())
        return nullptr;

    const int threadsPerBlock = 256;
    const int blocks = (candidateCount + threadsPerBlock - 1) / threadsPerBlock;
    int* deviceConditionSupports = context->supports;
    int* deviceRuleSupports = context->supports + candidateCount;
    hipLaunchKernelGGL(countSupportsKernel, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       context->values, context->activeRows, context->rightEligible,
                       context->leftCovered, context->rightCovered,
                       context->lowerBounds, context->upperBounds,
                       context->rowCount, context->activeRowCount, context->dimensions,
                       candidateCount, leftDimensions, evaluateLeft == JNI_TRUE,
                       evaluateRight == JNI_TRUE,
                       deviceConditionSupports, deviceRuleSupports);
    checkHip(env, hipGetLastError(), "HIP support-count kernel launch");
    if (env->ExceptionCheck())
        return nullptr;

    jint* hostSupports = new jint[static_cast<std::size_t>(candidateCount) * 2];
    checkHip(env, hipMemcpy(hostSupports, context->supports,
                            static_cast<std::size_t>(candidateCount) * 2 * sizeof(int),
                            hipMemcpyDeviceToHost),
             "HIP support-count download");
    if (!env->ExceptionCheck()) {
        jint* packedSupports = new jint[static_cast<std::size_t>(candidateCount) * 2];
        for (int candidate = 0; candidate < candidateCount; ++candidate) {
            packedSupports[candidate * 2] = hostSupports[candidate];
            packedSupports[candidate * 2 + 1] = hostSupports[candidate + candidateCount];
        }
        env->SetIntArrayRegion(javaResult, 0, candidateCount * 2, packedSupports);
        delete[] packedSupports;
    }
    delete[] hostSupports;
    return env->ExceptionCheck() ? nullptr : javaResult;
}

extern "C" JNIEXPORT void JNICALL
Java_src_solver_QuantMinerGpuSupportCounts_nativeReleaseContext(
    JNIEnv*, jclass, jlong contextHandle) {
    freeContext(reinterpret_cast<DeviceContext*>(contextHandle));
}
