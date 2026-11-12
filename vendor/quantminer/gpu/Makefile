HIPCC ?= hipcc
HIP_ARCH ?= gfx1030
JAVA_HOME ?= $(shell dirname $$(dirname $$(readlink -f $$(command -v javac))))
TARGET := libquantminer_gpu.so
SOURCE := quantminer_gpu_jni.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(HIPCC) -std=c++14 -O3 -fno-fast-math --offload-arch=$(HIP_ARCH) -shared -fPIC \
		-I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux $(SOURCE) -lamdhip64 -o $(TARGET)

clean:
	rm -f $(TARGET)
