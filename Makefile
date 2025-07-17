# Phase2 2-Phase Workflow MPI+CUDA Distributed Label Propagation Makefile

CXX = mpicxx
NVCC = nvcc

# 컴파일 플래그
CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall -g
NVCCFLAGS = -std=c++17 -O3 -arch=sm_70 -Xcompiler -fopenmp -g -G
INCLUDES = -I../include -I../../datasets -I/usr/local/cuda/include -I/usr/lib/x86_64-linux-gnu/openmpi/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcurand -fopenmp -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi

# 소스 파일
CUDA_SOURCES = src/mpi_distributed_label_propagation.cu src/cuda_kernels.cu
CPP_SOURCES = src/main.cpp
HEADERS = include/mpi_distributed_label_propagation.hpp include/multi_gpu_manager.hpp include/cuda_kernels.hpp

# 빌드 타겟
TARGET = mpi_distributed_label_propagation
COMPLETE_TEST = complete_test_example
GPU_DEBUG_TEST = gpu_debug_test
CUDA_OBJECT = src/mpi_distributed_label_propagation.o
CUDA_KERNELS_OBJECT = src/cuda_kernels.o
CPP_OBJECT = src/main.o
COMPLETE_TEST_OBJECT = src/complete_test_example.o
GPU_DEBUG_OBJECT = src/gpu_debug_test.o

.PHONY: all clean test complete-test gpu-debug-test

all: $(TARGET) $(COMPLETE_TEST) $(GPU_DEBUG_TEST)

$(TARGET): $(CUDA_OBJECT) $(CUDA_KERNELS_OBJECT) $(CPP_OBJECT)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(COMPLETE_TEST): $(CUDA_OBJECT) $(CUDA_KERNELS_OBJECT) $(COMPLETE_TEST_OBJECT)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(GPU_DEBUG_TEST): $(CUDA_OBJECT) $(CUDA_KERNELS_OBJECT) $(GPU_DEBUG_OBJECT)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(CUDA_OBJECT): src/mpi_distributed_label_propagation.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(CUDA_KERNELS_OBJECT): src/cuda_kernels.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(CPP_OBJECT): $(CPP_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(COMPLETE_TEST_OBJECT): src/complete_test_example.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(GPU_DEBUG_OBJECT): src/gpu_debug_test.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(CUDA_OBJECT) $(CUDA_KERNELS_OBJECT) $(CPP_OBJECT) $(COMPLETE_TEST_OBJECT) $(GPU_DEBUG_OBJECT) $(TARGET) $(COMPLETE_TEST) $(GPU_DEBUG_TEST)

# 테스트 실행
test: $(TARGET)
	mpirun -np 2 ./$(TARGET) ../../datasets/ljournal-2008 4

# GPU 디버그 테스트
gpu-debug-test: $(GPU_DEBUG_TEST)
	mpirun -np 2 ./$(GPU_DEBUG_TEST)

# 완전 테스트 실행 (MPI+OpenMP+CUDA)
complete-test: $(COMPLETE_TEST)
	mpirun -np 2 ./$(COMPLETE_TEST) ../../datasets/ljournal-2008.adj.graph-txt 4

install: $(TARGET) $(COMPLETE_TEST)
	cp $(TARGET) $(COMPLETE_TEST) ../../../local/bin/

.SUFFIXES: .cu .o