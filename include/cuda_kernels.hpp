#ifndef CUDA_KERNELS_HPP
#define CUDA_KERNELS_HPP

#include <cuda_runtime.h>

// 간단한 벡터 덧셈 커널 (테스트용)
__global__ void vectorAddKernel(int* input, int* output, int size, int addValue);

// 메모리 테스트 커널
__global__ void memoryTestKernel(int* array, int size, int* errorCount);

// 배열 초기화 커널
__global__ void initializeArrayKernel(int* array, int size, int initValue);

// 배열 검증 커널
__global__ void validateArrayKernel(int* array, int size, int expectedValue, int* errorCount);

#endif // CUDA_KERNELS_HPP