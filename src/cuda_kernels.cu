#include \"../include/cuda_kernels.hpp\"
#include <cuda_runtime.h>

// 간단한 벡터 덧셈 커널
__global__ void vectorAddKernel(int* input, int* output, int size, int addValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] + addValue;
    }
}

// 메모리 테스트 커널
__global__ void memoryTestKernel(int* array, int size, int* errorCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 원래 값 저장
        int original = array[idx];
        
        // 테스트 값 쓰기
        array[idx] = idx + 12345;
        
        // 즉시 읽어서 확인
        int readValue = array[idx];
        
        // 검증
        if (readValue != (idx + 12345)) {
            atomicAdd(errorCount, 1);
        }
        
        // 원래 값 복원
        array[idx] = original;
    }
}

// 배열 초기화 커널
__global__ void initializeArrayKernel(int* array, int size, int initValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        array[idx] = initValue + idx;
    }
}

// 배열 검증 커널
__global__ void validateArrayKernel(int* array, int size, int expectedValue, int* errorCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (array[idx] != (expectedValue + idx)) {
            atomicAdd(errorCount, 1);
        }
    }
}