#ifndef MULTI_GPU_MANAGER_HPP
#define MULTI_GPU_MANAGER_HPP

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <mpi.h>

struct GPUInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int warp_size;
    bool is_available;
    int mpi_rank; // 이 GPU를 사용하는 MPI 랭크
};

class MultiGPUManager {
private:
    std::vector<GPUInfo> gpu_infos_;
    std::string gpu_config_file_;
    int total_gpus_;
    int current_gpu_;
    int mpi_rank_;
    int mpi_size_;
    bool debug_mode_;
    
public:
    MultiGPUManager(bool debug = false) 
        : gpu_config_file_(\"gpu_config.json\"), total_gpus_(0), 
          current_gpu_(-1), debug_mode_(debug) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
    }
    
    ~MultiGPUManager() {
        cleanup();
    }
    
    // GPU 정보 수집 및 저장
    bool initializeGPUs() {
        // 항상 새로운 GPU 정보 수집 (안정성을 위해)
        if (debug_mode_) {
            std::cout << \"Rank \" << mpi_rank_ << \": Detecting GPUs...\" << std::endl;
        }
        
        // 새로운 GPU 정보 수집
        if (!detectGPUs()) {
            std::cerr << \"Rank \" << mpi_rank_ << \": Failed to detect GPUs\" << std::endl;
            return false;
        }
        
        // GPU 정보 저장 (랭크 0만)
        if (mpi_rank_ == 0) {
            saveGPUConfig();
        }
        
        return assignGPUToRank();
    }
    
    // GPU 감지 및 정보 수집
    bool detectGPUs() {
        cudaError_t error = cudaGetDeviceCount(&total_gpus_);
        if (error != cudaSuccess || total_gpus_ == 0) {
            if (debug_mode_) {
                std::cout << \"Rank \" << mpi_rank_ << \": No CUDA devices found\" << std::endl;
            }
            return false;
        }
        
        gpu_infos_.clear();
        gpu_infos_.reserve(total_gpus_);
        
        for (int i = 0; i < total_gpus_; i++) {
            GPUInfo info;
            cudaDeviceProp prop;
            
            error = cudaGetDeviceProperties(&prop, i);
            if (error != cudaSuccess) {
                std::cerr << \"Rank \" << mpi_rank_ << \": Failed to get properties for GPU \" << i << std::endl;
                continue;
            }
            
            // GPU 메모리 정보 얻기
            cudaSetDevice(i);
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            
            info.device_id = i;
            info.name = prop.name;
            info.total_memory = total_mem;
            info.free_memory = free_mem;
            info.compute_capability_major = prop.major;
            info.compute_capability_minor = prop.minor;
            info.multiprocessor_count = prop.multiProcessorCount;
            info.max_threads_per_block = prop.maxThreadsPerBlock;
            info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
            info.warp_size = prop.warpSize;
            info.is_available = true;
            info.mpi_rank = -1; // 아직 할당되지 않음
            
            gpu_infos_.push_back(info);
            
            if (debug_mode_) {
                std::cout << \"Rank \" << mpi_rank_ << \": Detected GPU \" << i 
                         << \" (\" << info.name << \")\" << std::endl;
                std::cout << \"  Memory: \" << (info.total_memory / (1024*1024)) << \" MB\" << std::endl;
                std::cout << \"  Compute: \" << info.compute_capability_major 
                         << \".\" << info.compute_capability_minor << std::endl;
            }
        }
        
        return !gpu_infos_.empty();
    }
    
    // MPI 랭크에 GPU 할당
    bool assignGPUToRank() {
        if (gpu_infos_.empty()) {
            return false;
        }
        
        // 라운드 로빈 방식으로 GPU 할당
        current_gpu_ = mpi_rank_ % total_gpus_;
        
        // GPU 설정
        cudaError_t error = cudaSetDevice(current_gpu_);
        if (error != cudaSuccess) {
            std::cerr << \"Rank \" << mpi_rank_ << \": Failed to set GPU \" << current_gpu_ << std::endl;
            return false;
        }
        
        // GPU 정보 업데이트
        if (current_gpu_ < gpu_infos_.size()) {
            gpu_infos_[current_gpu_].mpi_rank = mpi_rank_;
        }
        
        if (debug_mode_) {
            std::cout << \"Rank \" << mpi_rank_ << \": Assigned to GPU \" << current_gpu_ 
                     << \" (\" << gpu_infos_[current_gpu_].name << \")\" << std::endl;
        }
        
        return true;
    }
    
    // GPU 설정 파일 저장
    void saveGPUConfig() {
        std::ofstream file(gpu_config_file_);
        if (!file.is_open()) {
            std::cerr << \"Failed to open GPU config file for writing\" << std::endl;
            return;
        }
        
        file << \"{\" << std::endl;
        file << \"  \\\"total_gpus\\\": \" << total_gpus_ << \",\" << std::endl;
        file << \"  \\\"gpus\\\": [\" << std::endl;
        
        for (size_t i = 0; i < gpu_infos_.size(); i++) {
            const auto& info = gpu_infos_[i];
            file << \"    {\" << std::endl;
            file << \"      \\\"device_id\\\": \" << info.device_id << \",\" << std::endl;
            file << \"      \\\"name\\\": \\\"\" << info.name << \"\\\",\" << std::endl;
            file << \"      \\\"total_memory\\\": \" << info.total_memory << \",\" << std::endl;
            file << \"      \\\"compute_capability\\\": \\\"\" << info.compute_capability_major 
                 << \".\" << info.compute_capability_minor << \"\\\",\" << std::endl;
            file << \"      \\\"multiprocessor_count\\\": \" << info.multiprocessor_count << \",\" << std::endl;
            file << \"      \\\"max_threads_per_block\\\": \" << info.max_threads_per_block << std::endl;
            file << \"    }\";
            if (i < gpu_infos_.size() - 1) file << \",\";
            file << std::endl;
        }
        
        file << \"  ]\" << std::endl;
        file << \"}\" << std::endl;
        file.close();
        
        if (debug_mode_) {
            std::cout << \"GPU configuration saved to \" << gpu_config_file_ << std::endl;
        }
    }
    
    // 현재 GPU 정보 반환
    const GPUInfo* getCurrentGPUInfo() const {
        if (current_gpu_ >= 0 && current_gpu_ < gpu_infos_.size()) {
            return &gpu_infos_[current_gpu_];
        }
        return nullptr;
    }
    
    // 모든 GPU 정보 반환
    const std::vector<GPUInfo>& getAllGPUInfo() const {
        return gpu_infos_;
    }
    
    // 현재 GPU ID 반환
    int getCurrentGPU() const {
        return current_gpu_;
    }
    
    // 총 GPU 수 반환
    int getTotalGPUs() const {
        return total_gpus_;
    }
    
    // GPU 메모리 사용량 업데이트
    void updateMemoryInfo() {
        if (current_gpu_ >= 0) {
            cudaSetDevice(current_gpu_);
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            
            if (current_gpu_ < gpu_infos_.size()) {
                gpu_infos_[current_gpu_].free_memory = free_mem;
                gpu_infos_[current_gpu_].total_memory = total_mem;
            }
        }
    }
    
    // GPU 상태 출력
    void printGPUStatus() {
        updateMemoryInfo();
        
        std::cout << \"=== GPU Status (Rank \" << mpi_rank_ << \") ===\" << std::endl;
        std::cout << \"Current GPU: \" << current_gpu_ << std::endl;
        
        if (current_gpu_ >= 0 && current_gpu_ < gpu_infos_.size()) {
            const auto& info = gpu_infos_[current_gpu_];
            std::cout << \"Name: \" << info.name << std::endl;
            std::cout << \"Memory: \" << (info.free_memory / (1024*1024)) << \" MB free / \" 
                     << (info.total_memory / (1024*1024)) << \" MB total\" << std::endl;
            std::cout << \"Compute Capability: \" << info.compute_capability_major 
                     << \".\" << info.compute_capability_minor << std::endl;
            std::cout << \"Multiprocessors: \" << info.multiprocessor_count << std::endl;
        }
        std::cout << \"=========================\" << std::endl;
    }
    
    // 정리
    void cleanup() {
        if (current_gpu_ >= 0) {
            cudaDeviceReset();
        }
    }
    
    // 디버그 모드 설정
    void setDebugMode(bool debug) {
        debug_mode_ = debug;
    }
};

#endif // MULTI_GPU_MANAGER_HPP