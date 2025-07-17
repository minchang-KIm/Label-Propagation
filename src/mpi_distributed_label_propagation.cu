#include \"../include/mpi_distributed_label_propagation.hpp\"
#include \"../include/multi_gpu_manager.hpp\"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace MPIDistributed {

// ============================================================================
// Phase 2: 2-Phase Workflow CUDA 커널 구현
// ============================================================================

// Step 4: Dynamic Unweighted Label Propagation on Boundary Vertices Only
__global__ void dynamicLabelPropagationKernel(
    const BoundaryVertex* boundary_vertices,
    const NeighborVertex* neighbor_vertices,
    const PartitionInfo* partition_info,
    int* vertex_partitions,
    int* changed_vertices,
    int* pu_ro_candidates,
    int num_boundary_vertices,
    int num_partitions,
    int* debug_info) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boundary_vertices) return;
    
    // 디버깅 정보 초기화
    if (idx == 0 && debug_info) {
        debug_info[0] = blockDim.x;  // block size
        debug_info[1] = gridDim.x;   // grid size
        debug_info[2] = num_boundary_vertices; // total vertices
    }
    
    const BoundaryVertex& bv = boundary_vertices[idx];
    int current_partition = bv.current_partition;
    
    // Phase2: Dynamic unweighted label propagation
    // 경계 정점에서만 이웃들의 라벨 점수 계산
    float* partition_scores = new float[num_partitions];
    for (int p = 0; p < num_partitions; p++) {
        partition_scores[p] = 0.0f;
    }
    
    // 이웃 정점들로부터 점수 수집 (unweighted)
    for (int i = 0; i < bv.degree; i++) {
        int neighbor = bv.neighbors[i];
        if (neighbor >= 0) { // 유효한 이웃
            int neighbor_partition = vertex_partitions[neighbor];
            if (neighbor_partition >= 0 && neighbor_partition < num_partitions) {
                partition_scores[neighbor_partition] += 1.0f; // Unweighted
            }
        }
    }
    
    // 최적 파티션 선택
    int best_partition = current_partition;
    float best_score = partition_scores[current_partition];
    
    for (int p = 0; p < num_partitions; p++) {
        if (partition_scores[p] > best_score) {
            best_score = partition_scores[p];
            best_partition = p;
        }
    }
    
    // 라벨 변경 감지 및 PU_RO 후보 표시
    if (best_partition != current_partition) {
        vertex_partitions[bv.vertex_id] = best_partition;
        changed_vertices[idx] = bv.vertex_id;
        if (pu_ro_candidates && bv.vertex_id >= 0) {
            pu_ro_candidates[bv.vertex_id] = 1; // PU_RO 후보로 표시
        }
    } else {
        changed_vertices[idx] = -1; // 변경 없음
    }
    
    delete[] partition_scores;
}

// Edge-cut 계산 커널 (개선된 버전)
__global__ void computeEdgeCutKernel(
    int* row_ptr, int* col_indices, int* vertex_partitions,
    int* edge_cut_results, int num_vertices, int* debug_info) {
    
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    
    // 디버깅 정보
    if (v == 0 && debug_info) {
        debug_info[3] = blockDim.x;  // block size for edge cut
        debug_info[4] = gridDim.x;   // grid size for edge cut
    }
    
    int local_edge_cut = 0;
    int vertex_partition = vertex_partitions[v];
    
    // 경계 검사 강화
    if (vertex_partition < 0 || row_ptr[v] < 0 || row_ptr[v+1] < row_ptr[v]) {
        edge_cut_results[v] = 0;
        return;
    }
    
    for (int i = row_ptr[v]; i < row_ptr[v + 1]; i++) {
        int neighbor = col_indices[i];
        if (neighbor >= 0 && neighbor < num_vertices) {
            int neighbor_partition = vertex_partitions[neighbor];
            if (neighbor_partition >= 0 && neighbor_partition != vertex_partition) {
                local_edge_cut++;
            }
        }
    }
    
    edge_cut_results[v] = local_edge_cut;
}

// GPU 메모리 검사 커널
__global__ void memoryCheckKernel(int* test_array, int size, int* status) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // 메모리 읽기/쓰기 테스트
    test_array[idx] = idx;
    __syncthreads();
    
    if (test_array[idx] != idx) {
        atomicAdd(status, 1); // 오류 카운트
    }
}

} // namespace MPIDistributed

// ============================================================================
// Phase 2: 2-Phase Workflow MPI+CUDA 구현
// ============================================================================

using namespace MPIDistributed;

// 생성자
MPIDistributedLabelPropagation::MPIDistributedLabelPropagation(MPI_Comm comm)
    : comm_(comm), num_vertices_(0), num_edges_(0), num_partitions_(4),
      rank_(0), size_(0), device_id_(0), stream_(0),
      d_row_ptr_(nullptr), d_col_indices_(nullptr), d_values_(nullptr),
      d_vertex_partitions_(nullptr), d_boundary_nodes_(nullptr),
      d_penalty_scores_(nullptr), d_label_changes_(nullptr),
      d_label_scores_(nullptr), d_candidate_labels_(nullptr),
      d_local_labels_(nullptr), graph_(nullptr),
      total_computation_time_(0.0), total_communication_time_(0.0),
      boundary_processing_time_(0.0), total_iterations_(0) {
    
    // MPI 정보 가져오기
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
    
    // Multi-GPU 관리자 초기화
    gpu_manager_ = std::make_unique<MultiGPUManager>(true); // 디버그 모드
    
    // EdgeCutMetrics 초기화
    edge_cut_metrics_.current_edge_cut = 0;
    edge_cut_metrics_.previous_edge_cut = 0;
    edge_cut_metrics_.edge_cut_ratio = 0.0;
    edge_cut_metrics_.improvement_ratio = 0.0;
    edge_cut_metrics_.change_ratio = 0.0;
    edge_cut_metrics_.iteration = 0;
    edge_cut_metrics_.convergence_count = 0;
    
    // ImbalanceMetrics 초기화
    imbalance_metrics_.RV = 0.0;
    imbalance_metrics_.RE = 0.0;
    imbalance_metrics_.max_load = 0.0;
    imbalance_metrics_.avg_load = 0.0;
}

// 소멸자
MPIDistributedLabelPropagation::~MPIDistributedLabelPropagation() {
    Cleanup();
}

// 초기화
bool MPIDistributedLabelPropagation::Initialize() {
    if (rank_ == 0) {
        std::cout << \"=== Phase2 2-Phase Workflow MPI+CUDA 초기화 ===\" << std::endl;
        std::cout << \"프로세스 수: \" << size_ << std::endl;
    }
    
    // Multi-GPU 관리자로 GPU 초기화
    if (!gpu_manager_->initializeGPUs()) {
        std::cerr << \"Rank \" << rank_ << \": Failed to initialize GPUs\" << std::endl;
        return false;
    }
    
    device_id_ = gpu_manager_->getCurrentGPU();
    
    // CUDA 스트림 생성
    CUDA_CHECK(cudaStreamCreate(&stream_));
    
    if (rank_ == 0) {
        gpu_manager_->printGPUStatus();
        std::cout << \"=== 초기화 완료 ===\" << std::endl;
    }
    
    return true;
}

// 정리
void MPIDistributedLabelPropagation::Cleanup() {
    FreeGPUMemory();
    
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = 0;
    }
    
    if (graph_) {
        delete graph_;
        graph_ = nullptr;
    }
    
    if (gpu_manager_) {
        gpu_manager_->cleanup();
    }
}