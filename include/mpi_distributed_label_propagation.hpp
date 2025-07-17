#ifndef MPI_DISTRIBUTED_LABEL_PROPAGATION_HPP
#define MPI_DISTRIBUTED_LABEL_PROPAGATION_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>

// Forward declaration
class MultiGPUManager;

// CUDA 오류 확인 매크로
#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t error = call; \\
        if (error != cudaSuccess) { \\
            fprintf(stderr, \"CUDA error at %s:%d - %s\\n\", __FILE__, __LINE__, \\
                    cudaGetErrorString(error)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

namespace MPIDistributed {

// 2-Phase Workflow를 위한 데이터 구조들
struct BoundaryVertex {
    int vertex_id;
    int current_partition;
    int degree;
    int* neighbors;      // GPU용 포인터
    double* weights;     // GPU용 포인터
};

struct BoundaryVertexHost {
    int vertex_id;
    int current_partition;
    int degree;
    std::vector<int> neighbors;
    std::vector<double> weights;
};

struct NeighborVertex {
    int vertex_id;
    int partition_id;
    double weight;
};

struct PartitionInfo {
    int partition_id;
    int size;
    int vertex_count;
    int edge_count;
    double load_balance;
    double edge_cut_ratio;
    double RV;
    double RE;
    double imbalance_factor;
    double penalty_score; // Phase2용 추가
};

struct PartitionUpdate {
    int vertex_id;
    int old_partition;      // Phase2용 변경
    int new_partition;      // Phase2용 변경
    int source_rank;        // Phase2용 추가
    double improvement_score;
    std::vector<PartitionUpdate> to_send;                // Phase2용 수정
    std::vector<PartitionUpdate> received_from_others;   // Phase2용 수정
    std::vector<PartitionUpdate> received_from_own;      // Phase2용 수정
};

// 간단한 Graph 클래스 (Phase2용)
class Graph {
public:
    Graph() {}
    ~Graph() {}
    
    int GetRowStart(int v) const { return 0; } // 스텁
    int GetRowEnd(int v) const { return 0; }   // 스텁
    int GetColumn(int i) const { return 0; }   // 스텁
    double GetValue(int i) const { return 1.0; } // 스텁
};

struct EdgeCutMetrics {
    int total_edge_cut;
    int current_edge_cut;
    int previous_edge_cut;
    double edge_cut_ratio;
    double improvement_ratio;
    double change_ratio;
    int iteration;
    int convergence_count;
    
    static constexpr double EPSILON = 0.03;
    static constexpr int MAX_CONVERGENCE_COUNT = 10;
};

struct ImbalanceMetrics {
    double RV; // Vertex imbalance ratio
    double RE; // Edge imbalance ratio
    double max_load;
    double avg_load;
};

class MPIDistributedLabelPropagation {
private:
    // MPI 관련 변수들
    int rank_;
    int size_;
    MPI_Comm comm_;
    
    // CUDA 관련 변수들
    int device_id_;
    cudaStream_t stream_;
    std::unique_ptr<MultiGPUManager> gpu_manager_;
    
    // 그래프 데이터
    Graph* graph_;          // Phase2용 추가
    int num_vertices_;
    int num_edges_;
    int num_partitions_;
    
    // CPU 데이터
    std::vector<int> row_ptr_;
    std::vector<int> col_indices_;
    std::vector<double> values_;
    std::vector<int> vertex_partitions_;
    std::vector<int> boundary_nodes_;
    std::vector<double> penalty_scores_;
    std::vector<int> partition_sizes_;
    
    // GPU 데이터 포인터들
    int* d_row_ptr_;
    int* d_col_indices_;
    double* d_values_;
    int* d_vertex_partitions_;
    int* d_boundary_nodes_;
    double* d_penalty_scores_;
    int* d_label_changes_;
    double* d_label_scores_;
    int* d_candidate_labels_;
    int* d_local_labels_;
    
    // 메트릭 관련 변수들
    EdgeCutMetrics edge_cut_metrics_;
    ImbalanceMetrics imbalance_metrics_;
    std::vector<PartitionInfo> partition_info_;
    
    // 성능 측정
    double total_computation_time_;
    double total_communication_time_;
    double boundary_processing_time_;
    int total_iterations_;
    
public:
    MPIDistributedLabelPropagation(MPI_Comm comm = MPI_COMM_WORLD);
    ~MPIDistributedLabelPropagation();
    
    // 초기화 및 정리
    bool Initialize();
    void Cleanup();
    
    // 데이터 로딩
    bool LoadGraph(const std::string& filename, int num_partitions = 4);
    bool LoadPartitionedGraph(const std::string& graph_file, 
                            const std::string& partition_file, 
                            int num_partitions);
    
    // Phase2 2-Phase Workflow 실행
    bool RunPhase2Algorithm(int max_iterations = 100, double tolerance = 0.001);
    
    // 결과 저장 및 출력
    void SaveResults(const std::string& output_file);
    void PrintResults();
    void PrintMetrics();
    
    // Phase2 7-step 알고리즘 구현
    void Phase2Step1_BoundaryVertexIdentification();
    void Phase2Step2_NeighborPartitionAnalysis();
    void Phase2Step3_CandidateLabelGeneration();
    void Phase2Step4_LocalLabelScoreComputation();
    void Phase2Step5_GlobalLabelInformationExchange();
    void Phase2Step6_OptimalLabelSelection();
    void Phase2Step7_LabelUpdateAndSynchronization();
    
    // 메트릭 계산
    void CalculateEdgeCutMetrics();
    void CalculateImbalanceMetrics();
    void UpdatePartitionInfo();
    
    // GPU 메모리 관리
    bool AllocateGPUMemory();
    void FreeGPUMemory();
    bool CopyDataToGPU();
    bool CopyDataFromGPU();
    
    // 유틸리티 함수들
    bool IsConverged(double tolerance = 0.001);
    void LogIterationResults(int iteration);
    void ValidateResults();
    
    // 접근자 함수들
    const std::vector<int>& GetVertexPartitions() const { return vertex_partitions_; }
    const EdgeCutMetrics& GetEdgeCutMetrics() const { return edge_cut_metrics_; }
    const ImbalanceMetrics& GetImbalanceMetrics() const { return imbalance_metrics_; }
    int GetTotalIterations() const { return total_iterations_; }
    double GetTotalComputationTime() const { return total_computation_time_; }
    double GetTotalCommunicationTime() const { return total_communication_time_; }
};

} // namespace MPIDistributed

#endif // MPI_DISTRIBUTED_LABEL_PROPAGATION_HPP