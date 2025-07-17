#include \"../include/mpi_distributed_label_propagation.hpp\"
#include <iostream>

int main(int argc, char** argv) {
    // MPI 초기화
    MPI_Init(&argc, &argv);
    
    if (argc < 3) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << \"Usage: \" << argv[0] << \" <graph_path> <num_partitions>\" << std::endl;
            std::cout << \"Example: \" << argv[0] << \" ../../datasets/ljournal-2008 4\" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    std::string graph_path = argv[1];
    int num_partitions = std::atoi(argv[2]);
    
    try {
        // Phase2 2-Phase Workflow 실행
        MPIDistributed::MPIDistributedLabelPropagation phase2_workflow;
        
        // 초기화
        if (!phase2_workflow.Initialize()) {
            std::cerr << \"Failed to initialize Phase2 workflow\" << std::endl;
            MPI_Finalize();
            return 1;
        }
        
        // 그래프 로딩
        if (!phase2_workflow.LoadGraph(graph_path, num_partitions)) {
            std::cerr << \"Failed to load graph: \" << graph_path << std::endl;
            MPI_Finalize();
            return 1;
        }
        
        // Phase2 실행 (Phase1이 완료된 상태에서)
        if (!phase2_workflow.RunPhase2Algorithm(50, 0.001)) { // 최대 50회 반복, 0.1% 허용 오차
            std::cerr << \"Phase2 algorithm failed\" << std::endl;
            MPI_Finalize();
            return 1;
        }

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << \"Phase2 2-Phase Workflow 완료!\" << std::endl;
            phase2_workflow.PrintResults();
        }
        
    } catch (const std::exception& e) {
        std::cerr << \"Error: \" << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    MPI_Finalize();
    return 0;
}