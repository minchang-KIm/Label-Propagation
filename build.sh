#!/bin/bash

# Multi-GPU Graph Partitioning Build Script
# Automated build system for CUDA + MPI label propagation

PROJECT_NAME=\"weighted_label_propagation\"
BUILD_DIR=\"build\"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[0;33m'
BLUE='\\033[0;34m'
PURPLE='\\033[0;35m'
CYAN='\\033[0;36m'
NC='\\033[0m' # No Color

print_header() {
    echo -e \"${CYAN}========================================${NC}\"
    echo -e \"${CYAN}  Multi-GPU Graph Partitioning Builder${NC}\"
    echo -e \"${CYAN}========================================${NC}\"
}

print_usage() {
    echo -e \"${YELLOW}Usage: $0 [debug|release|profile|clean]${NC}\"
    echo -e \"${YELLOW}Build modes:${NC}\"
    echo -e \"  ${GREEN}debug${NC}   - Full logging and debugging (slower)\"
    echo -e \"  ${GREEN}release${NC} - High performance optimized (fastest)\"
    echo -e \"  ${GREEN}profile${NC} - Optimized with debug symbols\"
    echo -e \"  ${GREEN}clean${NC}   - Clean build directory\"
}

check_dependencies() {
    echo -e \"${BLUE}🔍 Checking dependencies...${NC}\"
    
    # Check CUDA
    if ! command -v nvcc &> /dev/null; then
        echo -e \"${RED}❌ CUDA not found. Please install CUDA Toolkit.${NC}\"
        exit 1
    fi
    
    # Check MPI
    if ! command -v mpirun &> /dev/null; then
        echo -e \"${RED}❌ MPI not found. Please install OpenMPI.${NC}\"
        exit 1
    fi
    
    # Check CMake
    if ! command -v cmake &> /dev/null; then
        echo -e \"${RED}❌ CMake not found. Please install CMake 3.16+.${NC}\"
        exit 1
    fi
    
    echo -e \"${GREEN}✓ All dependencies found${NC}\"
}

configure_build() {
    local mode=$1
    local cmake_build_type=\"\"
    local description=\"\"
    
    case $mode in
        \"debug\")
            cmake_build_type=\"Debug\"
            description=\"🐛 디버그 모드 - 전체 로깅 및 검증\"
            ;;
        \"release\")
            cmake_build_type=\"Release\"
            description=\"🚀 릴리즈 모드 - 고성능 최적화\"
            ;;
        \"profile\")
            cmake_build_type=\"RelWithDebInfo\"
            description=\"⚡ 프로파일 모드 - 최적화 + 디버그 심볼\"
            ;;
        *)
            echo -e \"${RED}❌ Invalid build mode: $mode${NC}\"
            print_usage
            exit 1
            ;;
    esac
    
    echo -e \"${PURPLE}📦 Configuring build: $description${NC}\"
    
    # Create and enter build directory
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    
    # Configure with CMake
    cmake -DCMAKE_BUILD_TYPE=$cmake_build_type .. || {
        echo -e \"${RED}❌ CMake configuration failed${NC}\"
        exit 1
    }
    
    cd ..
}

build_project() {
    echo -e \"${BLUE}🔨 Building project...${NC}\"
    
    cd $BUILD_DIR
    make -j$(nproc) || {
        echo -e \"${RED}❌ Build failed${NC}\"
        exit 1
    }
    cd ..
    
    echo -e \"${GREEN}✅ Build completed successfully!${NC}\"
}

clean_build() {
    echo -e \"${YELLOW}🧹 Cleaning build directory...${NC}\"
    rm -rf $BUILD_DIR
    echo -e \"${GREEN}✅ Clean completed${NC}\"
}

show_results() {
    echo -e \"${CYAN}📊 Build Results:${NC}\"
    
    if [ -f \"${BUILD_DIR}/cuda_weighted_label_prop\" ]; then
        echo -e \"${GREEN}📱 Main executable: ${BUILD_DIR}/cuda_weighted_label_prop${NC}\"
    fi
    if [ -f \"${BUILD_DIR}/cuda_weighted_label_prop_debug\" ]; then
        echo -e \"${GREEN}🔍 Debug executable: ${BUILD_DIR}/cuda_weighted_label_prop_debug${NC}\"
    fi
    if [ -f \"${BUILD_DIR}/mpi_distributed_label_prop\" ]; then
        echo -e \"${GREEN}🌐 MPI distributed executable: ${BUILD_DIR}/mpi_distributed_label_prop${NC}\"
        echo -e \"${CYAN}🚀 MPI usage: mpirun -np 4 ${BUILD_DIR}/mpi_distributed_label_prop graph.mtx${NC}\"
    fi
    if [ -f \"${BUILD_DIR}/libweighted_label_prop.a\" ]; then
        echo -e \"${GREEN}📚 Library: ${BUILD_DIR}/libweighted_label_prop.a${NC}\"
    fi
}

# Main script logic
print_header

# Default to release if no argument provided
MODE=${1:-release}

case $MODE in
    \"clean\")
        clean_build
        exit 0
        ;;
    \"debug\"|\"release\"|\"profile\")
        check_dependencies
        configure_build $MODE
        build_project
        show_results
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

echo -e \"${GREEN}🎉 Build script completed successfully!${NC}\"