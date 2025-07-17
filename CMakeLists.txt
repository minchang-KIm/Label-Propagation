cmake_minimum_required(VERSION 3.16)
project(WeightedLabelPropagation LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build Type Configuration - 디버그/릴리즈 모드 설정
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING 
        "Choose the type of build: Debug, Release, RelWithDebInfo, MinSizeRel" FORCE)
endif()

# Debug Mode Configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "🐛 DEBUG MODE - Full Logging and Debugging Enabled")
    add_definitions(-DDEBUG_MODE=1)
    add_definitions(-DENABLE_DETAILED_LOGGING=1)
    add_definitions(-DENABLE_FORMULA_VALIDATION=1)
    add_definitions(-DENABLE_GPU_MONITORING=1)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
    
# Release Mode Configuration  
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "🚀 RELEASE MODE - High Performance Optimized")
    add_definitions(-DDEBUG_MODE=0)
    add_definitions(-DENABLE_DETAILED_LOGGING=0)
    add_definitions(-DENABLE_FORMULA_VALIDATION=0)
    add_definitions(-DENABLE_GPU_MONITORING=1)  # GPU 모니터링은 유지
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")
    
# RelWithDebInfo Mode - 최적화 + 디버그 정보
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "⚡ RELEASE WITH DEBUG INFO - Optimized + Debug Symbols")
    add_definitions(-DDEBUG_MODE=0)
    add_definitions(-DENABLE_DETAILED_LOGGING=0)
    add_definitions(-DENABLE_FORMULA_VALIDATION=0)
    add_definitions(-DENABLE_GPU_MONITORING=1)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
endif()

# Set CUDA architecture - will be dynamically optimized at runtime
# Support multiple architectures for portability
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

# Find CUDA - use modern CMake CUDA support (no legacy FindCUDA)
enable_language(CUDA)

# Find MPI
find_package(MPI REQUIRED)

# CUDA include directories
include_directories(/usr/local/cuda/include)
include_directories(/usr/local/cuda-12.1/targets/x86_64-linux/include)
include_directories(include)

# MPI include directories
include_directories(${MPI_INCLUDE_PATH})

# Project information
message(STATUS "Building Pure CUDA Weighted Label Propagation")
message(STATUS "Dynamic CUDA system detection and optimization")
message(STATUS "Automatically adapts to available GPU hardware")
message(STATUS "")

# Include directories
include_directories(include)

# Source files (Enhanced with weighted label propagation and debug logging)
set(LIBRARY_SOURCES
    src/boundary_node_extractor.cu
    src/weighted_label_propagation_kernel.cu
    src/weighted_label_prop_debug.cu
    src/graph_loader.cu
    src/graph_partitioning_metrics.cu
    src/cuda_system_detector.cu
)

# MPI Distributed sources
set(MPI_DISTRIBUTED_SOURCES
    ${LIBRARY_SOURCES}
    src/mpi_distributed_label_propagation.cu
    src/mpi_distributed_main.cu
)

set(MAIN_SOURCES
    ${LIBRARY_SOURCES}
    src/main.cu
)

set(DEBUG_SOURCES
    ${LIBRARY_SOURCES}
    src/main_debug.cu
)

# Header files (Enhanced with debug logging system)
set(HEADERS
    include/weighted_label_propagation.hpp
    include/weighted_label_prop.hpp
    include/boundary_node_extractor.hpp
    include/weighted_label_propagation_kernel.hpp
    include/graph_loader.hpp
    include/graph_partitioning_metrics.hpp
    include/cuda_system_detector.hpp
    include/multi_gpu_processor.hpp
    include/debug_logger.hpp
    include/mpi_distributed_label_propagation.hpp
)

# Create main executable - Pure CUDA (production version)
add_executable(cuda_weighted_label_prop ${MAIN_SOURCES})
target_link_libraries(cuda_weighted_label_prop cudart)

# Create debug executable for testing and development
add_executable(cuda_weighted_label_prop_debug ${DEBUG_SOURCES})
target_link_libraries(cuda_weighted_label_prop_debug cudart)

# Create library for potential future use
add_library(weighted_label_prop ${LIBRARY_SOURCES} ${HEADERS})
target_link_libraries(weighted_label_prop cudart)

# Pure CUDA compilation settings for maximum performance  
set_property(TARGET weighted_label_prop PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_weighted_label_prop PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(weighted_label_prop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -O3
    --use_fast_math
    -lineinfo
    --ptxas-options=-v
>)

target_compile_options(cuda_weighted_label_prop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -O3
    --use_fast_math
    -lineinfo
    --ptxas-options=-v
>)

target_compile_options(cuda_weighted_label_prop_debug PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -O3
    --use_fast_math
    -lineinfo
    --ptxas-options=-v
>)

# C++ compilation settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(weighted_label_prop PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -O3 -march=native
    >)
    target_compile_options(cuda_weighted_label_prop PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -O3 -march=native
    >)
    target_compile_options(cuda_weighted_label_prop_debug PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -O3 -march=native
    >)
endif()

# Installation
install(TARGETS weighted_label_prop cuda_weighted_label_prop cuda_weighted_label_prop_debug
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# ============================================================================
# MPI + CUDA Distributed Target
# ============================================================================

# Create MPI distributed executable
add_executable(mpi_distributed_label_prop ${MPI_DISTRIBUTED_SOURCES})

# MPI + CUDA properties
set_target_properties(mpi_distributed_label_prop PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
    CXX_STANDARD 14
    CUDA_STANDARD 14
)

# MPI linking
target_link_libraries(mpi_distributed_label_prop ${MPI_LIBRARIES})
if(MPI_COMPILE_FLAGS)
    set_target_properties(mpi_distributed_label_prop PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()
if(MPI_LINK_FLAGS)
    set_target_properties(mpi_distributed_label_prop PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

# CUDA compilation settings for MPI target
target_compile_options(mpi_distributed_label_prop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -O3
    --use_fast_math
    -lineinfo
    --ptxas-options=-v
>)

# C++ compilation settings for MPI target
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mpi_distributed_label_prop PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -O3 -march=native
    >)
endif()

# Install MPI target
install(TARGETS mpi_distributed_label_prop
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS} DESTINATION include/weighted_label_propagation)

# Print configuration summary
message(STATUS "")
message(STATUS "Pure CUDA + MPI Distributed Configuration Summary:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA_ENABLED: TRUE (Pure CUDA with Dynamic Detection)")
message(STATUS "  MPI_ENABLED: TRUE (Distributed Multi-Cluster Support)")
message(STATUS "  CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  AUTO_GPU_DETECTION: Enabled")
message(STATUS "  RUNTIME_OPTIMIZATION: Automatic")
message(STATUS "  MPI_LIBRARIES: ${MPI_LIBRARIES}")
message(STATUS "")
message(STATUS "  Build Mode Features:")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "    DEBUG_MODE: Full logging and validation enabled")
    message(STATUS "    PERFORMANCE: Development focus (slower but detailed)")
    message(STATUS "    LOG_LEVEL: Maximum detail with formula validation")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "    RELEASE_MODE: High-performance optimized")
    message(STATUS "    PERFORMANCE: Production ready (fastest)")
    message(STATUS "    LOG_LEVEL: Minimal essential logging only")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "    HYBRID_MODE: Optimized with debug symbols")
    message(STATUS "    PERFORMANCE: Fast with profiling capability")
    message(STATUS "    LOG_LEVEL: Moderate logging for analysis")
endif()
message(STATUS "")