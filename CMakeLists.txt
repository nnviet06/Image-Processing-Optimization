cmake_minimum_required(VERSION 3.10)
project(ImageProcessingOptimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit 13.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============= PHASE 1: CPU OPTIMIZATION =============

# CPU shared utilities
add_library(cpu_filters_lib src/cpu/utils.cpp)
target_link_libraries(cpu_filters_lib PRIVATE ${OpenCV_LIBS})

# Phase 1.1: Naive Gaussian
add_executable(gaussian_naive src/cpu/naive/gaussian.cpp)
target_link_libraries(gaussian_naive PRIVATE cpu_filters_lib ${OpenCV_LIBS})

# Phase 1.2: SIMD Gaussian (AVX2)
add_executable(gaussian_simd src/cpu/optimized/gaussian_simd.cpp)
target_link_libraries(gaussian_simd PRIVATE cpu_filters_lib ${OpenCV_LIBS})
if(MSVC)
    target_compile_options(gaussian_simd PRIVATE /arch:AVX2 /O3)
else()
    target_compile_options(gaussian_simd PRIVATE -mavx2 -mfma -O3)
endif()

# Phase 1.3: Threading Gaussian
add_executable(gaussian_threaded src/cpu/optimized/gaussian_threaded.cpp)
target_link_libraries(gaussian_threaded PRIVATE cpu_filters_lib Threads::Threads ${OpenCV_LIBS})

# Phase 1.4: Cache-aware Gaussian
add_executable(gaussian_cache_aware src/cpu/optimized/gaussian_cache_aware.cpp)
target_link_libraries(gaussian_cache_aware PRIVATE cpu_filters_lib ${OpenCV_LIBS})

# ============= PHASE 2: GPU BATCH PROCESSING =============

# GPU shared utilities library
add_library(gpu_filters_lib src/gpu/utils.cu)
target_link_libraries(gpu_filters_lib PRIVATE ${OpenCV_LIBS} CUDA::cudart)
set_target_properties(gpu_filters_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Phase 2.1: GPU Kernel (basic CUDA Gaussian)
add_executable(gaussian_gpu_kernel src/gpu/kernels/gaussian_kernel.cu)
target_link_libraries(gaussian_gpu_kernel PRIVATE gpu_filters_lib ${OpenCV_LIBS} CUDA::cudart)
set_target_properties(gaussian_gpu_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Phase 2.2: GPU Memory Optimization (pinned memory + streams)
add_executable(gaussian_gpu_optimized src/gpu/memory/gaussian_gpu_optimized.cu)
target_link_libraries(gaussian_gpu_optimized PRIVATE gpu_filters_lib ${OpenCV_LIBS} CUDA::cudart)
set_target_properties(gaussian_gpu_optimized PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Phase 2.3: GPU Batch Pipeline
add_executable(gaussian_gpu_batch src/gpu/batch/gaussian_batch.cu)
target_link_libraries(gaussian_gpu_batch PRIVATE gpu_filters_lib ${OpenCV_LIBS} CUDA::cudart)
set_target_properties(gaussian_gpu_batch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)