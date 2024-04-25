cmake_minimum_required(VERSION 3.18)  # Minimum version that supports CUDA natively

project(DotProduct LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Optionally, set the CUDA architecture (depending on your GPU model)
# set(CMAKE_CUDA_ARCHITECTURES 72)  # Example for NVIDIA Tesla V100

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=sm_80")

set(CMAKE_CUDA_ARCHITECTURES 80)

# Add executable
add_executable(${PROJECT_NAME} src/utilities.cpp src/main.cpp src/cuda_kernels.cu src/cuda_operations.cu)

# Specify libraries to link (if any)
target_link_libraries(${PROJECT_NAME} cuda)

# Optional: specify additional flags for CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
