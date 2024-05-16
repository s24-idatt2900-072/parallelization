cmake_minimum_required(VERSION 3.18)  # Minimum version that supports CUDA natively

project(CudaCosineMaxPool LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

set(CMAKE_CUDA_ARCHITECTURES 80)

#cmake_policy(SET CMP0135 NEW)  # Ensure the timestamp of extracted files is the time of extraction
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)  # Ignore compatibility with CMake < 3.5 in GoogleTest
#cmake_policy(SET CMP0148 NEW)

# Add executable
add_executable(${PROJECT_NAME} src/utilities.cpp src/main.cpp src/cuda_kernels.cu src/cuda_operations.cu)

# Specify libraries to link (if any)
target_link_libraries(${PROJECT_NAME} cuda)

# Optional: specify additional flags for CUDA
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Setup testing
enable_testing()

# Include GoogleTest, pulling it from GitHub as needed
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
FetchContent_MakeAvailable(googletest)

# Add test executable
add_executable(${PROJECT_NAME}_tests
    test/src/test_cuda_kernels.cu
    src/cuda_kernels.cu
    src/utilities.cpp
)

target_link_libraries(${PROJECT_NAME}_tests gtest_main cuda)

set_target_properties(${PROJECT_NAME}_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add the tests to the project
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_tests)