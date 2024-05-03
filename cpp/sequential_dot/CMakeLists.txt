cmake_minimum_required(VERSION 3.10)
project(ImageProcessing)

# Set the C++ standard for the project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If using any specific compiler flags, like optimizations or warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Add directories to include header files
include_directories(include)

# Find packages for any libraries you need
# If you're using any parallelization libraries like OpenMP, uncomment the following lines:
# find_package(OpenMP REQUIRED)
# if(OPENMP_FOUND)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

# Add source files
set(SOURCE_FILES src/main.cpp src/operations.cpp src/utilities.cpp)

# Specify the target executable and its source files
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# If linking against libraries like OpenMP, uncomment the following line:
# target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
