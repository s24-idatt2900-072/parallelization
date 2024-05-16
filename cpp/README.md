# CUDA Cosine Similarity and Max Pooling

This project demonstrates the use of CUDA to perform cosine similarity and max pooling operations on large datasets efficiently. 
The main functionalities include manual cosine similarity and max pooling, as well as a research run to benchmark and analyze performance.


## Prerequisites

- CUDA Toolkit
- CMake (minimum version 3.18)
- C++17 compatible compiler


## CUDA Architecture Optimization

This project is optimized for the following CUDA architectures:

- CUDA Compute Capability 8.0: This corresponds to the NVIDIA Ampere architecture, which includes GPUs such as the NVIDIA A100 and RTX 30 series.

If you want to support multiple architectures, you can modify the CMakeLists.txt file to include other architectures as needed.

## Project Structure

- `include/`: Contains header files.
- `src/`: Contains the source code files.
- `data/mnist/`: Contains the MNIST image data.
- `data/filters/`: Contains the filter data.
- `test/`: Contains test files and test data.

## Building the Project

1. Clone the repository:
```sh
   git clone https://github.com/s24-idatt2900-072/parallelization.git
   cd cpp
```

2. Create a build directory and navigate into it:
```bash
    mkdir build
    cd build
```

3. Run CMake to configure the project:
```bash
    cmake ..
```

4. Build the project using make:
```bash
    make
```

## Running the Project

After building the project, you can run the executable:

```bash
    ./CudaCosineMaxPool
```

You will be presented with a menu to select from the available options:

1. Run with manual input
2. Run research
3. Exit

Follow the prompts to input the necessary parameters for each run.

## Testing the Project

The project includes unit tests for validating the correctness of the cosine similarity and max pooling operations. The tests are written using GoogleTest and can be built and run as follows:

1. After building the main project, run the tests with:

```bash
    ./CudaCosineMaxPool_tests
```