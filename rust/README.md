# Parallelization of Machine Learning Methods in Rust

This project aims to parallelize machine learning methods in Rust to enhance performance and scalability.

## Table of Contents
- [About](#about)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the Code](#run-the-code)
    - [Arguments](#arguments)
- [Test the Code](#test-the-code)
- [Available Packages](#available-packages)
    - [Run](#run)
    - [Test](#test)

## About
This project explores parallel computing techniques in Rust to optimize the execution of image processing using Gabor-like filters. By leveraging Rust's safety and concurrency features, the goal is to achieve significant speedups compared to traditional sequential implementations.

## Requirements
- Rust version 1.78

## Installation
To get started with the project **Clone the Repository**:
```bash
git clone https://github.com/s24-idatt2900-072/parallelization.git && cd parallelization/rust/
```
## Run the Code
To run the main files and the experiment:
```bash
cargo run
```

### Arguments
Arguments to specify what method to run:
#### Methods
- **cpu-seq**: Runs the experiment with the CPU sequentil
- **cpu**: Runs the experiment with the CPU parallel version using Rayon
- **gpu**: Runs the experiment with on the GPU with the "one image all filters" shader
- **gpu-sum**: Runs the experiment with on the GPU with the optimized "one image all filters" shader that runs wit all images and filters
- **gpu-loop**: Runs the experiment with on the GPU with the for-loop shader
- **gpu-par**: Runs the experiment with on the GPU with the parallel shader

```bash
cargo run <method> <nr-of-images> <nr-of-filters> <filter-start-and-increase> <max-pool-chunk>
```

#### Example using default values
```bash
cargo run gpu-loop
```

#### Example with all parameters
```bash
cargo run gpu-loop 100 1000 500 500
```

Add reales to run optimized version:
```bash
cargo run --release <method> <nr-of-images> <nr-of-filters> <filter-start-and-increase> <max-pool-chunk>
```

## Test the code
To run the test for the code use the following command:
```bash
cargo test
```

## Available Packages:
- **wgpu_test**: Testing and running code on the GPU using WGPU.
- **wgsl**: Library for generating WGSL code from Rust.
- **cpu_test**: Testing and running code using Rayon for CPU parallelism.
- **making_filters**: Creating filters for image processing by calling Python methods with Rust. Contains the corresponding python code.

### Run
Choose and Run a Package:
Run the desired package by using the -p flag
```bash
cargo run -p <package-directory>
```

#### Example
```bash
cargo run -p wgpu_test
```

### Test
Choose and Run Tests in a Package:
Run Tests from the desired package by using the -p flag
```bash
cargo test -p <package-directory>
```

#### Example
```bash
cargo test -p wgpu_test
```