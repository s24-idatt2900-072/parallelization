#include <iostream>

__global__ void dotProductKernel(float *a, float *b, float *out, unsigned int ilen, unsigned int alen, unsigned int blen) {
    unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    if(tidX >= alen || tidY >= blen) return;

    float dot = 0.0;
    for(unsigned int i = 0; i < ilen; ++i) {
        unsigned int a_index = tidX * ilen + i;
        unsigned int b_index = tidY * ilen + i;
        dot += a[a_index] * b[b_index];
    }

    unsigned int out_index = tidX * blen + tidY;
    out[out_index] = dot;
}

int main() {
    unsigned int alen = 5000;
    unsigned int blen = 100000;
    unsigned int ilen = 841;

    srand(123);

    size_t a_size = alen * ilen * sizeof(float); // alen * 841 floats
    size_t b_size = blen * ilen * sizeof(float); // blen * 841 floats
    size_t out_size = alen * blen * sizeof(float); // alen * blen floats

    float *a, *b, *out; // host arrays
    float *d_a, *d_b, *d_out; // device arrays

    // 1. GEN IMAGE FILTER
    // Allocate host memory and initialize
    //a = new float[alen * ilen] {1,2,3, 1,2,3};
    //b = new float[blen * ilen] {6,5,4, 1,2,3};


    // 2. GEN IMAGE FILTER
    //a = new float[alen * ilen];
    //std::fill_n(a, alen * ilen, 1.0f);

    //b = new float[blen * ilen];
    //std::fill_n(b, blen * ilen, 1.0f);

    // 2. GEN IMAGE FILTER
    a = new float[alen * ilen];

    b = new float[blen * ilen];

    for (size_t i = 0; i < alen * ilen; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (size_t i = 0; i < blen * ilen; ++i) {
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    out = new float[alen * blen];

    int device;
    cudaGetDevice(&device); // get current device ID
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device); // get device properties
    std::cout << "GPU: " << properties.name << std::endl;

    // allocate device memory
    cudaMalloc(&d_a, a_size);
    cudaMalloc(&d_b, b_size);
    cudaMalloc(&d_out, out_size);

    // copy data from host to device
    cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice);

    // define grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((alen + 15) / 16, (blen + 15) / 16);

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // launch the kernel
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_out, ilen, alen, blen);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(error) << std::endl;
    }

    // copy the result back to the host
    cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);

    error = cudaMalloc(&d_a, a_size);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error during cudaMalloc for d_a: " << cudaGetErrorString(error) << std::endl;
    }

    error = cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error during cudaMemcpy for d_a: " << cudaGetErrorString(error) << std::endl;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds.\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Checking Dot Product Results...\n";
    bool resultsPlausible = true;
    for (unsigned int i = 0; i < alen * blen; ++i) {
        if (out[i] < 0 || out[i] > ilen) {
            std::cout << "Mismatch at index " << i << ": Result is " << out[i] << ".\n";
            resultsPlausible = false;
            break;
        }
    }
    if (resultsPlausible) {
        std::cout << "All results are within the plausible range.\n";
    }

    // clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    delete[] a;
    delete[] b;
    delete[] out;

}

