#include "cuda_util.h"
#include "common.h"
#include "cuda_fp16.h"

void runCudaOperations(float* a, float* b, float* out, size_t a_size, size_t b_size, size_t out_size, unsigned int ilen, unsigned int alen, unsigned int blen) {
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, a_size);
    cudaMalloc(&d_b, b_size);
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((alen + 15) / 16, (blen + 15) / 16);

    // start measuring time of computing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_out, ilen, alen, blen);
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_out, ilen, alen, blen);
    cudaDeviceSynchronize();

    // stop measuring time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds.\n";

    cudaError_t error = cudaGetLastError();
    checkCudaError(error);

    cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);

    error = cudaMalloc(&d_a, a_size);
    checkCudaError(error);

    error = cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice);
    checkCudaError(error);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

void getSystemInformation() {
    int device;
    cudaGetDevice(&device); // get current device ID
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device); // get device properties
    std::cout << "GPU Name: " << properties.name << std::endl;
    std::cout << "Compute capability: " << properties.major << "." << properties.minor << std::endl;
    std::cout << "Total global memory: " << properties.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared memory per block: " << properties.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Registers per block: " << properties.regsPerBlock << std::endl;
    std::cout << "Warp size: " << properties.warpSize << std::endl;
    std::cout << "Memory Clock Rate (KHz): " << properties.memoryClockRate << std::endl;
    std::cout << "Memory Bus Width (bits): " << properties.memoryBusWidth << std::endl;
    if (properties.memoryBusWidth != 0) {
        float memoryBandwidth = 2.0 * properties.memoryClockRate * (properties.memoryBusWidth / 8) / 1.0e6;
        std::cout << "Theoretical Memory Bandwidth (GB/s): " << memoryBandwidth << std::endl;
    }
}