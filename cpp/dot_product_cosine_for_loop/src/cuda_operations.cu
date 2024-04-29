#include "cuda_util.h"
#include "common.h"
#include "cuda_fp16.h"


#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(error) << ", file: " << __FILE__ << ", line: " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

void runCudaOperations(float* images, float* filter_real, float* filter_abs, float* output, size_t images_size, size_t images_vector_len, size_t real_vector_len, size_t filters_size, size_t output_size, unsigned int inner_len, unsigned int image_len, unsigned int filter_len) {
    float *d_images, *d_filter_real, *d_filter_abs, *d_output;
    CUDA_CHECK(cudaMalloc(&d_images, images_size));
    CUDA_CHECK(cudaMalloc(&d_filter_real, filters_size));
    CUDA_CHECK(cudaMalloc(&d_filter_abs, filters_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));

    CUDA_CHECK(cudaMemcpy(d_images, images, images_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter_real, filter_real, filters_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter_abs, filter_abs, filters_size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((image_len + 15) / 16, (filter_len + 15) / 16);

    // start measuring time of computing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cosineSimilarityKernel<<<blocksPerGrid, threadsPerBlock>>>(d_images, d_filter_real, d_filter_abs, d_output, inner_len, image_len, images_vector_len, real_vector_len, filter_len);
    cudaDeviceSynchronize();

    // stop measuring time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    //std::cout << "Kernel execution time: " << milliseconds << " milliseconds.\n";

    CUDA_CHECK(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_images);
    cudaFree(d_filter_real);
    cudaFree(d_filter_abs);
    cudaFree(d_output);
}


/**
 * 
 * // now max pool the output with a given pool size
    // pool size is 5
    // this will also be done on the GPU
    // output is a 1D array of size image_len * filter_len
    // therefore, the max pooling processing has to take into account the chunks of image * filter length
    // also take into account if the pool size does not perfectly line up with the last chunk

 *    runMaxPool(
        output.data(),
        pooled_output.data(),
        output_size,
        pool_size,
        pool_len,
        pool_size_mod
    );
    
*/

void runMaxPool(float* output, float* pooled_output, size_t output_size, unsigned int pool_size, unsigned int pool_len) {
    float *d_output, *d_pooled_output;
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    CUDA_CHECK(cudaMalloc(&d_pooled_output, pool_len * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_output, output, output_size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(256, 1, 1);
    dim3 blocksPerGrid((pool_len + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);



    // start measuring time of computing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    maxPoolKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_pooled_output, pool_size, pool_len);
    cudaDeviceSynchronize();

    // stop measuring time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    //std::cout << "Kernel execution time: " << milliseconds << " milliseconds.\n";

    CUDA_CHECK(cudaMemcpy(pooled_output, d_pooled_output, pool_len * sizeof(float), cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_output);
    cudaFree(d_pooled_output);
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

