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

void runCudaOperations(float* images, float* filter_real, float* filter_abs, float* output, size_t images_size, size_t images_vector_len, size_t real_vector_len, size_t filters_size, size_t output_size, unsigned int inner_len, unsigned int image_len, unsigned int filter_len, size_t &memory_used_MB_dot_product, size_t &memory_free_MB_dot_product) {
    float *d_images, *d_filter_real, *d_filter_abs, *d_output;
    size_t free_before, total_before, free_after, total_after;

    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

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

    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));

    memory_used_MB_dot_product = (free_before - free_after) / 1024 / 1024;
    memory_free_MB_dot_product = free_after / 1024 / 1024;

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

void runMaxPool(float* output, float* pooled_output, size_t output_size, unsigned int pool_size, unsigned int pool_len, size_t &memory_used_MB_max_pool, size_t &memory_free_MB_max_pool) {
    float *d_output, *d_pooled_output;
    size_t free_before, total_before, free_after, total_after;

    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

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

    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));

    memory_used_MB_max_pool = (free_before - free_after) / 1024 / 1024;
    memory_free_MB_max_pool = free_after / 1024 / 1024;

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

void runCombinedOperations(
    float* images, float* filter_real, float* filter_abs, float* output, float* pooled_output,
    size_t images_size, size_t images_vector_len, size_t real_vector_len, size_t filters_size, size_t output_size, size_t pooled_output_size,
    unsigned int inner_len, unsigned int image_len, unsigned int filter_len, unsigned int pool_size, unsigned int pool_len,
    size_t &memory_used, size_t &memory_free) {

    float *d_images, *d_filter_real, *d_filter_abs, *d_output, *d_pooled_output;
    size_t free_before, total_before, free_after, total_after;

    // Get initial memory status
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

    // Allocate memory on device
    CUDA_CHECK(cudaMalloc(&d_images, images_size));
    CUDA_CHECK(cudaMalloc(&d_filter_real, filters_size));
    CUDA_CHECK(cudaMalloc(&d_filter_abs, filters_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    CUDA_CHECK(cudaMalloc(&d_pooled_output, pooled_output_size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_images, images, images_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter_real, filter_real, filters_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter_abs, filter_abs, filters_size, cudaMemcpyHostToDevice));

    unsigned int numBlocksX = (images_vector_len + inner_len - 1) / inner_len;
    unsigned int numBlocksY = (real_vector_len + inner_len - 1) / inner_len;


    int device;
    cudaGetDevice(&device); // get current device ID
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device); // get device properties
    int maxThreadsPerBlock = properties.maxThreadsPerBlock;
    int maxThreadsPerBlockDim = properties.maxThreadsDim[0];
    int maxThreadsPerBlockDim2 = properties.maxThreadsDim[1];
    dim3 threadsPerBlock(maxThreadsPerBlockDim, maxThreadsPerBlockDim2, 1);
    dim3 blocksPerGridCosine((numBlocksX + maxThreadsPerBlockDim - 1) / maxThreadsPerBlockDim,
                         (numBlocksY + maxThreadsPerBlockDim2 - 1) / maxThreadsPerBlockDim2);

    // start measuring time of computing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cosineSimilarityKernel<<<blocksPerGridCosine, threadsPerBlock>>>(d_images, d_filter_real, d_filter_abs, d_output, inner_len, image_len, images_vector_len, real_vector_len, filter_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Define grid and block sizes for pooling
    dim3 threadsPerBlockPool(256, 1, 1);
    dim3 blocksPerGridPool((pool_len + threadsPerBlockPool.x - 1) / threadsPerBlockPool.x, 1, 1);

    // Run max pooling kernel
    maxPoolKernel<<<blocksPerGridPool, threadsPerBlockPool>>>(d_output, d_pooled_output, pool_size, pool_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get final memory status
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));

    // Calculate memory used
    memory_used = (free_before - free_after) / 1024 / 1024;
    memory_free = free_after / 1024 / 1024;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(pooled_output, d_pooled_output, pooled_output_size, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_images);
    cudaFree(d_filter_real);
    cudaFree(d_filter_abs);
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

    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status) {
        std::cout << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << std::endl;
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::cout << "GPU memory usage: used = " << used_db / 1024.0 / 1024.0 << " MB, free = " << free_db / 1024.0 / 1024.0 << " MB, total = " << total_db / 1024.0 / 1024.0 << " MB" << std::endl;

}
