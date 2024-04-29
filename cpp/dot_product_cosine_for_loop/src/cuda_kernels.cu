#include "cuda_util.h"


__global__ void cosineSimilarityKernel(float *images, float *filters_real, float *filters_abs, float *output, unsigned int inner_len, unsigned int images_len,  size_t images_vector_len, size_t real_vector_len, unsigned int filters_len) {
    unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    if(tidX >= images_vector_len / inner_len || tidY >= real_vector_len / inner_len){
        return;
    }

    float dot = 0.0;
    float norm = 0.0;

    for(unsigned int i = 0; i < inner_len; ++i) {
        unsigned int image_index = tidX * inner_len + i;
        unsigned int filter_index = tidY * inner_len + i;
        float d = images[image_index] * filters_abs[filter_index];
        dot = dot + d * filters_real[filter_index];
        norm = norm + d * d;
    }

    unsigned int out_index = tidX * filters_len + tidY;
    if (norm > 0.0f) {
        output[out_index] = dot / sqrtf(norm);
    } else {
        output[out_index] = 0.0f;
    }
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

 *     float *d_output, *d_pooled_output;
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    CUDA_CHECK(cudaMalloc(&d_pooled_output, output_size);

    CUDA_CHECK(cudaMemcpy(d_output, output, output_size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((pool_len + 15) / 16, (pool_len + 15) / 16);

    // start measuring time of computing
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    maxPoolKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_pooled_output, pool_size, pool_len, pool_size_mod);
    cudaDeviceSynchronize();

    // stop measuring time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " milliseconds.\n";

    CUDA_CHECK(cudaMemcpy(pooled_output, d_pooled_output, output_size, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_output);
    cudaFree(d_pooled_output);
}
*/

__global__ void maxPoolKernel(float *output, float *pooled_output, unsigned int pool_size, unsigned int pool_len) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool_len) {
        float max_val = -FLT_MAX;
        unsigned int start_idx = idx * pool_size;
        unsigned int end_idx = start_idx + pool_size;
        
        for (unsigned int j = start_idx; j < end_idx; ++j) {
            max_val = fmaxf(max_val, output[j]);
        }
        pooled_output[idx] = max_val;
    }
}