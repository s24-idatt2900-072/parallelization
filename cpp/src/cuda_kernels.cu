#include "cuda_util.h"


__global__ void cosineSimilarityKernel(float *images, float *filters_real, float *filters_abs, float *output, unsigned int inner_len, unsigned int images_len,  size_t images_vector_len, size_t real_vector_len, unsigned int filters_len) {
    unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int maxIndexX = (images_vector_len / inner_len) * inner_len;
    unsigned int maxIndexY = (real_vector_len / inner_len) * inner_len;

    if(tidX * inner_len >= maxIndexX || tidY * inner_len >= maxIndexY){
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

    output[out_index] = dot / sqrtf(norm);
}


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