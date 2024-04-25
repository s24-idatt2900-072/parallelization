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