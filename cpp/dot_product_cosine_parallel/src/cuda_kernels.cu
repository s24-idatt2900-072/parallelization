#include "cuda_util.h"


__global__ void cosineSimilarityKernel(float *images, float *filters_real, float *filters_abs, float *output, unsigned int ilen, unsigned int alen, unsigned int blen) {
    unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    if(tidX >= alen || tidY >= blen) return;

    float dot = 0.0;
    float norm_a = 0.0;
    float norm_b = 0.0;

    for(unsigned int i = 0; i < ilen; ++i) {
        unsigned int idx_a = tidX * ilen + i;
        unsigned int idx_b = tidY * ilen + i;
        float a_val = images[idx_a];
        float b_val = filters_real[idx_b] * filters_abs[idx_b];
        dot += a_val * b_val;
        norm_a += a_val * a_val;
        norm_b += b_val * b_val;
    }

    float cosine_sim = norm_a > 0.0f && norm_b > 0.0f ? dot / (sqrt(norm_a) * sqrt(norm_b)) : 0.0f;

    unsigned int out_index = tidX * blen + tidY;
    output[out_index] = cosine_sim;
}