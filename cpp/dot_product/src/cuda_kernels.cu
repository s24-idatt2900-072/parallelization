#include "cuda_util.h"


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
