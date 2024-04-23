#pragma once
#include <cuda_runtime.h>

__global__ void dotProductKernel(float *a, float *b, float *out, unsigned int ilen, unsigned int alen, unsigned int blen);

void runCudaOperations(float* a, float* b, float* out, size_t a_size, size_t b_size, size_t out_size, unsigned int ilen, unsigned int alen, unsigned int blen);

void getSystemInformation();
