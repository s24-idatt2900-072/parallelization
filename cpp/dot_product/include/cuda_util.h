#pragma once
#include <cuda_runtime.h>

__global__ void cosineSimilarityKernel(float *images, float *filters_real, float *filters_abs, float *output, unsigned int ilen, unsigned int alen, unsigned int blen);

void runCudaOperations(float* images, float* filter_real, float* filer_abs, float* output, size_t images_size, size_t filters_size, size_t output_size, unsigned int inner_len, unsigned int image_len, unsigned int filter_len);

void getSystemInformation();
