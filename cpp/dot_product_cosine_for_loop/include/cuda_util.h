#pragma once
#include <cuda_runtime.h>
#include <cfloat>


__global__ void cosineSimilarityKernel(float *images, float *filters_real, float *filters_abs, float *output, unsigned int inner_len, unsigned int images_len,  size_t images_vector_len, size_t real_vector_len, unsigned int filters_len);

__global__ void maxPoolKernel(float *output, float *pooled_output, unsigned int pool_size, unsigned int pool_len);


void runCudaOperations(float* images, float* filter_real, float* filter_abs, float* output, size_t images_size, size_t images_vector_len, size_t real_vector_len, size_t filters_size, size_t output_size, unsigned int inner_len, unsigned int image_len, unsigned int filter_len);

void runMaxPool(float* output, float* pooled_output, size_t output_size, unsigned int pool_size, unsigned int pool_len);


void getSystemInformation();
