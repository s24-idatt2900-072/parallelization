#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_util.h"
#include "common.h"

void fillVector(std::vector<float>& v, float value) {
    std::fill(v.begin(), v.end(), value);
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}


// Kernel wrapper to simplify testing
void runCosineSimilarityKernel(const float* images, const float* filters_real, const float* filters_abs, float* output,
                            size_t images_size, size_t image_vector_len, size_t real_vector_len, size_t filters_size, size_t output_size, unsigned int inner_len, unsigned int image_len, unsigned int filter_len) {
    // Allocate device memory
    float *d_images, *d_filters_real, *d_filters_abs, *d_output;
    cudaMalloc(&d_images, images_size);
    cudaMalloc(&d_filters_real, filters_size);
    cudaMalloc(&d_filters_abs, filters_size);
    cudaMalloc(&d_output, output_size);

    // Copy data to device
    cudaMemcpy(d_images, images, images_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filters_real, filters_real, filters_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filters_abs, filters_abs, filters_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((image_len + 15) / 16, (filter_len + 15) / 16);

    // Run the kernel
    cosineSimilarityKernel<<<blocksPerGrid, threadsPerBlock>>>(d_images, d_filters_real, d_filters_abs, d_output, inner_len, image_len, image_vector_len, real_vector_len, filter_len);
    cudaDeviceSynchronize();

    checkCudaError("cosineSimilarityKernel");

    // Copy the results back to host
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy");

    // Free device memory
    cudaFree(d_images);
    cudaFree(d_filters_real);
    cudaFree(d_filters_abs);
    cudaFree(d_output);
}


void runMaxPoolKernel(const float* output, float* pooled_output, size_t output_size, unsigned int pool_size, unsigned int pool_len) {
    float *d_output, *d_pooled_output;
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_pooled_output, output_size);

    cudaMemcpy(d_output, output, output_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256, 1, 1);
    dim3 blocksPeGrid((pool_len + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    maxPoolKernel<<<blocksPeGrid, threadsPerBlock>>>(d_output, d_pooled_output, pool_size, pool_len);
    cudaDeviceSynchronize();

    cudaMemcpy(pooled_output, d_pooled_output, pool_len*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(d_pooled_output);
}

TEST(CosineSimilarityKernelTest, HandlesZeroInput) {
    int inner_len = 841;
    int images_len = 10;
    int filters_len = 10;
    int real_vector_len = filters_len * inner_len;

    int images_size = images_len * inner_len * sizeof(float);
    int image_vec_len = images_len * inner_len;
    int filters_size = filters_len * inner_len * sizeof(float);
    int output_size = images_len * filters_len * sizeof(float);

    // Create host vectors
    std::vector<float> h_images(images_len*inner_len, 0.0f);
    std::vector<float> h_filters_real(real_vector_len*inner_len, 0.0f);
    std::vector<float> h_filters_abs(real_vector_len*inner_len, 0.0f);
    std::vector<float> h_output(images_len*filters_len, -1.0f); // Initialize to -1 to check if it's modified

    // Run kernel
    runCosineSimilarityKernel(h_images.data(), h_filters_real.data(), h_filters_abs.data(), h_output.data(),
                              images_size, image_vec_len, real_vector_len, filters_size, output_size, inner_len, images_len, filters_len);
    

    // Check the output
    for (float val : h_output) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }
}

TEST(CosineSimilarityKernelTest, HandlesPositiveInput) {
    int inner_len = 841;
    int images_len = 10;
    int filters_len = 10;
    int real_vector_len = filters_len * inner_len;

    int images_size = images_len * inner_len * sizeof(float);
    int image_vec_len = images_len * inner_len;
    int filters_size = filters_len * inner_len * sizeof(float);
    int output_size = images_len * filters_len * sizeof(float);


    // Create host vectors
    std::vector<float> h_images(images_len*inner_len, 1.0f);
    std::vector<float> h_filters_real(real_vector_len*inner_len, 1.0f);
    std::vector<float> h_filters_abs(real_vector_len*inner_len, 1.0f);
    std::vector<float> h_output(images_len*filters_len, -1.0f); // Initialize to -1 to check if it's modified


    // Define grid and block dimensions
    dim3 blocks(16, 16, 1);
    dim3 threads((images_len + 15) / 16, (filters_len + 15) / 16);

    // Run kernel
    runCosineSimilarityKernel(h_images.data(), h_filters_real.data(), h_filters_abs.data(), h_output.data(),
                              images_size, image_vec_len, real_vector_len, filters_size, output_size, inner_len, images_len, filters_len);

    // Check the output
    for (float val : h_output) {
        EXPECT_FLOAT_EQ(val, 29.0f);
    }
}


TEST(CosineSimilarityKernelTest, HandlesRealCaseNumers) {
    int inner_len = 841;
    int image_len = 2;
    int filter_len = 2;

    size_t images_size = image_len * inner_len * sizeof(float); // alen * 841 floats
    size_t filters_size = filter_len * inner_len * sizeof(float); // blen * 841 floats
    size_t output_size = image_len * filter_len * sizeof(float); // alen * blen floats

    std::vector<float> images(image_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> filters_real(filter_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> filters_abs(filter_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> output(image_len * filter_len, 0.0f);   

    int image_vec_len = image_len * inner_len;
    int real_vector_len = filter_len * inner_len;

    loadDataFromFile("mnist/mnist_padded_29x29.csv", images);
    loadDataFromFile("filters/filters_real.csv", filters_real);
    loadDataFromFile("filters/filters_abs.csv", filters_abs);


    runCosineSimilarityKernel(images.data(), filters_real.data(), filters_abs.data(), output.data(),
                              images_size, image_vec_len, real_vector_len, filters_size, output_size, inner_len, image_len, filter_len);



    const float epsilon = 3e-6f;

    EXPECT_NEAR(output[0], 0.9454422f, epsilon);
    EXPECT_NEAR(output[1], 0.94615983f, epsilon);
    EXPECT_NEAR(output[2], 0.95451612f, epsilon);
    EXPECT_NEAR(output[3], 0.95522778f, epsilon);

}


TEST(MaxPoolKernelTest, HandlesRealCaseNumers) {
    int inner_len = 841;
    int image_len = 10;
    int filter_len = 10;

    size_t images_size = image_len * inner_len * sizeof(float); // alen * 841 floats
    size_t filters_size = filter_len * inner_len * sizeof(float); // blen * 841 floats
    size_t output_size = image_len * filter_len * sizeof(float); // alen * blen floats

    std::vector<float> images(image_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> filters_real(filter_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> filters_abs(filter_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> output(image_len * filter_len, 0.0f);   

    int image_vec_len = image_len * inner_len;
    int real_vector_len = filter_len * inner_len;

    loadDataFromFile("mnist/mnist_padded_29x29.csv", images);
    loadDataFromFile("filters/filters_real.csv", filters_real);
    loadDataFromFile("filters/filters_abs.csv", filters_abs);


    runCosineSimilarityKernel(images.data(), filters_real.data(), filters_abs.data(), output.data(),
                              images_size, image_vec_len, real_vector_len, filters_size, output_size, inner_len, image_len, filter_len);


    int pool_size = 5;
    int pool_len = (image_len * filter_len + pool_size - 1) / pool_size;

    std::vector<float> pooled_output(pool_len, 0.0f);

    runMaxPoolKernel(output.data(), pooled_output.data(), output_size, pool_size, pool_len);

    const float epsilon = 2e-5f;

    EXPECT_NEAR(pooled_output[0], 0.94816587f, epsilon);
    EXPECT_NEAR(pooled_output[1], 0.95112157f, epsilon);
    EXPECT_NEAR(pooled_output[2], 0.95722482f, epsilon);
    EXPECT_NEAR(pooled_output[3], 0.96015735f, epsilon);
    EXPECT_NEAR(pooled_output[4], -0.00291395f, epsilon);
    EXPECT_NEAR(pooled_output[5], -0.00245371f, epsilon);
    EXPECT_NEAR(pooled_output[6], -0.00241296f, epsilon);
    EXPECT_NEAR(pooled_output[7], -0.00190928f, epsilon);
    EXPECT_NEAR(pooled_output[8], -0.02609556f, epsilon);
    EXPECT_NEAR(pooled_output[9], -0.02077638f, epsilon);
    EXPECT_NEAR(pooled_output[10], 0.95638361f, epsilon);
    EXPECT_NEAR(pooled_output[11], 0.95942413f, epsilon);
    EXPECT_NEAR(pooled_output[12], 0.95982387f, epsilon);
    EXPECT_NEAR(pooled_output[13], 0.9626387f, epsilon);
    EXPECT_NEAR(pooled_output[14], 0.959446119f, epsilon);
    EXPECT_NEAR(pooled_output[15], 0.96225881f, epsilon);
    EXPECT_NEAR(pooled_output[16], 0.96422294f, epsilon);
    EXPECT_NEAR(pooled_output[17], 0.96686587f, epsilon);
    EXPECT_NEAR(pooled_output[18], 0.95831125f, epsilon);
    EXPECT_NEAR(pooled_output[19], 0.96106478f, epsilon);

}