#include "common.h"


void cosineSimiliarity(float* images, float* filter_real, float* filter_abs, float* output, int num_images, int num_filters, int image_size) {
    std::vector<float> d(num_images * num_filters);
    for (int filter_idx = 0; filter_idx < num_filters; ++filter_idx) {
        float* abs_filter = filter_abs + filter_idx * image_size;
        float* real_filter = filter_real + filter_idx * image_size;

        for (int image_idx = 0; image_idx < num_images; ++image_idx) {
            float dot = 0.0;
            float norm = 0.0;
            for (int i = 0; i < image_size; ++i) {
                d[image_idx * image_size + i] = images[image_idx * image_size + i] * abs_filter[i];
                dot += d[image_idx * image_size + i] * real_filter[i];
                norm += d[image_idx * image_size + i] * d[image_idx * image_size + i];
            }
            output[image_idx * num_filters + filter_idx] = dot / sqrt(norm);
        }
    }
}



void maxPooling(float* array, float* pooled_results, int num_images, int num_filters, int pool_size) {
    int num_complete_pools = num_filters / pool_size;
    int remainder = num_filters % pool_size;

    for (int image_idx = 0; image_idx < num_images; ++image_idx) {
        for (int pool_idx = 0; pool_idx < num_complete_pools; ++pool_idx) {
            float max_val = array[image_idx * num_filters + pool_idx * pool_size];
            for (int i = 1; i < pool_size; ++i) {
                float current_val = array[image_idx * num_filters + pool_idx * pool_size + i];
                if (current_val > max_val) max_val = current_val;
            }
            pooled_results[image_idx * num_complete_pools + pool_idx] = max_val;
        }
        if (remainder != 0) {
            float max_val = array[image_idx * num_filters + num_complete_pools * pool_size];
            for (int i = 1; i < remainder; ++i) {
                float current_val = array[image_idx * num_filters + num_complete_pools * pool_size + i];
                if (current_val > max_val) max_val = current_val;
            }
            pooled_results[image_idx * num_complete_pools + num_complete_pools] = max_val;
        }
    }
}

