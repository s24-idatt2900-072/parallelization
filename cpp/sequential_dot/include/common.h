#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cmath>


void loadDataFromFile(const std::string& filename, std::vector<float>& array);

void writeDataToFile(const std::string& filename, const std::vector<float>& array, int num_images, int num_filters_per_image);

void cosineSimiliarity(float* images, float* filter_real, float* filter_abs, float* output, int num_images, int num_filters, int image_size);

void maxPooling(float* array, float* pooled_results, int num_images, int num_filters, int pool_size);


template <typename T>
void expandVector(std::vector<T>& vec, size_t original_len, unsigned int expansion) {
    size_t current_size = vec.size();
    size_t new_len = vec.size() + expansion;
    vec.resize(new_len);

    for (size_t i = 0; i < expansion; ++i) {
        vec[current_size + i] = vec[i % original_len];
    }

}


