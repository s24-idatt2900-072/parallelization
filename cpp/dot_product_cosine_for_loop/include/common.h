#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

void checkCudaError(cudaError_t error);

void loadDataFromFile(const std::string& filename, std::vector<float>& array);

void loadResultsFromFile(const std::string& filename, std::vector<float>& results);

template <typename T>
void expandVector(std::vector<T>& vec, size_t original_len, unsigned int expansion) {
    size_t current_size = vec.size();
    size_t new_len = vec.size() + expansion;
    vec.resize(new_len);

    for (size_t i = 0; i < expansion; ++i) {
        vec[current_size + i] = vec[i % original_len];
    }

}
