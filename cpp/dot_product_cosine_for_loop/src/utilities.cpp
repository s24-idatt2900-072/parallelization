#include "common.h"


void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void loadDataFromFile(const std::string& filename, std::vector<float>& array) {
    std::string path = "../data/" + filename;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    size_t count = 0;
    while (std::getline(file, line) && count < array.size()) {
        std::istringstream iss(line);
        std::string value;
        while (std::getline(iss, value, ',') && count < array.size()) {
            array[count++] = std::stof(value);
        }
    }

    file.close();

    if (count < array.size()) {
        std::cerr << "Warning: Not enough data was read from " << path << "; expected " << array.size() << " values, but got " << count << std::endl;
    }
}
