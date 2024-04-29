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
    while (std::getline(file, line)) {
        if (line[0] == '#') continue;  // Skip #id lines
        std::istringstream iss(line);
        std::string value;
        while (std::getline(iss, value, ',')) {
            float num = std::stof(value);
            if (count < array.size()) {
                array[count] = num;
            } else {
                array.push_back(num);
            }
            count++;
        }
    }

    file.close();
}






/**
 * void loadDataFromFile(const std::string& filename, std::vector<float>& array, unsigned int total_len) {
    std::string path = "../data/" + filename;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    size_t count = 0;
    while (std::getline(file, line) && count < total_len) {
        if (line[0] == '#') continue;  // Skip #id lines
        std::istringstream iss(line);
        std::string value;
        while (std::getline(iss, value, ',') && count < total_len) {
            array[count % array.size()] = std::stof(value);  // Wrap around if index exceeds initial size
            count++;
        }
    }

    file.close();

    if (count < total_len) {
        std::cerr << "Warning: Not enough data was read from " << path << "; expected " << total_len
                  << " values, but got " << count << std::endl;
    }
}
*/