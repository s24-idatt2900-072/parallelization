#include "common.h"

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


void writeDataToFile(const std::string& filename, const std::vector<float>& array, int num_images, int num_filters_per_image) {
    std::string path = "../data/" + filename;
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < num_filters_per_image; ++j) {
            file << array[i * num_filters_per_image + j];
            if (j < num_filters_per_image - 1) {
                file << ",";
            }
        }
        file << std::endl; 
    }

    file.close();
}
