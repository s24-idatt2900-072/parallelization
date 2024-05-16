#include "common.h"
#include "cuda_util.h"

const unsigned int IMAGE_SIZE = 841;
const unsigned int initial_image_count = 1000;
const unsigned int initial_filter_count = 1000;
const std::string IMAGE_FILE = "mnist/mnist_padded_29x29.csv";
const std::string FILTERS_REAL_FILE = "filters/filters_real.csv";
const std::string FILTERS_ABS_FILE = "filters/filters_abs.csv";

unsigned int readPositiveInteger(const std::string& prompt) {
    unsigned int value = 0;
    std::string input;
    while (true) {
        std::cout << prompt;
        std::cin >> input;
        try {
            value = std::stoul(input);
            if (value > 0) break;
            std::cout << "Please enter a positive integer.\n";
        } catch (std::exception& e) {
            std::cout << "Invalid input. Please enter a valid integer.\n";
        }
    }
    return value;
}

void loadDataToVectors(std::vector<float>& images, std::vector<float>& filters_real, std::vector<float>& filters_abs) {
    loadDataFromFile(IMAGE_FILE, images);
    loadDataFromFile(FILTERS_REAL_FILE, filters_real);
    loadDataFromFile(FILTERS_ABS_FILE, filters_abs);
}


int manual_cosine_similarity_and_max_pool() {
    unsigned int image_len = readPositiveInteger("Enter number of images: ");
    
    unsigned int filter_len = readPositiveInteger("Enter number of filters: ");


    std::vector<float> images(image_len * IMAGE_SIZE, 0.0f);  
    std::vector<float> filters_real(filter_len * IMAGE_SIZE, 0.0f);
    std::vector<float> filters_abs(filter_len * IMAGE_SIZE, 0.0f);  

    size_t memory_used, memory_free;

    loadDataToVectors(images, filters_real, filters_abs);

    size_t images_size = images.size() * sizeof(float);
    size_t filters_size = filters_real.size() * sizeof(float);
    size_t output_size = image_len * filter_len * sizeof(float);
    std::vector<float> output(image_len * filter_len, 0.0f);

    const unsigned int pool_size = 500;
    unsigned int pool_len = (image_len * filter_len + pool_size - 1) / pool_size;

    std::vector<float> pooled_output(pool_len, 0.0f);

    size_t images_vector_len = images.size();
    size_t real_vector_len = filters_real.size();

   
    runCombinedOperations(
                    images.data(),
                    filters_real.data(),
                    filters_abs.data(),
                    pooled_output.data(),
                    images_size,
                    images_vector_len,
                    real_vector_len,
                    filters_size,
                    output_size,
                    pool_len * sizeof(float),
                    IMAGE_SIZE,
                    image_len,
                    filter_len,
                    pool_size,
                    pool_len,
                    memory_used,
                    memory_free
                );


    std::cout << "Size of pooled output: " << pooled_output.size() << std::endl;

    std::cout << "Results:\n";
    for (unsigned int i = 0; i < pooled_output.size(); ++i) {
        std::cout << "Pooled output[" << i << "] = " << pooled_output[i] << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << std::endl;

    return 0;
}


int research_run() {

    unsigned int image_len = readPositiveInteger("Enter number of images: ");
    
    unsigned int filter_len = readPositiveInteger("Enter number of starting filters: ");

    unsigned int increment = readPositiveInteger("Enter increment amount: ");

    unsigned int max_filters = readPositiveInteger("Enter maximum number of filters: ");

    unsigned int pool_size = readPositiveInteger("Enter pool size: ");


    std::vector<float> images(image_len * IMAGE_SIZE, 0.0f);  
    std::vector<float> filters_real(filter_len * IMAGE_SIZE, 0.0f);
    std::vector<float> filters_abs(filter_len * IMAGE_SIZE, 0.0f);  

    size_t memory_used, memory_free;

    loadDataToVectors(images, filters_real, filters_abs);

    size_t images_size = images.size() * sizeof(float);
    size_t filters_size = filters_real.size() * sizeof(float);
    size_t output_size = image_len * filter_len * sizeof(float);


    unsigned int pool_len = (image_len * filter_len + pool_size - 1) / pool_size;
    size_t pooled_output_size = pool_len * sizeof(float);

    std::vector<float> pooled_output(pool_len, 0.0f);

    size_t images_vector_len = images.size();
    size_t real_vector_len = filters_real.size();

    std::string file_name = "CPP_GPU_img_" + std::to_string(image_len) + "_" + std::to_string(max_filters) + "_" + std::to_string(std::time(0)) + ".csv";

    unsigned int previous_filter_len = filter_len;

    std::vector<std::string> buffer;

    buffer.push_back("Filter, ID, Time_us, Average_time, Memory_used_MiB, Memory_free_MiB");

    while (true) {
        previous_filter_len = filter_len;
        std::vector<unsigned int> time_us_vec;
        unsigned int time_us = 0;
        if (filter_len > 5000) {
            std::cout << "Processing " << image_len << " images with " << filter_len << " filters...\n";
        }

        for (unsigned int i = 1; i < 61; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            runCombinedOperations(
                images.data(),
                filters_real.data(),
                filters_abs.data(),
                pooled_output.data(),
                images_size,
                images_vector_len,
                real_vector_len,
                filters_size,
                output_size,
                pooled_output_size,
                IMAGE_SIZE,
                image_len,
                filter_len,
                pool_size,
                pool_len,
                memory_used,
                memory_free
            );

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::micro> duration = end - start;
            time_us = (unsigned int)std::round(duration.count());
            time_us_vec.push_back(time_us);
            if (i == 1 || filter_len != previous_filter_len) {
                buffer.push_back(std::to_string(filter_len) + ", " + std::to_string(i) + ", " + std::to_string(time_us) + ", 0" + ", " + std::to_string(memory_used) + ", " + std::to_string(memory_free));
            } else {
                buffer.push_back("0, " + std::to_string(i) + ", " + std::to_string(time_us) + ", 0" + ", " + std::to_string(memory_used) + ", " + std::to_string(memory_free));
            }
        }

        float sum = 0.0f;
        for (float t : time_us_vec) {
            sum += t;
        }
        unsigned int average_time = (unsigned int)std::round(sum / time_us_vec.size());
        if (average_time == 0) {
            average_time = 1;
        }
        buffer.push_back("0, 0, 0, " + std::to_string(average_time));
        std::ofstream outFile(file_name, std::ios::app);
        if (!outFile) {
            std::cerr << "Error opening file for writing.\n";
        } else {
            for (const std::string& line : buffer) {
                outFile << line << std::endl;
            }
            outFile.close();
        }

        unsigned int zero_occurrences = 0;
        for (unsigned int i = 0; i < pooled_output.size(); ++i) {
            if (pooled_output[i] == 0.0f) {
                zero_occurrences++;
            }
        }

        if (zero_occurrences != 0) {
            std::cout << "Zero occurrences: " << zero_occurrences << std::endl;
        }  

        buffer.clear();

        pooled_output.clear();

        filter_len += increment;

        expandVector(filters_real, initial_filter_count * IMAGE_SIZE, increment * IMAGE_SIZE); 
        expandVector(filters_abs, initial_filter_count * IMAGE_SIZE, increment * IMAGE_SIZE);

        filters_size = filters_real.size() * sizeof(float);
        output_size = image_len * filter_len * sizeof(float);

        pool_len = (image_len * filter_len + pool_size - 1) / pool_size;

        pooled_output_size = pool_len * sizeof(float);

        pooled_output.resize(pool_len, 0.0f);

        images_vector_len = images.size();
        real_vector_len = filters_real.size();

        if (filter_len > max_filters) {
            std::cout << "Results written to " << file_name << std::endl;
            break;
        }
    }

    return 0;

}

int main() {
    getSystemInformation();
    std::cout << "\nPlease select an option:\n";
    std::cout << "1. Run with manual input\n";
    std::cout << "2. Run research\n";
    std::cout << "3. Exit\n";
    
    unsigned int choice = readPositiveInteger("Enter your choice: ");
    switch (choice) {
        case 1:
            manual_cosine_similarity_and_max_pool();
            break;
        case 2:
            research_run();
            break;
        case 3:
            std::cout << "Exiting...\n";
            break;
        default:
            std::cout << "Invalid choice. Exiting...\n";
            break;
    }

    return 0;
}