#include "common.h"
#include "cuda_util.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <alen> <blen>\n";
        return 1;
    }

    unsigned int image_len = atoi(argv[1]);
    unsigned int filter_len = atoi(argv[2]);
    unsigned int inner_len = 841;

    srand(123);

    size_t images_size = image_len * inner_len * sizeof(float); // alen * 841 floats
    size_t filters_size = filter_len * inner_len * sizeof(float); // blen * 841 floats
    size_t output_size = image_len * filter_len * sizeof(float); // alen * blen floats

    std::vector<float> images(image_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> filters_real(filter_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> filters_abs(filter_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> output(image_len * filter_len, 0.0f);

    std::cout << "Importing images and filters..." << std::endl;
    //loadDataFromFile("images.txt", images);
    //loadDataFromFile("filters_real.txt", filters_real);
    //loadDataFromFile("filters_abs.txt", filters_abs);

    for (unsigned int i = 0; i < image_len * inner_len; ++i) {
        images[i] = (float)rand() / RAND_MAX;
    }

    for (unsigned int i = 0; i < filter_len * inner_len; ++i) {
        filters_real[i] = (float)rand() / RAND_MAX;
    }

    for (unsigned int i = 0; i < filter_len * inner_len; ++i) {
        filters_abs[i] = (float)rand() / RAND_MAX;
    }


    std::cout << "Size of images: " << images_size / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Size of filters_real: " << filters_size / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Size of filters_abs: " << filters_size / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Size of output (out): " << output_size / (1024.0 * 1024.0) << " MB\n";
    
    if (image_len * inner_len > 0) {
        std::cout << "\nFirst element of a: " << images[0] << std::endl;
    }
    if (filter_len * inner_len > 0) {
        std::cout << "\nFirst element of b: " << filters_real[0] << std::endl;
        std::cout << "\nFirst element of b: " << filters_abs[0] << std::endl;
    }

    getSystemInformation();

    runCudaOperations(
        images.data(),
        filters_real.data(),
        filters_abs.data(),
        output.data(),
        images_size, 
        filters_size, 
        output_size, 
        inner_len, 
        image_len, 
        filter_len
        );

    std::cout << "Checking Dot Product Results...\n";
    bool resultsPlausible = true;
    for (unsigned int i = 0; i < image_len * filter_len; ++i) {
        if (output[i] < 0 || output[i] > inner_len) {
            std::cout << "Mismatch at index " << i << ": Result is " << output[i] << ".\n";
            resultsPlausible = false;
            break;
        }
    }
    if (!resultsPlausible) {
        std::cout << "Results are not plausible.\n";
    }
    if (image_len * filter_len < 250000) {
        std::cout << "Results exceed the threshold and are plausible.\n";
        std::ofstream outFile("better_results.txt");
        if (!outFile) {
            std::cerr << "Error opening file for writing.\n";
        } else {
            for (unsigned int i = 0; i < image_len * filter_len; ++i) {
                outFile << output[i] << std::endl; // Write each element followed by a newline
            }
            outFile.close(); // Close the file after writing
            std::cout << "Dot product results have been written to better_results.txt\n";
        }
    } else {
        std::cout << "Exceeded treshold. No file written" << std::endl;
    }


    std::cout << std::endl;
    std::cout << std::endl;

    return 0;

}

