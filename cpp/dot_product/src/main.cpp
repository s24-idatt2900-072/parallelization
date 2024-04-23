#include "common.h"
#include "cuda_util.h"

int main(int argc, char* argv[]) {
    //unsigned int alen = 10000;
    //unsigned int blen = 100000;
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <alen> <blen>\n";
        return 1;
    }

    unsigned int alen = atoi(argv[1]);
    unsigned int blen = atoi(argv[2]);
    unsigned int ilen = 841;

    srand(123);

    size_t a_size = alen * ilen * sizeof(float); // alen * 841 floats
    size_t b_size = blen * ilen * sizeof(float); // blen * 841 floats
    size_t out_size = alen * blen * sizeof(float); // alen * blen floats

    std::vector<float> a(alen * ilen, 0.0f);  // Initialize with some values
    std::vector<float> b(blen * ilen, 0.0f);  // Initialize with some values
    std::vector<float> out(alen * blen, 0.0f);

    /*
    for (size_t i = 0; i < alen * ilen; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (size_t i = 0; i < blen * ilen; ++i) {
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    */
    

    std::cout << "Importing images and filters..." << std::endl;
    loadDataFromFile("images_test.txt", a);
    loadDataFromFile("filters_test.txt", b);


    std::cout << "Size of a: " << a_size / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Size of b: " << b_size / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Size of output (out): " << out_size / (1024.0 * 1024.0) << " MB\n";
    
    if (alen * ilen > 0) {
        std::cout << "First element of a: " << a[0] << std::endl;
    }
    if (blen * ilen > 0) {
        std::cout << "First element of b: " << b[0] << std::endl;
    }

    getSystemInformation();

    runCudaOperations(a.data(), b.data(), out.data(), a_size, b_size, out_size, ilen, alen, blen);

    std::cout << "Checking Dot Product Results...\n";
    bool resultsPlausible = true;
    for (unsigned int i = 0; i < alen * blen; ++i) {
        if (out[i] < 0 || out[i] > ilen) {
            std::cout << "Mismatch at index " << i << ": Result is " << out[i] << ".\n";
            resultsPlausible = false;
            break;
        }
    }
    if (!resultsPlausible) {
        std::cout << "Results are not plausible.\n";
    }
    if (alen * blen < 250000) {
        std::cout << "Results exceed the threshold and are plausible.\n";
        std::ofstream outFile("better_results.txt");
        if (!outFile) {
            std::cerr << "Error opening file for writing.\n";
        } else {
            for (unsigned int i = 0; i < alen * blen; ++i) {
                outFile << out[i] << std::endl; // Write each element followed by a newline
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

