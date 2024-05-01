#include "common.h"
#include "cuda_util.h"

int manual() {
    unsigned int image_len = 0;

    while (image_len < 1) {
        std::cout << "Enter number of images: ";
        std::string input;
        std::cin >> input;
        try {
            image_len = std::stoi(input);
            if (image_len < 1) {
                std::cout << "Please enter a positive integer.\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Invalid input. Please enter a valid integer.\n";
        }
    }
    
    unsigned int filter_len = 0;

    while (filter_len < 1) {
        std::cout << "Enter number of filters: ";
        std::string input;
        std::cin >> input;
        try {
            filter_len = std::stoi(input);
            if (filter_len < 1) {
                std::cout << "Please enter a positive integer.\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Invalid input. Please enter a valid integer.\n";
        }
    }

    unsigned int inner_len = 841;


    size_t images_size = image_len * inner_len * sizeof(float); // alen * 841 floats
    size_t filters_size = filter_len * inner_len * sizeof(float); // blen * 841 floats
    size_t output_size = image_len * filter_len * sizeof(float); // alen * blen floats

    std::vector<float> images(image_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> filters_real(filter_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> filters_abs(filter_len * inner_len, 0.0f);  // Initialize with some values
    std::vector<float> output(image_len * filter_len, 0.0f);

    size_t images_vector_len = images.size();
    size_t real_vector_len = filters_real.size();

    std::cout << "Importing images and filters..." << std::endl;
    loadDataFromFile("mnist/mnist_padded_29x29.csv", images);
    loadDataFromFile("filters/filters_real.csv", filters_real);
    loadDataFromFile("filters/filters_abs.csv", filters_abs);

    std::cout << "Images and filters imported." << std::endl;
    // print length of images and filters
    std::cout << "Length of images: " << images.size() << std::endl;
    std::cout << "Length of filters_real: " << filters_real.size() << std::endl;
    std::cout << "Length of filters_abs: " << filters_abs.size() << std::endl;

    /*
    for (unsigned int i = 0; i < image_len * inner_len; ++i) {
        images[i] = (float)rand() / RAND_MAX;
    }

    for (unsigned int i = 0; i < filter_len * inner_len; ++i) {
        filters_real[i] = (float)rand() / RAND_MAX;
    }

    for (unsigned int i = 0; i < filter_len * inner_len; ++i) {
        filters_abs[i] = (float)rand() / RAND_MAX;
    }
    */

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

    runCudaOperations(
        images.data(),
        filters_real.data(),
        filters_abs.data(),
        output.data(),
        images_size, 
        images_vector_len,
        real_vector_len,
        filters_size, 
        output_size, 
        inner_len, 
        image_len, 
        filter_len
        );


    // as each image will perform dot proudct operation with every filter, the output will be image_len * filter_len
    // output is a 1D array of size image_len * filter_len
    // print the first 10 dot products of image 1 with all filters
    for (unsigned int i = 0; i < 10; ++i) {
        std::cout << "Output[" << i << "] = " << output[i] << std::endl;
    }


    std::cout << "Checking Dot Product Results...\n";
    bool resultsPlausible = true;
    for (unsigned int i = 0; i < image_len * filter_len; ++i) {
        if (output[i] < -1 || output[i] > 1 || output[i] > inner_len) {
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

    // print output length
    std::cout << "Output length: " << output.size() << std::endl;


    // now max pool the output with a given pool size
    // pool size is 5
    // this will also be done on the GPU
    // output is a 1D array of size image_len * filter_len
    // therefore, the max pooling processing has to take into account the chunks of image * filter length
    // also take into account if the pool size does not perfectly line up with the last chunk

    unsigned int pool_size = 5;

    if (filter_len < pool_size){
        pool_size = filter_len;
    }

    unsigned int pool_len = (image_len * filter_len + pool_size - 1) / pool_size;

    std::cout << "Pool size: " << pool_size << std::endl;
    std::cout << "Pool len: " << pool_len << std::endl;

    std::vector<float> pooled_output(pool_len, 0.0f);

    runMaxPool(
        output.data(),
        pooled_output.data(),
        output_size,
        pool_size,
        pool_len
    );


    // print size of pooled output
    std::cout << "Size of pooled output: " << pooled_output.size() << std::endl;

    
    /*
    std::cout << "Printing Max Pool Results...\n" << std::endl;

    // print all max pooled results
    for (unsigned int i = 0; i < pool_len; ++i) {
        std::cout << "Pooled Output[" << i << "] = " << pooled_output[i] << std::endl;
    }
    */
    
    std::cout << std::endl;
    std::cout << std::endl;

    return 0;
}



void research() {

    std::string person;
    while (person.empty()) {
        std::cout << "Enter your name: ";
        std::cin >> person;
        if (person.empty()) {
            std::cout << "Please enter a valid name.\n";
        }
    }

    unsigned int images_and_filters_in_file = 1000;

    unsigned int image_len = 0;

    while (image_len < 1) {
        std::cout << "Enter number of images: ";
        std::string input;
        std::cin >> input;
        try {
            image_len = std::stoi(input);
            if (image_len < 1) {
                std::cout << "Please enter a positive integer.\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Invalid input. Please enter a valid integer.\n";
        }
    }

    unsigned int expansion_amount = 0;

    while (expansion_amount < 1) {
        std::cout << "Enter expansion amount: ";
        std::string input;
        std::cin >> input;
        try {
            expansion_amount = std::stoi(input);
            if (expansion_amount < 1) {
                std::cout << "Please enter a positive integer.\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Invalid input. Please enter a valid integer.\n";
        }
    }

    unsigned int filter_len = 0;

    if (expansion_amount < 500) {
        while (filter_len < 1) {
            std::cout << "Enter number of filters: ";
            std::string input;
            std::cin >> input;
            try {
                filter_len = std::stoi(input);
                if (filter_len < 1) {
                    std::cout << "Please enter a positive integer.\n";
                }
            } catch (const std::exception& e) {
                std::cout << "Invalid input. Please enter a valid integer.\n";
            }
        }
    } else {
        filter_len = 500;
    }

    // 29x29, size of each image and filter
    unsigned int inner_len = 841;

    // pool size. the factor of how much the vector of dot products will be reduced
    unsigned int pool_size = 0;

    while (pool_size < 1) {
        std::cout << "Enter pool size: ";
        std::string input;
        std::cin >> input;
        try {
            pool_size = std::stoi(input);
            if (pool_size < 1) {
                std::cout << "Please enter a positive integer.\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Invalid input. Please enter a valid integer.\n";
        }
    }


    unsigned int when_to_stop = 0;

    while (when_to_stop < 1) {
        std::cout << "Enter when to stop: (amount of filters)\n";
        std::string input;
        std::cin >> input;
        try {
            when_to_stop = std::stoi(input);
            if (when_to_stop < 1) {
                std::cout << "Please enter a positive integer.\n";
            }
        } catch (const std::exception& e) {
            std::cout << "Invalid input. Please enter a valid integer.\n";
        }
    }

    // size of each image in bytes.
    // constant for entire program
    size_t images_size = image_len * inner_len * sizeof(float); // alen * 841 floats
    
    // size of each filter in bytes. 
    // must be updated after each expansion
    size_t filters_size = filter_len * inner_len * sizeof(float); // blen * 841 floats
    
    // size of output in bytes. 
    // must be updated after each expansion
    size_t output_size = image_len * filter_len * sizeof(float); // alen * blen floats

    // initialize images, filters_real, filters_abs, and output vectors
    std::vector<float> images(image_len * inner_len, 0.0f);  
    std::vector<float> filters_real(filter_len * inner_len, 0.0f);
    std::vector<float> filters_abs(filter_len * inner_len, 0.0f); 
    std::vector<float> output(image_len * filter_len, 0.0f);

    // length of images vector 
    // constant for entire program
    size_t images_vector_len = images.size();

    // length of filters_real and filters_abs vectors
    // must be updated after each expansionW
    size_t real_vector_len = filters_real.size();

    // original length of filters_real and filters_abs vectors
    // used for expansion and is constant for entire program
    size_t original_filter_len = real_vector_len;

    std::cout << "Importing images and filters..." << std::endl;
    loadDataFromFile("mnist/mnist_padded_29x29.csv", images);
    loadDataFromFile("filters/filters_real.csv", filters_real);
    loadDataFromFile("filters/filters_abs.csv", filters_abs);

    std::cout << "Images and filters imported." << std::endl;

    // file_name = f"CPU_img_{num_images}_{current_UNIX_time)}.csv"
    // convert to cpp
    std::string file_name = person + "CPP_GPU_img_" + std::to_string(image_len) + "_" + std::to_string(std::time(0)) + ".csv";

    unsigned int previous_filter_len = filter_len;


    unsigned int pool_len = (image_len * filter_len + pool_size - 1) / pool_size;

    // buffer
    std::vector<std::string> buffer;

    // add ("Filter", "ID", "Time_ms", "Average_time") which is the column names of the csv file
    buffer.push_back("Filter, ID, Time_ms, Average_time");

    while (true) {
        previous_filter_len = filter_len;
        // vector called time_ms containing time taken in ms for each dot product and max pool operation
        std::vector<unsigned int> time_ms_vec;
        unsigned int time_ms = 0;
        std::cout << "Processing " << image_len << " images with " << filter_len << " filters...\n";
        // it will be done 30 times to get an average time and ensure standard deviation 

        if (filter_len < pool_size){
            pool_size = filter_len;
        }


        std::vector<float> pooled_output(pool_len, 0.0f);
        
        // data will be written to a csv file over the course of processing
        // adds data to a form of buffer to prevent writing to file every time
        // buffer will hold it in string format
        // add to write bufer to file (Filter, ID, Time_ms, Average_time_)

        
        
        for (unsigned int i = 1; i < 31; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            runCudaOperations(
                images.data(),
                filters_real.data(),
                filters_abs.data(),
                output.data(),
                images_size, 
                images_vector_len,
                real_vector_len,
                filters_size, 
                output_size, 
                inner_len, 
                image_len, 
                filter_len
                );

            //std::cout << "Output length: " << output.size() << std::endl;

            runMaxPool(
                output.data(),
                pooled_output.data(),
                output_size,
                pool_size,
                pool_len
            );
            // print size of pooled output
            //std::cout << "Size of pooled output: " << pooled_output.size() << std::endl;


            //std::cout << "Done with max pool operation.\n";


            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = end - start;
            time_ms = (unsigned int)std::round(duration.count());
            time_ms_vec.push_back(time_ms);
            if (i == 1 || filter_len != previous_filter_len) {
                // add to write bufer to file (filter amount, i, time spent in ms, 0)
                buffer.push_back(std::to_string(filter_len) + ", " + std::to_string(i) + ", " + std::to_string(time_ms) + ", 0");
            } else {
                // add to write bufer to file (0, i, time spent in ms, 0)
                buffer.push_back("0, " + std::to_string(i) + ", " + std::to_string(time_ms) + ", 0");
            }

            //clear output
            output.clear();
            output.resize(image_len * filter_len, 0.0f);

            //clear pooled_output
            pooled_output.clear();
            pooled_output.resize(pool_len, 0.0f);


        }
        // calculate average time of 30 runs
        float sum = 0.0f;
        for (float t : time_ms_vec) {
            sum += t;
        }
        unsigned int average_time = (unsigned int)std::round(sum / time_ms_vec.size());
        // add to write bufer to file (0, 0, 0, average_time)
        buffer.push_back("0, 0, 0, " + std::to_string(average_time));
        // write buffer to file
        // TODO: move to separate function in separate file once working
        std::ofstream outFile(file_name, std::ios::app);
        if (!outFile) {
            std::cerr << "Error opening file for writing.\n";
        } else {
            for (const std::string& line : buffer) {
                outFile << line << std::endl; // Write each element followed by a newline
            }
            outFile.close(); // Close the file after writing
        }

        // clear buffer
        buffer.clear();
        filter_len += expansion_amount;
        
        // expand filters_real and filters_abs
        expandVector(filters_real, original_filter_len, expansion_amount * inner_len);
        expandVector(filters_abs, original_filter_len, expansion_amount * inner_len);

        // reset output
        output.clear();
        output.resize(image_len * filter_len, 0.0f);

        // update filters_size
        filters_size = filter_len * inner_len * sizeof(float);

        // update output_size
        output_size = image_len * filter_len * sizeof(float);

        // update real_vector_len
        real_vector_len = filters_real.size();

        // update previous_filter_len
        previous_filter_len = filter_len;

        // resize pooled_output
        pooled_output.clear();
        pool_len = (image_len * filter_len + pool_size - 1) / pool_size;
        pooled_output.resize(pool_len, 0.0f);


        if (filter_len > when_to_stop) {
            break;
        }


    }



}


int main() {
    getSystemInformation();
    std::cout << "\nPlease select an option:\n";
    std::cout << "1. Run with manual input\n";
    std::cout << "2. Run research\n";
    std::cout << "3. Exit\n";
    std::string input;
    std::cin >> input;
    if (input == "1") {
        manual();
    } else if (input == "2") {
        research();
    } else if (input == "3") {
        std::cout << "Exiting...\n";
    } else {
        std::cout << "Invalid input. Exiting...\n";
    }
    return 0;
    

}