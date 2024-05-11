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

    /*
    if (initial_image_count < image_len) {
        expandVector(images, initial_image_count * IMAGE_SIZE, (image_len - initial_image_count) * IMAGE_SIZE);
    } else {
        images.resize(image_len * IMAGE_SIZE);
    }
    if (initial_filter_count < filter_len) {
        expandVector(filters_real, initial_filter_count * IMAGE_SIZE, (filter_len - initial_filter_count) * IMAGE_SIZE);
        expandVector(filters_abs, initial_filter_count * IMAGE_SIZE, (filter_len - initial_filter_count) * IMAGE_SIZE);
    } else {
        filters_real.resize(filter_len * IMAGE_SIZE);
        filters_abs.resize(filter_len * IMAGE_SIZE);
    }
    */

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
                    //output.data(),
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


    // print size of pooled output
    std::cout << "Size of pooled output: " << pooled_output.size() << std::endl;


    // print all of pooled_output for loop

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

    /*
    if (initial_image_count < image_len) {
        expandVector(images, initial_image_count * IMAGE_SIZE, (image_len - initial_image_count) * IMAGE_SIZE);
    } else {
        images.resize(image_len * IMAGE_SIZE);
    }
    if (initial_filter_count < filter_len) {
        expandVector(filters_real, initial_filter_count * IMAGE_SIZE, (filter_len - initial_filter_count) * IMAGE_SIZE);
        expandVector(filters_abs, initial_filter_count * IMAGE_SIZE, (filter_len - initial_filter_count) * IMAGE_SIZE);
    } else {
        filters_real.resize(filter_len * IMAGE_SIZE);
        filters_abs.resize(filter_len * IMAGE_SIZE);
    }
    */

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

        for (unsigned int i = 1; i < 31; ++i) {
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
                outFile << line << std::endl; // Write each element followed by a newline
            }
            outFile.close(); // Close the file after writing
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



/*
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

    size_t memory_used_MB_dot_product, memory_free_MB_dot_product;
    size_t memory_used_MB_max_pool, memory_free_MB_max_pool;


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
        filter_len,
        memory_used_MB_dot_product,
        memory_free_MB_dot_product
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

    unsigned int pool_size = 500;

    unsigned int pool_len = (image_len * filter_len + pool_size - 1) / pool_size;

    std::cout << "Pool size: " << pool_size << std::endl;
    std::cout << "Pool len: " << pool_len << std::endl;

    std::vector<float> pooled_output(pool_len, 0.0f);

    runMaxPool(
        output.data(),
        pooled_output.data(),
        output_size,
        pool_size,
        pool_len,
        memory_used_MB_max_pool,
        memory_free_MB_max_pool
    );


    // print size of pooled output
    std::cout << "Size of pooled output: " << pooled_output.size() << std::endl;


    // print all of pooled_output for loop

    std::cout << "Results:\n";
    for (unsigned int i = 0; i < pooled_output.size(); ++i) {
        std::cout << "Pooled output[" << i << "] = " << pooled_output[i] << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << std::endl;

    return 0;
}
*/

/*
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

    size_t memory_used, memory_free;
    

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
    std::string file_name = person + "_CPP_GPU_img_" + std::to_string(image_len) + "_" + std::to_string(when_to_stop) + "_" + std::to_string(std::time(0)) + ".csv";

    unsigned int previous_filter_len = filter_len;


    unsigned int pool_len = (image_len * filter_len + pool_size - 1) / pool_size;

    // buffer
    std::vector<std::string> buffer;

    // add ("Filter", "ID", "Time_us", "Average_time") which is the column names of the csv file
    buffer.push_back("Filter, ID, Time_us, Average_time, Memory_used_MiB, Memory_free_MiB");

    while (true) {
        previous_filter_len = filter_len;
        // vector called time_us containing time taken in us for each dot product and max pool operation
        std::vector<unsigned int> time_us_vec;
        unsigned int time_us = 0;
        // only print if it wont slow down processing speed std::cout << "Processing " << image_len << " images with " << filter_len << " filters...\n";
        if (filter_len > 5000) {
            std::cout << "Processing " << image_len << " images with " << filter_len << " filters...\n";
        }


        // it will be done 30 times to get an average time and ensure standard deviation 

        if (filter_len < pool_size){
            pool_size = filter_len;
        }


        std::vector<float> pooled_output(pool_len, 0.0f);
        
        // data will be written to a csv file over the course of processing
        // adds data to a form of buffer to prevent writing to file every time
        // buffer will hold it in string format
        // add to write bufer to file (Filter, ID, Time_us, Average_time_)

        
        
        for (unsigned int i = 1; i < 31; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            runCombinedOperations(
                images.data(),
                filters_real.data(),
                filters_abs.data(),
                //output.data(),
                pooled_output.data(),
                images_size,
                images_vector_len,
                real_vector_len,
                filters_size,
                output_size,
                pool_len * sizeof(float),
                inner_len,
                image_len,
                filter_len,
                pool_size,
                pool_len,
                memory_used,
                memory_free
            );

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = end - start;
            time_us = (unsigned int)std::round(duration.count());
            time_us_vec.push_back(time_us);
            if (i == 1 || filter_len != previous_filter_len) {
                // add to write bufer to file (filter amount, i, time spent in us, 0)
                buffer.push_back(std::to_string(filter_len) + ", " + std::to_string(i) + ", " + std::to_string(time_us) + ", 0" + ", " + std::to_string(memory_used) + ", " + std::to_string(memory_free));
            } else {
                // add to write bufer to file (0, i, time spent in us, 0)
                buffer.push_back("0, " + std::to_string(i) + ", " + std::to_string(time_us) + ", 0" + ", " + std::to_string(memory_used) + ", " + std::to_string(memory_free));
            }

            std::cout << "Results:\n";

            for (unsigned int i = 0; i < output.size(); ++i) {
                std::cout << "output[" << i << "] = " << output[i] << std::endl;
            }
            
            for (unsigned int i = 0; i < pooled_output.size(); ++i) {
                std::cout << "Pooled output[" << i << "] = " << pooled_output[i] << std::endl;
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
        for (float t : time_us_vec) {
            sum += t;
        }
        unsigned int average_time = (unsigned int)std::round(sum / time_us_vec.size());
        if (average_time == 0) {
            average_time = 1;
        }
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
            std::cout << "Results written to " << file_name << std::endl;
            break;
        }   

        // write out pooled_output using a for loop
        // print all of pooled_output for loop

        std::cout << "Results:\n";
        for (unsigned int i = 0; i < pooled_output.size(); ++i) {
            std::cout << "Pooled output[" << i << "] = " << pooled_output[i] << std::endl;
        }

    }



}
*/

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