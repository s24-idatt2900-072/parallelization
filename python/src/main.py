from data_loader import load_data, load_and_stack_data
from image_processing import process_images_with_filters
from pooling import max_pooling
from file_writer import write_to_file
import time
import sys
import numpy as np

def main():
    initial_ui()

def research():
    """
    Will do the research run
    start amount of image set by the user
    start with 20 filters, do 60 runs with those configurations, then increase by 1 and continue indefinitely

    logging will be done to a csv file
    columns: Filter, ID, Time_ms, Average_time
    filter will be the filter number for the 60 runs
    id will be the id of the run, starting at 1 and go on to 60
    time_ms will be the time it took to run the process_images_with_filters function and max_pooling function
    average_time will be the average time of the 60 runs for that filter amount
    """
    
    print("Old way or new way?")
    print("1. Old way")
    print("2. New way")
    method = 0
    while method < 1 or method > 2:
        print("Enter the method number:")
        method = int(input())
        if method < 1 or method > 2:
            print("Invalid method number.")

    print("Enter the amount of images to process:")
    num_images = 0

    while num_images < 1:
        print("Enter the number of images:")
        num_images = int(input())
        if num_images < 1:
            print("Invalid amount of images.")


    print("Loading data...")

    start_time = time.time()
    # Load MNIST dataset
    abs_filters = load_data("files/filters/filters_abs.csv")
    real_filters = load_data("files/filters/filters_real.csv")
    mnist_images = load_data("files/mnist/mnist_padded_29x29.csv")
    print(f"Data loading took {time.time() - start_time} seconds.")

    # file with unique id using unix timestamp
    file_name = f"platform-X-python-{int(time.time())}.csv"

    filter_amount = 10

    increment = 10

    pool_size = 500

    data_to_write = [("Filter", "ID", "Time_us", "Average_time")]

    images_stack = load_and_stack_data(mnist_images, num_images)

    while True:
        prev_filter_amount = filter_amount
        time_us = []
        print(f"Processing with {filter_amount} filters...")
        for i in range(1, 61):
            abs_filters_stack = load_and_stack_data(abs_filters, filter_amount)
            real_filters_stack = load_and_stack_data(real_filters, filter_amount)
            
            start_time = time.time()

            if method == 1:
                # The Old way
                processed_results = process_images_with_filters(images_stack, real_filters_stack, abs_filters_stack)
            else:
                # The New way
                d = images_stack[:, np.newaxis, :] * abs_filters_stack
                dot = np.sum(d * real_filters_stack[np.newaxis, :, :], axis=(2,3))
                norm = np.sum(d * d, axis=(2,3))
                processed_results = dot / np.sqrt(norm)

            # asserts that the shape of the processed_results is correct numimgaes, num_filters
            assert processed_results.shape == (num_images, filter_amount)
            
            max_pooled_results = max_pooling(processed_results, pool_size)
            time_spent_us = int(round((time.time() - start_time) * 1000000))
            time_us.append(time_spent_us)
            if i == 1 or filter_amount != prev_filter_amount:
                data_to_write.append((filter_amount, i, time_spent_us, 0))
            else:
                data_to_write.append((0, i, time_spent_us, 0))
        average_time = int(round(sum(time_us) / len(time_us)))
        data_to_write.append((0, 0, 0, average_time))
        write_to_file(file_name, data_to_write)
        data_to_write = []
        filter_amount += increment
        prev_filter_amount = filter_amount
        if filter_amount > 2500:
            break




def manual():
    """
    mathematical ops for machine learning method processing

    image: 29x29 numpy array
    real_filter: 29x29 numpy array
    abs_filter: 29x29 numpy array

    d = image (element-wise multiplication) abs_filter
    dot = dot + d (element-wise multiplication) real_filter
    norm = norm + d (element-wise multiplication) d

    output = dot / sqrt(norm)

    """

    num_images = 0
    while num_images < 1:
        print("Enter the number of images:")
        num_images = int(input())
        if num_images < 1:
            print("Invalid amount of images.")

    num_filters = 0
    while num_filters < 1:
        print("Enter the number of filters:")
        num_filters = int(input())
        if num_filters < 1:
            print("Invalid amount of filters.")

    print("Loading data...")


    start_time = time.time()
    # Load MNIST dataset
    abs_filters = load_data("files/filters/filters_abs.csv")
    real_filters = load_data("files/filters/filters_real.csv")
    mnist_images = load_data("files/mnist/mnist_padded_29x29.csv")
    print(f"Data loading took {time.time() - start_time} seconds.")

    start_time = time.time()
    images_stack = load_and_stack_data(mnist_images, num_images)
    abs_filters_stack = load_and_stack_data(abs_filters, num_filters)
    real_filters_stack = load_and_stack_data(real_filters, num_filters)

    print(f"Data stacking took {time.time() - start_time} seconds.")
    print("Processing images with filters...")

    total_start_time = time.time()
    dot_start_time = time.time()
    processed_results = process_images_with_filters(images_stack, real_filters_stack, abs_filters_stack)
    dot_end_time = time.time()

    pool_size = 500


    max_start_time = time.time()
    max_pooled_results = max_pooling(processed_results, pool_size)
    max_end_time = time.time()
    
    print(f"Total time: {time.time() - total_start_time} seconds.")
    print(f"Dot product took {dot_end_time - dot_start_time} seconds.")
    print(f"Max pooling took {max_end_time - max_start_time} seconds.")

    print("Done.")


def initial_ui():
    print("Select mode:")
    print("1. Manual run")
    print("2. Research run")
    print("3. Exit")
    mode = input("Enter mode: ")
    if mode == "1":
        manual()
    elif mode == "2":
        research()
    elif mode == "3":
        print("Exiting...")
        exit()
    else:
        print("Invalid mode.")
        initial_ui()

if __name__ == "__main__":
    main()
