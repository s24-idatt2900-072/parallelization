from data_loader import load_data, load_and_stack_data
from image_processing import process_images_with_filters
from pooling import max_pooling
import time

def main():
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

    num_images = 10
    num_filters = 100

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

    # print image 3
    print("Image 3:")
    print(images_stack[2])
    print()
    print()

    #print filter 3
    print("Abs Filter 3:")
    print(abs_filters_stack[2])
    print()
    print()

    #print filter 3
    print("Real Filter 3:")
    print(real_filters_stack[2])
    print()
    print()
    print("Processing images with filters...")
    # time the processing of images with filters
    start_time = time.time()
    processed_results = process_images_with_filters(images_stack, real_filters_stack, abs_filters_stack)

    print(f"Processing took {time.time() - start_time} seconds.")

    print(processed_results.shape)
    #print(processed_results)

    print("Result for image 1 with filter 1:", processed_results[0, 0])

    pool_size = 5

    print("Max pooling...")
    # time the max pooling operation
    start_time = time.time()
    #max_pooled_results = max_pooling(processed_results, pool_size)
    print(f"Max pooling took {time.time() - start_time} seconds.")

    #print(max_pooled_results.shape)

    print("Max pooling 1D...")
    # time the 1D max pooling operation
    start_time = time.time()
    max_pooled_results_1d = max_pooling(processed_results, pool_size)
    print(f"1D max pooling took {time.time() - start_time} seconds.")

    print(max_pooled_results_1d.shape)
    #print(max_pooled_results_1d)


if __name__ == "__main__":
    main()