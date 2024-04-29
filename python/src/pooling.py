import numpy as np


def max_pooling(array, pool_size):
    """
    Apply max pooling to each row of a 2D array, where each row represents one image with all filters.
    
    Args:
        array (np.array): A 2D numpy array where each row represents one image with all filters.

    Returns:
        np.array: A 2D numpy array with the pooled results.
    """
    #print("Max pooling...")
    #print("Input array:")
    #print(array) # 2D array containing the 1D array of the dot produt results of each image with all filters
    #print(array.shape) # (10, 10) 10 images with 10 filters

    num_images, num_filters = array.shape
    #print(f"Number of images: {num_images}")
    #print(f"Number of filters per image: {num_filters}")
    
    if num_filters < pool_size:
        print(f"Adjusting pool size from {pool_size} to {num_filters} as there are fewer filters than the pool size.")
        pool_size = num_filters

    # calculate the number of pools
    num_pools = num_filters // pool_size
    #print(f"Number of pools per image: {num_pools}")

    # create an empty array to store the pooled results 
    # (each row will contain the pooled results of one image)
    # therefore size must be adjusted based on the size of 1D array and pool size
    pooled_results = np.zeros((num_images, num_pools))

    # iterate over each image
    for i in range(num_images):
        # iterate over each pool
        for j in range(num_pools):
            # get the start and end indices of the pool
            start = j * pool_size
            end = start + pool_size
            # get the pool
            pool = array[i, start:end]
            # get the max value in the pool
            max_value = np.max(pool)
            # store the max value in the pooled results
            pooled_results[i, j] = max_value

    #print(f"pooled_results shape: {pooled_results.shape}")

    # flatten the pooled results
    pooled_results = pooled_results.flatten()

    #print(f"pooled_results shape: {pooled_results.shape}")
    #print(pooled_results)

    return pooled_results