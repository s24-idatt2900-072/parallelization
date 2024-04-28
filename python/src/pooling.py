import numpy as np


def max_pooling(array, pool_size):
    """
    Apply 1D max pooling to each row of a 2D array.
    
    Args:
        array (np.array): A 2D numpy array where each row represents different filter responses for an image.
        pool_size (int): The size of the pooling window.

    Returns:
        np.array: A 2D numpy array with the pooled results.
    """
    num_images, num_filters = array.shape
    # Calculate the number of pooled elements
    pooled_length = num_filters // pool_size
    # Initialize the output array
    max_pooled = np.empty((num_images, pooled_length))

    for i in range(num_images):
        for j in range(pooled_length):
            start = j * pool_size
            end = start + pool_size
            max_pooled[i, j] = np.max(array[i, start:end])

    return max_pooled