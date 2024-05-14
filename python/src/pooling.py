import numpy as np


def max_pooling(array, pool_size):
    """
    Apply max pooling to each row of a 2D array, where each row represents one image with all filters.
    
    Args:
        array (np.array): A 2D numpy array where each row represents one image with all filters.

    Returns:
        np.array: A 2D numpy array with the pooled results.
    """

    num_images, num_filters = array.shape
    
    if num_filters < pool_size:
        pool_size = num_filters

    # calculate the number of pools
    num_complete_pools = num_filters // pool_size
    remainder = num_filters % pool_size
    
    if num_complete_pools:
        pooled_results_complete = np.max(array[:, :num_complete_pools * pool_size].reshape(num_images, num_complete_pools, pool_size), axis=2)
    else:
        pooled_results_complete = np.array([]).reshape(num_images, 0)

    if remainder != 0:
        pooled_results_remainder = np.max(array[:, num_complete_pools * pool_size:], axis=1)

        pooled_results = np.hstack([pooled_results_complete, pooled_results_remainder.reshape(num_images, 1)])
    else:
        pooled_results = pooled_results_complete
        
    return pooled_results