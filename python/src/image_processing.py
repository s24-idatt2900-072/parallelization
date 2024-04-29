import numpy as np

# Process image using only numpy, no double loops
def dot_product(image, real_filter, abs_filter):
    """
    Process an image with given real and absolute filters using element-wise operations.
    
    Args:
        image (np.array): 29x29 numpy array representing an image.
        real_filter (np.array): 29x29 numpy array representing the real part of the filter.
        abs_filter (np.array): 29x29 numpy array representing the absolute part of the filter.

    Returns:
        np.array: Processed image output.
    """
    # Element-wise multiplication between the image and the absolute filter
    d = image * abs_filter

    # Element-wise multiplications for dot and norm calculations
    dot = np.sum(d * real_filter, axis=(0, 1))  # Summing up all the products for the entire matrix
    norm = np.sum(d * d, axis=(0, 1))  # Summing up all the squared elements for the entire matrix

    # Calculate the output using element-wise division
    output = dot / np.sqrt(norm)
    return output



def process_images_with_filters(images, real_filters, abs_filters):
    """
    
    Process images with given real and absolute filters using element-wise operations.
    Utilizes stack and numpy operations to avoid double loops.

    Args:
        images (np.array): 3D numpy array representing images.
        real_filters (np.array): 3D numpy array representing the real part of the filters.
        abs_filters (np.array): 3D numpy array representing the absolute part of the filters.

    Returns:
        np.array: 2D numpy array with 1D dot product results for each image againt all filters.
    
    """
    #print(f"Images shape: {images.shape}")
    num_images = images.shape[0]
    #print(f"Num images: {num_images}")
    num_filters = real_filters.shape[0]
    #print(f"Num filters: {num_filters}")
    results = np.empty((num_images, num_filters))
    #print(f"Results shape: {results.shape}")

    for filter_idx in range(num_filters):
        abs_filter = abs_filters[filter_idx]
        real_filter = real_filters[filter_idx]

        d = images * abs_filter[np.newaxis, :, :]

        dot = np.sum(d * real_filter[np.newaxis, :, :], axis=(1, 2))
        # norm = d * d

        norm = np.sum(d * d, axis=(1, 2))

        output = dot/np.sqrt(norm)
        results[:, filter_idx] = output

    # print if array is 1d or 2d
    #if len(results.shape) == 1:
    #    print("Result is 1D array")
    #else:
    #    print("Result is 2D array")
    return results