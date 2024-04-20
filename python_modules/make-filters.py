import time
import numpy as np
from gaussian import gaussian

np.set_printoptions(precision=2)

def get_list_of_filters(size, sigma_values, num_of_x, num_of_y):
    
    #start timer here 
    start = time.time()

    N = size
    n1 = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
    n2 = n1.T
    n = np.array([n1, n2])

    len_x = len(num_of_x)
    len_y = len(num_of_y)
    len_sigma = len(sigma_values)
    output_size = len_x * len_y * len_sigma

    #kast faktor to int
    filters = np.zeros((output_size * 2, N , N))
    filter = 0

    # loop over all elemnts in the list of x and y values
    # and create a filter for each combination of x and y values with the sigma values
    for i in num_of_x:
        for j in num_of_y:
            for k in sigma_values:
                mu = np.array([i , j])
                sigma = k
                spatial_domain = gaussian(n, mu, sigma, True)
                filters[filter] = spatial_domain.real
                filters[filter + output_size] = spatial_domain.imag
                filter += 1  

    # end timer here
    end = time.time()
    print("tiden det tok å lage filterene i python ", end - start)
    return filters


def main():
    size = 29
    sigma_values = range(50, 51, 1)
    print("sigma_values", sigma_values)
    num_of_x = range(0, 2, 2)
    print("num_of_x", num_of_x)
    num_of_y = range(0, 1, 1)
    print("num_of_y", num_of_y)

    filters = get_list_of_filters(size, sigma_values, num_of_x, num_of_y)
    print("filters", filters.shape)

    start = time.time()
    with open('filters.csv', 'w') as file:
        for i in range(filters.shape[0]):
            # Write each 2D slice (matrix) to the file
            np.savetxt(file, filters[i], delimiter=',')
            # Optionally add an empty line or some marker after each matrix for readability
            file.write("\n# New matrix\n")
    end = time.time()
    print("tiden metoden i python brukte på å skrive til fil", end - start)

main()