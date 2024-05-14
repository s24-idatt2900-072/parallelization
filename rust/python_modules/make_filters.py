import time
import numpy as np
from gaussian_v2 import gaussian

np.set_printoptions(precision=2)

def get_list_of_filters(size, sigma_values, num_of_x, num_of_y):
    N = size
    n1 = np.arange(N).reshape((-1, 1)).repeat(N, axis=1) - (N - 1) / 2
    n2 = n1.T
    n = np.array([n1, n2])

    len_x = len(num_of_x)
    len_y = len(num_of_y)
    len_sigma = len(sigma_values)
    output_size = len_x * len_y * len_sigma

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
                filter_abs = np.abs(spatial_domain)
                # print("filter_abs", filter_abs[0])
                filter_real = spatial_domain.real
                # print("filter_real", filter_real[0])
                filet_imag = spatial_domain.imag
                # print("filet_imag", filet_imag[0])
                filter_real_flatten = filter_real.flatten()
                l2_norm = np.linalg.norm(filter_real_flatten)
                pre_prosessed_real = filter_real / l2_norm
                filters[filter] = filter_abs
                filters[filter + output_size] = pre_prosessed_real
                filter += 1

    return filters

def write_filters_to_file_seperetly(filters, output_size):
    with open('src/files/filters/filters_abs.csv', 'w') as file:
        for i in range(output_size):
            file.write(f"#{i}\n")
            np.savetxt(file, filters[i], delimiter=',')
        file.write("#end")
        file.close()
    
    with open('src/files/filters/filters_real.csv', 'w') as file:
        for i in range(output_size):
            file.write(f"#{i}\n")
            np.savetxt(file, filters[i + output_size], delimiter=',')
        file.write("#end")
        file.close()

def write_filters_to_file(filters, output_size):
    with open('filters.csv', 'w') as file:
        file.write(f"the amount of filters is {output_size * 2}\n")
        file.write(f"first abs then real, then new filter seperatet with #id\n")
        for i in range(output_size):
            file.write(f"#{i}\n")
            np.savetxt(file, filters[i], delimiter=',')
            file.write(f"#{i}\n")
            np.savetxt(file, filters[i + output_size], delimiter=',')
        file.write("#end")
        file.close()

def main():
    size = 29
    sigma_values = range(90, 110, 2) # 16 values from 92 to 108
    num_of_x = range(-10, 10, 2) # 25 values from -11 to 14
    num_of_y = range(-10, 10, 2) # 25 values from -11 to 14
    # 16 * 25 * 25 = 10000 filtere, med abs og real del
    print("sigma_values", len(sigma_values))
    print("num_of_x", len(num_of_x))
    print("num_of_y", len(num_of_y))
    output_size = len(num_of_x) * len(num_of_y) * len(sigma_values)
    print("output_size", output_size)

    filters = get_list_of_filters(size, sigma_values, num_of_x, num_of_y)
    write_filters_to_file_seperetly(filters, output_size)
    # write_filters_to_file(filters, output_size)

main()