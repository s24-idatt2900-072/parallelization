#include "common.h"

int main() {
    const int num_images = 1000;
    const int num_filters = 1000;
    const int image_size = 29;
    const int pool_size = 500;

    std::vector<float> images(num_images * image_size * image_size);
    std::vector<float> real_filters(num_filters * image_size * image_size);
    std::vector<float> abs_filters(num_filters * image_size * image_size);

    loadDataFromFile("mnist/mnist_padded_29x29.csv", images);
    loadDataFromFile("filters/filters_real.csv", real_filters);
    loadDataFromFile("filters/filters_abs.csv", abs_filters);

    std::vector<float> output(num_images * num_filters);
    cosineSimiliarity(images.data(), real_filters.data(), abs_filters.data(), output.data(), num_images, num_filters, image_size * image_size);

    std::vector<float> pooled_results(num_images * ((num_filters / pool_size) + (num_filters % pool_size ? 1 : 0)));
    maxPooling(output.data(), pooled_results.data(), num_images, num_filters, pool_size);

    int num_results_per_image = num_filters / pool_size + (num_filters % pool_size ? 1 : 0);

    writeDataToFile("output.csv", pooled_results, num_images, num_results_per_image);

    return 0;
}