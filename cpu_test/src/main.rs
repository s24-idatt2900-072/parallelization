mod config;
mod data;
mod processing;
mod utils;

use config::settings::*;
use processing::parallell::process_images_parallel;
use processing::sequential::process_images_sequential;
use utils::file_io::*;

fn main() {
    // Settings are set in the config/settings.rs file
    println!("Number of images: {}", NUM_IMAGES);
    println!("Image size: {}", IMAGE_SIZE);
    println!("Segment size: {}", SEGMENT_SIZE);
    println!("Output parallel path: {}", OUTPUT_PARALLELL_PATH);
    println!("Output sequential path: {}", OUTPUT_SEQUENTIAL_PATH);
    println!("Filters path: {}", FILTERS_PATH);

    println!(
        "Number of threads in the pool: {}",
        rayon::current_num_threads()
    );

    let mnist_images = data::mnist_data::load_mnist_dataset(NUM_IMAGES);

    println!("Loaded {} MNIST images.", mnist_images.train_images.len());

    let filters = read_filters_from_file(FILTERS_PATH).expect("Failed to read filters from file.");

    println!("Loaded {} filters.", filters.len());

    process_images_parallel(
        mnist_images.train_images.clone(),
        filters.clone(),
        SEGMENT_SIZE,
        OUTPUT_PARALLELL_PATH,
    );

    process_images_sequential(
        mnist_images.train_images.clone(),
        filters.clone(),
        SEGMENT_SIZE,
        OUTPUT_SEQUENTIAL_PATH,
    );
}
