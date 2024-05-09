use rayon::prelude::*;
use std::time::Instant;

use crate::utils::file_io::write_to_file;
use crate::utils::matrix_ops::apply_dot_product;
use crate::utils::matrix_ops::max_pooling;

pub fn process_images_parallel(
    images: Vec<Vec<Vec<f32>>>,
    filters: Vec<Vec<Vec<f32>>>,
    segment_size: usize,
    output_path: &str,
) -> Vec<Vec<(usize, f32)>> {
    println!("GO PARALLEL");
    let start = Instant::now();

    let dot_product_results: Vec<Vec<(usize, f32)>> = images
        .par_iter()
        .enumerate()
        .map(|(_, image)| {
            filters
                .par_iter()
                .enumerate()
                .map(move |(j, filter)| (j, apply_dot_product(image, filter)))
                .collect::<Vec<(usize, f32)>>()
        })
        .collect();

    let max_pooling_results: Vec<Vec<(usize, f32)>> = dot_product_results
        .par_iter()
        .map(|dot_product_result| max_pooling(dot_product_result, segment_size))
        .collect();

    println!("DONE");
    println!("Elapsed time: {:?}", start.elapsed());

    if let Err(e) = write_to_file(
        output_path,
        &images,
        &filters,
        &dot_product_results,
        &max_pooling_results,
    ) {
        eprintln!("Error writing to file: {}", e);
    }

    max_pooling_results
}
