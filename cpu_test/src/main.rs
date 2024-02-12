use rayon::prelude::*;
use rand::Rng;


fn main() {
    let image_size = 29;

    let images: Vec<Vec<Vec<f32>>> = (0..5).map(|_| generate_random_matrix(image_size)).collect();
    let filters: Vec<Vec<Vec<f32>>> = (0..10).map(|_| generate_random_matrix(image_size)).collect();

    println!("Number of threads in the pool: {}", rayon::current_num_threads());

    println!("GO");

    let dot_product_results: Vec<Vec<(usize, f32)>> = images.par_iter().enumerate().map(|(_, image)| {
        filters.par_iter().enumerate().map(move |(j, filter)| {
            (j, apply_dot_product(&image, filter))
        }).collect::<Vec<(usize, f32)>>()
    }).collect();

    println!("Dot product results");
    println!("{:?}", dot_product_results);

    let segment_size = 2;

    let max_pooling_results: Vec<Vec<(usize, f32)>> = dot_product_results.par_iter().map(|dot_product_result| {
        max_pooling(dot_product_result, segment_size)
    }).collect();

    println!("Max pooling results");
    println!("{:?}", max_pooling_results);

    println!("DONE");

}

fn generate_random_matrix(size: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| {
        (0..size).map(|_| {
            rng.gen::<f32>()
        }).collect()
    }).collect()
}


fn apply_dot_product(image: &Vec<Vec<f32>>, filter: &Vec<Vec<f32>>) -> f32 {
    image.iter().zip(filter.iter()) // pairs each row of the img with the corresponding row in filter
         .map(|(img_row, filter_row)| {
             img_row.iter().zip(filter_row.iter()) // iterates over each element in the rows 
                    .map(|(img_val, filter_val)| img_val * filter_val)
                    .sum::<f32>() // sums product of a single row
         })
         .sum() // collects and adds up all the row dot products 
}

fn max_pooling(dot_product_result: &Vec<(usize, f32)>, segment_size: usize) -> Vec<(usize, f32)> {
    dot_product_result.chunks(segment_size).map(|segment| { // split the dot_product_result into chunks
        let segment_iter = segment.iter(); // create iterator

        let max_value_result = segment_iter.max_by(|(_, x),(_, y)| { // find max value in the segment using max_by()
            
            let comparison_result = x.partial_cmp(y); // compare x and y using partial_cmp() method

            comparison_result.unwrap()
        });

        max_value_result.unwrap().clone()

    }).collect()
}