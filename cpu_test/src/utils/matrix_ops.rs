use rand::Rng;

#[allow(dead_code)]
pub fn generate_random_matrix(size: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| (0..size).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

pub fn apply_dot_product(image: &[Vec<f32>], filter: &[Vec<f32>]) -> f32 {
    image
        .iter()
        .zip(filter.iter()) // pairs each row of the img with the corresponding row in filter
        .map(|(img_row, filter_row)| {
            img_row
                .iter()
                .zip(filter_row.iter()) // iterates over each element in the rows
                .map(|(img_val, filter_val)| img_val * filter_val)
                .sum::<f32>() // sums product of a single row
        })
        .sum() // collects and adds up all the row dot products
}

pub fn max_pooling(dot_product_result: &[(usize, f32)], segment_size: usize) -> Vec<(usize, f32)> {
    dot_product_result
        .chunks(segment_size)
        .map(|segment| {
            // split the dot_product_result into chunks
            let segment_iter = segment.iter(); // create iterator

            let max_value_result = segment_iter.max_by(|(_, x), (_, y)| {
                // find max value in the segment using max_by()

                let comparison_result = x.partial_cmp(y); // compare x and y using partial_cmp() method

                comparison_result.unwrap()
            });

            *max_value_result.unwrap()
        })
        .collect()
}
