use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use std::fs::File;
use std::io::{self, BufRead};
use mnist::*;
use ndarray::{Array, Array3, s};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

fn main() {
    let image_size = 29;
    let mut output_path = "files/output_parallell.csv";
    let segment_size = 3;



    let Mnist { trn_img, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let num_images = 5;

    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.0);

    let images: Vec<Vec<Vec<f32>>> = (0..num_images)
    .map(|image_num| {
        let image = train_data.slice(s![image_num, .., ..]);
        // Create a new 29x29 array filled with 0s for padding.
        let mut padded_image = Array::zeros((29, 29));
        // Copy the original image data into the top-left corner of the padded image.
        padded_image.slice_mut(s![0..28, 0..28]).assign(&image);
        // Convert the padded image to Vec<Vec<f32>>
        padded_image.outer_iter().map(|row| row.to_vec()).collect()
    })
    .collect();

    println!("Number of extracted images: {}", images.len());

    // print vizualization of the first image
    for (i, pixel) in images[0].iter().enumerate() {
        if i % 29 == 0 {
            println!();
        }
        print!("{:.?} ", pixel);
    }

    // print len of images
    println!("{:?}", images.len());
    
    println!();
    let num_values = images[0].iter().map(|row| row.len()).sum::<usize>();
    println!("Number of numerical values in the first image: {}", num_values);


    let path = "files/filters.csv";

    let filters = read_filters_from_file(path).expect("Failed to read filters");

    println!("Successfully read {} filters.", filters.len());

    /*
    //print image
    for (i, image) in images.iter().enumerate() {
        println!("Image {}", i);
        for (j, row) in image.iter().enumerate() {
            println!("Row {}: {:?}", j, row);
        }
    }
    */

    // prints filters
    /*
    for filter in &filters {
        println!("Filter:");
        for row in filter {
            println!("{:?}", row);
        }
    }
    */

    for (i, filter) in filters.iter().enumerate() {
        if !filter.is_empty() {
            let rows = filter.len();
            let cols = filter[0].len();
            println!("Filter {}: Size = {}x{}", i+1, rows, cols);

            // Optionally verify that all rows are of equal length (i.e., a proper matrix)
            let all_rows_equal = filter.iter().all(|row| row.len() == cols);
            if !all_rows_equal {
                println!("Warning: Not all rows in Filter {} are of equal length.", i+1);
            }
        } else {
            println!("Filter {} is empty.", i+1);
        }
    }

    //let images: Vec<Vec<Vec<f32>>> = (0..5).map(|_| generate_random_matrix(image_size)).collect();
    /*
    let filters: Vec<Vec<Vec<f32>>> = (0..10000)
        .map(|_| generate_random_matrix(image_size))
        .collect();
    */

    println!(
        "Number of threads in the pool: {}",
        rayon::current_num_threads()
    );

    println!("GO PARALLEL");
    let start = Instant::now();

    let dot_product_results: Vec<Vec<(usize, f32)>> = images
        .par_iter()
        .enumerate()
        .map(|(_, image)| {
            filters
                .par_iter()
                .enumerate()
                .map(move |(j, filter)| (j, apply_dot_product(&image, filter)))
                .collect::<Vec<(usize, f32)>>()
        })
        .collect();

    //println!("Dot product results");
    //println!("{:?}", dot_product_results);


    let max_pooling_results: Vec<Vec<(usize, f32)>> = dot_product_results
        .par_iter()
        .map(|dot_product_result| max_pooling(dot_product_result, segment_size))
        .collect();

    println!("Max pooling results");
    println!("{:?}", max_pooling_results);

    println!("DONE");
    println!("Elapsed time: {:?}", start.elapsed());

    // print each max pool result 

    for result in &max_pooling_results {
        println!("{:?}", result);
    }


    if let Err(e) = write_to_file(output_path, &images, &filters, &dot_product_results, &max_pooling_results) {
        eprintln!("Failed to write to file: {}", e);
    }

    println!("GO SEQUENTIAL");
    let start = Instant::now();

    let dot_product_results: Vec<Vec<(usize, f32)>> = images
        .iter()
        .enumerate()
        .map(|(_, image)| {
            filters
                .iter()
                .enumerate()
                .map(move |(j, filter)| (j, apply_dot_product(&image, filter)))
                .collect::<Vec<(usize, f32)>>()
        })
        .collect();

    //println!("Dot product results");
    //println!("{:?}", dot_product_results);

    let max_pooling_results: Vec<Vec<(usize, f32)>> = dot_product_results
        .iter()
        .map(|dot_product_result| max_pooling(dot_product_result, segment_size))
        .collect();

    println!("Max pooling results");
    println!("{:?}", max_pooling_results);

    println!("DONE");
    println!("Elapsed time: {:?}", start.elapsed());

    output_path = "files/output_sequential.csv";
    if let Err(e) = write_to_file(output_path, &images, &filters, &dot_product_results, &max_pooling_results) {
        eprintln!("Failed to write to file: {}", e);
    }

}

fn generate_random_matrix(size: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| (0..size).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn apply_dot_product(image: &Vec<Vec<f32>>, filter: &Vec<Vec<f32>>) -> f32 {
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

fn max_pooling(dot_product_result: &Vec<(usize, f32)>, segment_size: usize) -> Vec<(usize, f32)> {
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

            max_value_result.unwrap().clone()
        })
        .collect()
}


fn read_filters_from_file(path: &str) -> io::Result<Vec<Vec<Vec<f32>>>> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut filters = Vec::new();
    let mut current_filter = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim() == "# New matrix" {
            if !current_filter.is_empty() {
                filters.push(current_filter.clone());
                current_filter = Vec::new();
            }
        }
        else if !line.trim().is_empty() {
            let row: Vec<f32> = line
                .split(",")
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            current_filter.push(row);
        }
    }

    if !current_filter.is_empty() {
        filters.push(current_filter);
    }
    Ok(filters)
}

fn write_to_file(
    path: &str,
    images: &Vec<Vec<Vec<f32>>>,
    filters: &Vec<Vec<Vec<f32>>>,
    dot_product_results: &Vec<Vec<(usize, f32)>>,
    max_pooling_results: &Vec<Vec<(usize, f32)>>
) -> io::Result<()> {
    let path = Path::new(path);
    let mut file = File::create(path)?;

    for (index, image) in images.iter().enumerate() {
        writeln!(file, "\n# Image {}", index)?;
        for row in image {
            writeln!(file, "{:?}", row)?;
        }
    }

    for (index, filter) in filters.iter().enumerate() {
        writeln!(file, "\n# Filter {}", index)?;
        for row in filter {
            writeln!(file, "{:?}", row)?;
        }
    }

    // Write dot product results
    writeln!(file, "\n# Dot Product")?;
    for (i, result) in dot_product_results.iter().enumerate() {
        writeln!(file, "Image @ Filter: {:?}", result)?;
    }

    // Write max pooling results
    writeln!(file, "\n# Max Pooling")?;
    for (i, result) in max_pooling_results.iter().enumerate() {
        writeln!(file, "{}: {:?}", i, result)?;
    }

    file.flush()?; // Ensure all data is written to disk

    Ok(())
}