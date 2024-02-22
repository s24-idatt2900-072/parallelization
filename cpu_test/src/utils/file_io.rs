use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead, Error, ErrorKind};
use std::path::Path;

pub fn read_filters_from_file(path: &str) -> io::Result<Vec<Vec<Vec<f32>>>> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut filters = Vec::new();
    let mut current_filter = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim() == "# New matrix" {
            if !current_filter.is_empty() {
                if !rows_are_equal_length(&current_filter) {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("Filter {} rows are not of equal length.", filters.len() + 1),
                    ));
                }
                filters.push(current_filter.clone());
                current_filter = Vec::new();
            }
        } else if !line.trim().is_empty() {
            let row: Vec<f32> = line
                .split(",")
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            current_filter.push(row);
        }
    }

    if !current_filter.is_empty() {
        if !rows_are_equal_length(&current_filter) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Filter {} rows are not of equal length.", filters.len() + 1),
            ));
        }
        filters.push(current_filter);
    }
    Ok(filters)
}

pub fn write_to_file(
    path: &str,
    images: &Vec<Vec<Vec<f32>>>,
    filters: &Vec<Vec<Vec<f32>>>,
    dot_product_results: &Vec<Vec<(usize, f32)>>,
    max_pooling_results: &Vec<Vec<(usize, f32)>>,
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
    for (_i, result) in dot_product_results.iter().enumerate() {
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

fn rows_are_equal_length(filter: &Vec<Vec<f32>>) -> bool {
    filter
        .get(0)
        .map(|first_row| filter.iter().all(|row| row.len() == first_row.len()))
        .unwrap_or(true)
}
