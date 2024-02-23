use std::vec;

use pyo3::prelude::*;
use pyo3::types::PyList;
use array2d::Array2D;
use image::{ImageBuffer, Luma};


#[derive(Debug)]
pub struct Filter {
    filter_real: Array2D<f64>,
    filter_imag: Array2D<f64>,
}

impl Filter {
    pub fn new(rows_real: Vec<Vec<f64>>, rows_imag: Vec<Vec<f64>>) -> Self {
        let filter_real = Array2D::from_rows(&rows_real).expect("REASON"); // fikse feil håndtering her
        let filter_imag = Array2D::from_rows(&rows_imag).expect("REASON"); // fikse feil håndtering her
        Self { filter_real,
            filter_imag,
        }
    }
}

fn map_value_to_color(normalized_value: f64) -> u8 {
    let g = (normalized_value * 255.0) as u8;
    g
}

fn visualize_filter(filter: &Filter, real: bool) -> Result<(), Box<dyn std::error::Error>> {

    let filter_to_make;

    if real {
        filter_to_make = &filter.filter_real;
    } else {
        filter_to_make = &filter.filter_imag;
    }

    let (image_width, image_height) = (filter_to_make.num_columns() as u32, filter_to_make.num_rows() as u32);
    let mut imgbuf: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(image_height, image_width);

    let max_value = filter_to_make.as_rows().iter().chain(filter.filter_imag.as_rows().iter())
        .flatten()
        .cloned()
        .fold(f64::MIN, f64::max);
    let min_value = filter_to_make.as_rows().iter().chain(filter.filter_imag.as_rows().iter())
        .flatten()
        .cloned()
        .fold(f64::MAX, f64::min);

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let i = y as usize; // rows
        let j = x as usize; // columns
    
        let value = filter_to_make[(i,j)]; 
        let normalized_value = (value - min_value) / (max_value - min_value);
        let color_value =  map_value_to_color(normalized_value);
    
        // Set the pixel in the image buffer.
        *pixel = Luma([color_value]);
    }
    
    imgbuf.save("filter.png").unwrap();
    Ok(())
}


fn main() -> PyResult<()> {

    Python::with_gil(|py| {
        let sys = PyModule::import(py, "sys")?;

        let path: &PyList = sys.getattr("path")?.downcast()?;

        path.insert(0, "../python_modules")?; // Update this path

        let gaussian_module = PyModule::import(py, "gaussian")?;
        let mu = vec![20, 20];
        let sigma = 100;
        let size = 101;
        let use_log = true;

        let args = (size, mu, sigma, use_log);
        
        let (result_real, result_imag): (Vec<Vec<f64>>, Vec<Vec<f64>>) = gaussian_module
            .getattr("gaussian")?
            .call1(args)?
            .extract()?;

        let filter: Filter = Filter::new(result_real, result_imag);
        let mut filters: Vec<Filter> = vec![];
        filters.push(filter);

        let real = false;
        let _ = visualize_filter(&filters[0], real);

        Ok(())
    })
}