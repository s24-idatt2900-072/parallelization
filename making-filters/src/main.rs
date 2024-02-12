use std::vec;

use pyo3::prelude::*;
use pyo3::types::PyList;

#[derive(Debug)]
pub struct Filter {
    filter: Vec<Vec<f64>>,
    shape: (usize, usize),
}

impl Filter {
    pub fn new(filter: Vec<Vec<f64>>) -> Self {
        let shape = (filter.len() , filter[0].len());
        Self {  filter,
                shape,
             }
    }
}

fn main() -> PyResult<()> {


    Python::with_gil(|py| {
        let sys = PyModule::import(py, "sys")?;

        let path: &PyList = sys.getattr("path")?.downcast()?;
        println!("");
        println!("Python sys.path: {:?}", path);

        path.insert(0, "../../python_modules")?; // Update this path

        let gaussian_module = PyModule::import(py, "gaussian")?;
        let mu = vec![40, 40];

        let args = (3, mu, 10, true);
        
        let result: Vec<Vec<f64>> = gaussian_module
            .getattr("gaussian")?
            .call1(args)?
            .extract()?;
        let filter: Filter = Filter::new(result);
        let mut filters: Vec<Filter> = vec![];
        filters.push(filter);
        
        println!("Gaussian result: {:?}", filters.len());
        println!("Gaussian result: {:?}", filters[0].shape);
        println!("Gaussian result: {:?}", filters[0]);


        Ok(())
    })
}