use csv::ReaderBuilder;
use plotters::prelude::*;
use serde_derive::Deserialize;
use std::error::Error;
use std::fs::File;

#[derive(Deserialize, Debug)]
struct BenchmarkData {
    benchmark: String,
    input_size: u32,
    time_ms: f64,
    device: String,
}

const OUT_FILE_NAME: &str = "running_time_comparison.svg";

fn load_benchmark_data(file_path: &str) -> Result<Vec<BenchmarkData>, Box<dyn Error>> {
    println!("Current directory: {:?}", std::env::current_dir()?);
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut benchmark_data: Vec<BenchmarkData> = Vec::new();

    for result in rdr.deserialize() {
        let record: BenchmarkData = result?;
        benchmark_data.push(record);
    }

    for data in benchmark_data.iter() {
        println!("{:?}", data);
    }

    Ok(benchmark_data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "./files/mock_dot_product_benchmark_data.csv";
    //let file_path = "../files/benchmark_data.csv";
    let benchmark_data = load_benchmark_data(file_path)?;

    let max_input_size = benchmark_data
        .iter()
        .max_by_key(|d| d.input_size)
        .unwrap()
        .input_size as f64;
    let max_time_ms = benchmark_data
        .iter()
        .max_by(|a, b| a.time_ms.partial_cmp(&b.time_ms).unwrap())
        .unwrap()
        .time_ms;

    let x_range = 0.0..(max_input_size + 1.0);
    let y_range = 0.0..(max_time_ms + (max_time_ms * 0.1));

    let root = SVGBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, _) = root.split_vertically(768 - 18);

    let mut chart = ChartBuilder::on(&upper)
        .caption("CPU vs GPU Computational Running Time", ("sans-serif", 50))
        .set_label_area_size(LabelAreaPosition::Left, 50)
        .set_label_area_size(LabelAreaPosition::Bottom, 50)
        .margin(10)
        .build_cartesian_2d(x_range, y_range)?;

    chart
        .configure_mesh()
        .x_desc("Computation Index (Input size)")
        .y_desc("Running Time (Ms)")
        .draw()?;

    let cpu_data: Vec<_> = benchmark_data
        .iter()
        .filter(|&d| d.device == "CPU")
        .collect();
    let gpu_data: Vec<_> = benchmark_data
        .iter()
        .filter(|&d| d.device == "GPU")
        .collect();

    let mut cpu_data_sorted = cpu_data.clone();
    let mut gpu_data_sorted = gpu_data.clone();
    cpu_data_sorted.sort_by_key(|d| d.input_size);
    gpu_data_sorted.sort_by_key(|d| d.input_size);

    chart
        .draw_series(LineSeries::new(
            cpu_data_sorted
                .iter()
                .map(|d| (d.input_size as f64, d.time_ms)),
            &RED,
        ))?
        .label("CPU")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            gpu_data_sorted
                .iter()
                .map(|d| (d.input_size as f64, d.time_ms)),
            &BLUE,
        ))?
        .label("GPU")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    root.present().expect("Unable to write result to file");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}
