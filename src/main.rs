use cpu_test::{utils::*, *};
use parallelization::research;
use wgpu_test::*;
use wgsl::*;
use std::env;

const MNIST_PATH: &str = "cpu_test/data/";
const FILTER_PATH: &str = "src/files/filters/";
const MAX_MNIST: u32 = 50_000;

const NR_IMG: usize = 1_000;
const NR_FILTER: usize = 1_000;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run <method>");
        println!("Available methods:");
        println!("  cpu - Runs CPU method");
        println!("  gpu - Runs GPU method, one image with all filters");
        println!("  gpu-par - Runs GPU method with parallel shader, all images with all filters");
        println!("  gpu-loop - Runs GPU method with loop shader, all images with all filters");
        return;
    }
    println!("Initializing data..");

    println!("Loading MNIST..");
    let images = data::mnist_data::load_mnist_dataset_flattened(MAX_MNIST, MNIST_PATH);
    let images = adjust_length(images, NR_IMG);
    println!("Done loading..");

    println!("Reading filters..");
    let (re, abs) =
        file_io::read_filters_from_file_flattened(FILTER_PATH).expect("Could not read filters");
    let re = adjust_length(re, NR_FILTER);
    let abs = adjust_length(abs, NR_FILTER);
    println!("Done reading..");

    println!("Loaded {} images & {} filters", images.len(), re.len());
    let gpu = Extractor::new().unwrap();

    let method = &args[1];
    match method.as_str() {
        "cpu" => {
            println!("Computing CPU");
            research::run_research_cpu(&images, &abs, &re, 500);
        },
        "gpu" => {
            println!("Computing GPU");
        },
        "gpu-par" | "gpu-loop" => {
            let max_chunk = 500;
            let ilen = images[0].len();

            let images = flatten_content(images);
            let re = flatten_content(re);
            let abs = flatten_content(abs);
            let (cosine_shader, max_shader) = match method.as_str() {
                "gpu-par" => {
                    println!("Computing GPU with parallel shader");
                    let wg_size = (253, 1, 1);
                    let chunk = 10;
                    (get_parallel_cosine_similarity_shader(NR_IMG, NR_FILTER, ilen, chunk, wg_size).to_string(),
                    get_parallel_max_pool_shader(max_chunk, chunk, wg_size).to_string()
                    )
                },
                "gpu-loop" => {
                    println!("Computing GPU with loop shader");
                    let wg_size = (16, 16, 1);
                    (get_for_loop_cosine_similarity_shader(NR_IMG, NR_FILTER, ilen, wg_size).to_string(),
                    get_for_loop_max_pool_shader(max_chunk, wg_size).to_string()
                )
                },
                _ => panic!("Invalid"),
            };
                research::run_research_gpu(
                    &method,
                    &images,
                    &re,
                    &abs,
                    &cosine_shader,
                    &max_shader,
                    ilen,
                    max_chunk,
                    &gpu,
                );
        },
        _ => {
            println!("Invalid method");
            return;
        }
    }
    println!("Done!");
}

fn adjust_length<T>(list: Vec<T>, to: usize) -> Vec<T>
where
    T: Clone,
{
    let mut it = list.iter().cycle();
    let list = vec![0; to]
        .iter()
        .map(|_| it.next().unwrap().clone())
        .collect::<Vec<T>>();
    list
}

fn flatten_content<T>(content: Vec<Vec<T>>) -> Vec<T> {
    content.into_iter().flatten().collect()
}
