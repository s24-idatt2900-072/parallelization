use cpu_test::{utils::*, *};
use parallelization::research;
use std::env;
use wgpu_test::*;
use wgsl::*;

const MNIST_PATH: &str = "cpu_test/data/";
const FILTER_PATH: &str = "src/files/filters/";
const MAX_MNIST: u32 = 50_000;

fn main() {
    let mut nr_imgs: usize = 1_000;
    let mut nr_filters: usize = 1_000;
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run <method>");
        println!("Available methods:");
        println!("  cpu - Runs CPU method");
        println!("  gpu - Runs GPU method, one image with all filters");
        println!("  gpu-par - Runs GPU method with parallel shader, all images with all filters");
        println!("  gpu-loop - Runs GPU method with loop shader, all images with all filters");
        println!("  \"nr images\" - Argument 2, number of images to use");
        println!("  \"nr filters\" - Argument 3, number of filters to use");
        return;
    }
    if args.len() == 2 {
        println!("Using default values for number of filters and images");
    } else if args.len() == 3 {
        nr_imgs = args[2].parse().unwrap();
    } else if args.len() == 4 {
        nr_imgs = args[2].parse().unwrap();
        nr_filters = args[3].parse().unwrap();
    }

    let mut mnist_imgs = MAX_MNIST;
    if nr_filters < mnist_imgs as usize {
        mnist_imgs = nr_filters as u32;
    }

    println!("Initializing data..");

    println!("Loading MNIST..");
    let images = data::mnist_data::load_mnist_dataset_flattened(mnist_imgs, MNIST_PATH);
    let images = adjust_length(images, nr_imgs);
    println!("Done loading..");

    println!("Reading filters..");
    let (re, abs) =
        file_io::read_filters_from_file_flattened(FILTER_PATH).expect("Could not read filters");
    let re = adjust_length(re, nr_filters);
    let abs = adjust_length(abs, nr_filters);
    println!("Done reading..");

    println!("Loaded {} images & {} filters", images.len(), re.len());
    let gpu = Extractor::new().unwrap();

    let method = &args[1];
    match method.as_str() {
        "cpu" => {
            println!("Computing CPU");
            research::run_research_cpu(&images, &abs, &re, 500);
        }
        "gpu" => {
            println!("Computing GPU");
            let max_chunk = 500;
            let ilen = images[0].len();
            let images = flatten_content(images);
            let re = flatten_content(re);
            let abs = flatten_content(abs);
            let shader = get_cosine_similarity_shader(ilen, (256, 1, 1)).to_string();
            //println!("{}", shader);
            let shader = include_str!("../wgpu_test/src/shaders/dot_summerize.wgsl").to_string();
            let max_shader = get_for_loop_max_pool_shader(ilen as u64, (16, 16, 1)).to_string();
            research::run_research_gpu(
                &method,
                &images,
                &re,
                &abs,
                &shader,
                &max_shader,
                max_chunk,
                ilen,
                &gpu,
            );
        }
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
                    (
                        get_parallel_cosine_similarity_shader(
                            nr_imgs, nr_filters, ilen, chunk, wg_size,
                        )
                        .to_string(),
                        get_parallel_max_pool_shader(max_chunk, chunk, wg_size).to_string(),
                    )
                }
                "gpu-loop" => {
                    println!("Computing GPU with loop shader");
                    let wg_size = (16, 16, 1);
                    (
                        get_for_loop_cosine_similarity_shader(ilen, wg_size).to_string(),
                        get_for_loop_max_pool_shader(max_chunk, wg_size).to_string(),
                    )
                }
                _ => panic!("Invalid"),
            };
            research::run_research_gpu_all_images(
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
        }
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
