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
    let mut filter_inc = 500;
    let mut max_pool_chunk = 500;
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run <method>");
        println!("Available methods:");
        println!("  gpu-par - Runs GPU method with parallel shader, all images with all filters");
        println!("  gpu-loop - Runs GPU method with loop shader, all images with all filters");
        println!("  \"nr images\" - Argument 2, number of images to use");
        println!("  \"max nr filters\" - Argument 3, max number of filters to use");
        println!(
            "  \"filter increment and start\" - Argument 4, increment of filters to use and start"
        );
        println!("  \"max poll chunk\" - Argument 5, chunk size for max pooling");
        println!(
            "  Default values: 1000 images, 1000 filters, 500 filter increment, 500 max pool chunk"
        );
        return;
    }
    if args.len() == 2 {
        println!("Using default values for number of filters and images");
    } else if args.len() == 3 {
        nr_imgs = args[2].parse().unwrap();
    } else if args.len() == 4 {
        nr_imgs = args[2].parse().unwrap();
        nr_filters = args[3].parse().unwrap();
    } else if args.len() == 5 {
        nr_imgs = args[2].parse().unwrap();
        nr_filters = args[3].parse().unwrap();
        filter_inc = args[4].parse().unwrap();
    } else if args.len() == 6 {
        nr_imgs = args[2].parse().unwrap();
        nr_filters = args[3].parse().unwrap();
        filter_inc = args[4].parse().unwrap();
        max_pool_chunk = args[5].parse().unwrap();
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
    println!(
        "filter_inc: {}, max_pool_chunk: {}",
        filter_inc, max_pool_chunk
    );
    let gpu = Extractor::new().unwrap();

    let method = &args[1];
    match method.as_str() {
        "gpu-par" | "gpu-loop" => {
            let ilen = images[0].len();

            let images = flatten_content(images);
            let re = flatten_content(re);
            let abs = flatten_content(abs);
            let (cosine_shader, max_shader, shader) = match method.as_str() {
                "gpu-par" => {
                    println!("Computing GPU with parallel shader");
                    let wg_size_cos = (253, 1, 1);
                    let wg_size_max = (249, 1, 1);
                    let chunk = 10;
                    (
                        get_parallel_cosine_similarity_shader(
                            nr_imgs,
                            nr_filters,
                            ilen,
                            chunk,
                            wg_size_cos,
                        )
                        .to_string(),
                        get_parallel_max_pool_shader(max_pool_chunk as u64, chunk, wg_size_max)
                            .to_string(),
                        research::GPUShader::AllImgsAllFilters,
                    )
                }
                "gpu-loop" => {
                    println!("Computing GPU with loop shader");
                    let wg_size = (16, 16, 1);
                    (
                        get_for_loop_cosine_similarity_shader(ilen, wg_size).to_string(),
                        get_for_loop_max_pool_shader(max_pool_chunk as u64, (256, 1, 1))
                            .to_string(),
                        research::GPUShader::AllImgsAllFilters,
                    )
                }
                _ => panic!("Invalid"),
            };
            research::run_research_gpu(
                (&images, &re, &abs),
                (&cosine_shader, &max_shader),
                ilen,
                max_pool_chunk as u64,
                &gpu,
                (filter_inc, &shader),
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
