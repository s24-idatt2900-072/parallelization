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
        println!("  cpu - Runs CPU method");
        println!("  gpu - Runs GPU method, one image with all filters");
        println!("  gpu-par - Runs GPU method with parallel shader, all images with all filters");
        println!("  gpu-loop - Runs GPU method with loop shader, all images with all filters");
        println!("  \"nr images\" - Argument 2, number of images to use");
        println!("  \"nr filters\" - Argument 3, number of filters to use");
        println!("  \"filter increment\" - Argument 4, increment of filters to use");
        println!("  \"max poll chunk\" - Argument 5, maximum chunk size for max pooling");
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
        "cpu" => {
            println!("Computing CPU");
            research::run_research_cpu(&images, &abs, &re, max_pool_chunk, filter_inc);
        }
        "gpu" | "gpu-par" | "gpu-loop" => {
            let ilen = images[0].len();

            let images = flatten_content(images);
            let re = flatten_content(re);
            let abs = flatten_content(abs);
            let (cosine_shader, max_shader, all_images) = match method.as_str() {
                "gpu" => (
                    include_str!("../wgpu_test/src/shaders/dot_summerize.wgsl").to_string(),
                    get_for_loop_max_pool_shader(ilen as u64, (16, 16, 1)).to_string(),
                    false,
                ),
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
                        true,
                    )
                }
                "gpu-loop" => {
                    println!("Computing GPU with loop shader");
                    let wg_size = (16, 16, 1);
                    (
                        get_for_loop_cosine_similarity_shader(ilen, wg_size).to_string(),
                        get_for_loop_max_pool_shader(max_pool_chunk as u64, wg_size).to_string(),
                        true,
                    )
                }
                _ => panic!("Invalid"),
            };
            research::run_research_gpu(
                method,
                (&images, &re, &abs),
                (&cosine_shader, &max_shader),
                ilen,
                max_pool_chunk as u64,
                &gpu,
                (filter_inc, all_images),
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

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    const IMG_LEN: usize = 10;
    const FILTER_LEN: usize = 500;
    const EXPECTED: [[f32; 1]; IMG_LEN] = [
        [0.9958068],
        [0.023218844],
        [0.99713045],
        [0.9985984],
        [0.9986014],
        [0.99841976],
        [0.9984336],
        [0.99874896],
        [0.9983187],
        [0.3554934],
    ];

    #[test]
    fn test_cpu_computing() {
        let images = data::mnist_data::load_mnist_dataset_flattened(IMG_LEN as u32, MNIST_PATH);
        let (re, abs) = file_io::read_filters_from_file_flattened(FILTER_PATH).unwrap();
        let re = adjust_length(re, FILTER_LEN);
        let abs = adjust_length(abs, FILTER_LEN);
        let res = research::compute_cpu(&images, &re, &abs, FILTER_LEN);
        assert_eq!(res, EXPECTED);
    }

    #[test]
    fn test_gpu_computing() {
        let ex = Extractor::new();
        match ex {
            Ok(_) => {}
            Err(wgpu_test::WgpuContextError::NoAdapterError) => {
                assert!(true);
                return;
            }
            _ => assert!(false),
        }
        let ex = ex.unwrap();
        let images: Vec<Vec<f32>> =
            data::mnist_data::load_mnist_dataset_flattened(IMG_LEN as u32, MNIST_PATH);
        let (re, abs) = file_io::read_filters_from_file_flattened(FILTER_PATH).unwrap();
        let re: Vec<Vec<f32>> = adjust_length(re, FILTER_LEN);
        let abs: Vec<Vec<f32>> = adjust_length(abs, FILTER_LEN);

        let cosine_dis = (re.len() as u32, 1, 1);
        let max_dis = (((images.len() * re.len()) / FILTER_LEN) as u32, 1, 1);
        let ilen = images[0].len();

        let images = flatten_content(images);
        let re = flatten_content(re);
        let abs = flatten_content(abs);

        let cosine_shader = include_str!("../wgpu_test/src/shaders/dot_summerize.wgsl").to_string();
        let max_shader = get_for_loop_max_pool_shader(ilen as u64, (16, 16, 1)).to_string();
        let res: Vec<f32> = ex
            .cosine_simularity_max_one_img_all_filters(
                &images,
                (&re, &abs),
                (&cosine_shader, cosine_dis),
                (&max_shader, max_dis),
                FILTER_LEN as u64,
                ilen,
            )
            .unwrap();
        let expected = EXPECTED.iter().flatten().cloned().collect::<Vec<f32>>();
        assert!(res
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a * 1_00.0).round() == (b * 1_00.0).round()));
    }

    #[test]
    fn test_gpu_computing_parallel_shader() {
        let ex = Extractor::new();
        match ex {
            Ok(_) => {}
            Err(wgpu_test::WgpuContextError::NoAdapterError) => {
                assert!(true);
                return;
            }
            _ => assert!(false),
        }
        let ex = ex.unwrap();
        let images: Vec<Vec<f32>> =
            data::mnist_data::load_mnist_dataset_flattened(IMG_LEN as u32, MNIST_PATH);
        let (re, abs) = file_io::read_filters_from_file_flattened(FILTER_PATH).unwrap();
        let re: Vec<Vec<f32>> = adjust_length(re, FILTER_LEN);
        let abs: Vec<Vec<f32>> = adjust_length(abs, FILTER_LEN);

        let cosine_dis = (images.len() as u32, re.len() as u32, 1);
        let max_dis = (((images.len() * re.len()) / FILTER_LEN) as u32, 1, 1);
        let ilen = images[0].len();
        let fi_len = re.len();
        let im_len = images.len();

        let images = flatten_content(images);
        let re = flatten_content(re);
        let abs = flatten_content(abs);

        let cosine_shader =
            get_parallel_cosine_similarity_shader(im_len, fi_len, ilen, 10, (253, 1, 1))
                .to_string();
        let max_shader =
            get_parallel_max_pool_shader(FILTER_LEN as u64, 10, (249, 1, 1)).to_string();
        let res: Vec<f32> = ex
            .compute_cosine_simularity_max_pool_all_images(
                &images,
                &re,
                &abs,
                (&cosine_shader, cosine_dis),
                (&max_shader, max_dis),
                (im_len * fi_len, FILTER_LEN as u64),
            )
            .unwrap();
        let expected = EXPECTED.iter().flatten().cloned().collect::<Vec<f32>>();
        assert!(res
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a * 100_000.0).round() == (b * 100_000.0).round()));
    }

    #[test]
    fn test_gpu_computing_lopp_shader() {
        let ex = Extractor::new();
        match ex {
            Ok(_) => {}
            Err(wgpu_test::WgpuContextError::NoAdapterError) => {
                assert!(true);
                return;
            }
            _ => assert!(false),
        }
        let ex = ex.unwrap();
        let images: Vec<Vec<f32>> =
            data::mnist_data::load_mnist_dataset_flattened(IMG_LEN as u32, MNIST_PATH);
        let (re, abs) = file_io::read_filters_from_file_flattened(FILTER_PATH).unwrap();
        let re: Vec<Vec<f32>> = adjust_length(re, FILTER_LEN);
        let abs: Vec<Vec<f32>> = adjust_length(abs, FILTER_LEN);

        let cosine_dis = (images.len() as u32, re.len() as u32, 1);
        let max_dis = (((images.len() * re.len()) / FILTER_LEN) as u32, 1, 1);
        let ilen = images[0].len();
        let fi_len = re.len();
        let im_len = images.len();

        let images = flatten_content(images);
        let re = flatten_content(re);
        let abs = flatten_content(abs);

        let wg_size = (16, 16, 1);
        let cosine_shader = get_for_loop_cosine_similarity_shader(ilen, wg_size).to_string();
        let max_shader = get_for_loop_max_pool_shader(FILTER_LEN as u64, wg_size).to_string();
        let res: Vec<f32> = ex
            .compute_cosine_simularity_max_pool_all_images(
                &images,
                &re,
                &abs,
                (&cosine_shader, cosine_dis),
                (&max_shader, max_dis),
                (im_len * fi_len, FILTER_LEN as u64),
            )
            .unwrap();
        let expected = EXPECTED.iter().flatten().cloned().collect::<Vec<f32>>();
        assert!(res
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a * 100_000.0).round() == (b * 100_000.0).round()));
    }

    #[test]
    fn test_adjust_length() {
        let list = vec![1, 2, 3, 4, 5];
        let res = adjust_length(list, 10);
        assert_eq!(res.len(), 10);
        assert_eq!(res[0], 1);
        assert_eq!(res[5], 1);
        assert_eq!(res[9], 5);
    }

    #[test]
    fn test_flatten_content() {
        let content = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let res = flatten_content(content);
        assert_eq!(res.len(), 6);
        assert_eq!(res[0], 1);
        assert_eq!(res[5], 6);
    }
}
