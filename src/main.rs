use cpu_test::{utils::*, *};
use parallelization::research;
use wgpu_test::*;
use wgsl::*;

const MNIST_PATH: &str = "cpu_test/data/";
const FILTER_PATH: &str = "src/files/filters/";

fn main() {
    println!("Initializing data..");

    println!("Loading MNIST..");
    let images = data::mnist_data::load_mnist_dataset_flattened(1_000, MNIST_PATH);
    println!("Done loading..");

    println!("Reading filters..");
    let (re, abs) =
        file_io::read_filters_from_file_flattened(FILTER_PATH).expect("Could not read filters");
    println!("Done reading..");
    println!("len {}  {} ", re.len(), abs.len());
    println!("imlen: {}", images.len());

    println!("Computing CPU");
    research::run_research_cpu(&images, &abs, &re, 500);
    println!("Done..");

    let max_chunk = 500;
    let ilen = images[0].len();
    let im_len = images.len();
    let fi_len = re.len();

    let images = flatten_content(images);
    let re = flatten_content(re);
    let abs = flatten_content(abs);

    let gpu = Extractor::new().unwrap();
    println!("Computing GPU..");
    let cosine_shader =
        get_parallel_cosine_similarity_shader(im_len, fi_len, ilen, 10, (253, 1, 1)).to_string();
    let cosine_shader =
        get_for_loop_cosine_similarity_shader(im_len, fi_len, ilen, (16, 16, 1)).to_string();
    println!("HERE IS THE COSINE SHADER: \n{}", cosine_shader);

    let max_shader = get_parallel_max_pool_shader(max_chunk, 10, (250, 1, 1)).to_string();
    let max_shader = get_for_loop_max_pool_shader(max_chunk, (256, 1, 1)).to_string();
    println!("\n\nHERE IS THE MAX SHADER: \n{}", max_shader);

    /*let start = std::time::Instant::now();
    let res: Vec<f32> = gpu
        .compute_cosine_simularity_max_pool(
            &images,
            &re,
            &abs,
            (im_len as u32, 65_535 as u32, 1),
            (65_535, 1, 1),
            &cosine_shader,
            &max_shader,
            im_len * fi_len,
            max_chunk,
        )
        .unwrap();
    println!("Elapsed time shader computation: {:?}", start.elapsed());
    //println!("res: {:?}", res);
    println!("res.len(): {:?}", res.len());
    extractor::test_res(res, 29.);*/

    research::run_research_gpu(
        "parallel_shader",
        &images,
        &re,
        &abs,
        &cosine_shader,
        &max_shader,
        ilen,
        max_chunk,
        &gpu,
    );
    println!("Done!");
}

fn flatten_content<T>(content: Vec<Vec<T>>) -> Vec<T> {
    content.into_iter().flatten().collect()
}
