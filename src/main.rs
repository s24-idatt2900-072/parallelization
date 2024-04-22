use cpu_test::{utils::*, *};
use wgpu_test::*;
use wgsl::*;

fn main() {
    let gpu = Extractor::new().unwrap();
    println!("Initializing data..");
    let images = vec![vec![1.; 841]; 10];
    let re = vec![vec![1.; 841]; 100];
    let abs = vec![vec![1.; 841]; 100];
    let max_chunk = 500;

    let ilen = images[0].len();
    let im_len = images.len();
    let fi_len = re.len();

    let images = flatten_content(images);
    let re = flatten_content(re);
    let abs = flatten_content(abs);

    println!("Computing..");
    let cosine_shader = include_str!("../wgpu_test/src/shaders/parallel_cosine_similarity.wgsl");
    let cosine_shader = include_str!("../wgpu_test/src/shaders/for_loop_cosine_similarity.wgsl");
    let cosine_shader =
        get_for_loop_cosine_similarity_shader(im_len, fi_len, ilen, (16, 16, 1)).to_string();
    let cosine_shader =
        get_parallel_cosine_similarity_shader(im_len, fi_len, ilen, 10, (253, 1, 1)).to_string();
    let cosine_shader = cosine_shader.as_str();
    println!("HERE IS THE COSINE SHADER: \n{}", cosine_shader);

    let max_shader = include_str!("../wgpu_test/src/shaders/parallel_max_pool.wgsl");
    let max_shader = include_str!("../wgpu_test/src/shaders/for_loop_max_pool.wgsl");
    let max_shader = get_for_loop_max_pool_shader(max_chunk, im_len, (256, 1, 1)).to_string();
    let max_shader = get_parallel_max_pool_shader(max_chunk, 10, (250, 1, 1)).to_string();
    let max_shader = max_shader.as_str();
    println!("\n\nHERE IS THE MAX SHADER: \n{}", max_shader);

    let start = std::time::Instant::now();
    let res: Vec<f32> = gpu
        .compute_cosine_simularity_max_pool(
            &images,
            &re,
            &abs,
            (im_len as u32, fi_len as u32, 1),
            cosine_shader,
            max_shader,
            im_len * fi_len,
            max_chunk,
        )
        .unwrap();
    println!("Elapsed time shader computation: {:?}", start.elapsed());
    //println!("res: {:?}", res);
    println!("res.len(): {:?}", res.len());
    extractor::test_res(res, 29.);
}

fn flatten_content<T>(content: Vec<Vec<T>>) -> Vec<T> {
    content.into_iter().flatten().collect()
}
