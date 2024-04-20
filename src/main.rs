use cpu_test::{utils::*, *};
use wgpu_test::*;
use wgsl::*;

fn main() {
    let gpu = Extractor::new().unwrap();
    println!("Initializing data..");
    let images = vec![vec![1.; 841]; 10];
    let re = vec![vec![1.; 841]; 500];
    let abs = vec![vec![1.; 841]; 500];

    println!("Computing..");
    let shader = include_str!("../wgpu_test/src/shaders/parallel_cosine_similarity.wgsl");
    //let shader = include_str!("../wgpu_test/src/shaders/for_loop_cosine_similarity.wgsl");
    //let shader = include_str!("../wgpu_test/src/shaders/parallel_max_pool.wgsl");
    //let shader = include_str!("../wgpu_test/src/shaders/for_loop_max_pool.wgsl");
    let start = std::time::Instant::now();
    let res: Vec<f32> = gpu
        .compute_cosine_simularity(&images, &re, &abs, (10, 10, 1), shader)
        .unwrap();
    println!("Elapsed time shader computation: {:?}", start.elapsed());
    println!("res: {:?}", res);
    println!("res.len(): {:?}", res.len());
    //extractor::test_res(res, images.len(), re.len(), 29.);
}
