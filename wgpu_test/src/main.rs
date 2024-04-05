use rayon::prelude::*;
use std::time::Instant;
use wgpu_test::Extractor;

fn main() {
    // TODO: initiate logger instead of print
    print_devices();
    test_simple_dot_for_loop_shader();
    test_simple_dot_parallel_shader();
}

fn test_simple_dot_parallel_shader() {
    // Data for computation
    println!("\nInitializing data for temp buffer dot shader..");
    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 6_000];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 65_535];
    // Result buffer
    let mut res: Vec<Vec<f32>> = vec![vec![0.; b.len()]; a.len()];

    println!("Computing temp buffer dot shader..");
    let ex = Extractor::new().unwrap();
    let now = Instant::now();
    let flat_output = ex
        .dot(&a, &b, (2_200, 65_535, 1), "parallel_dot.wgsl")
        .unwrap();
    let elapsed = now.elapsed();
    //println!("Flat output: {:?}", flat_output);
    println!("Elapsed computing time: {:.2?}", elapsed);

    println!("Finalizing temp buffer dot shader..");
    let chunk_sizes = res.iter().map(|inner| inner.len()).collect::<Vec<_>>();
    let mut starts = vec![0];
    let mut sum = 0;
    for size in &chunk_sizes {
        sum += *size;
        starts.push(sum);
    }

    res.par_iter_mut().enumerate().for_each(|(i, inner)| {
        let start = starts[i];
        let end = starts[i + 1];
        let slice = &flat_output[start..end];
        inner
            .iter_mut()
            .enumerate()
            .for_each(|(j, r)| *r = slice[j]);
    });
    let total_elements = res.par_iter().flatten().count();
    let wrong_elements = res.par_iter().flatten().filter(|i| **i != 841.).count();

    let percentage_wrong = (wrong_elements as f64 / total_elements as f64) * 100.0;

    println!("Total number of elements: {}", total_elements);
    println!("Number of elements wrong: {}", wrong_elements);

    println!("Percentage of elements wrong: {:.2}%", percentage_wrong);
}

fn test_simple_dot_for_loop_shader() {
    // Data for computation
    println!("\nInitializing data for for_loop shader..");
    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 3792];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 100_000];
    // Result buffer
    let mut res: Vec<Vec<f32>> = vec![vec![0.; b.len()]; a.len()];

    println!("Computing for_loop shader..");
    let ex = Extractor::new().unwrap();
    let now = Instant::now();
    let flat_output = ex.dot(&a, &b, (238, 6_250, 1), "for_loop.wgsl").unwrap();
    let elapsed = now.elapsed();
    //println!("Flat output: {:?}", flat_output);
    println!("Elapsed computing time: {:.2?}", elapsed);

    println!("Finalizing for_loop shader..");
    let chunk_sizes = res.iter().map(|inner| inner.len()).collect::<Vec<_>>();
    let mut starts = vec![0];
    let mut sum = 0;
    for size in &chunk_sizes {
        sum += *size;
        starts.push(sum);
    }

    res.par_iter_mut().enumerate().for_each(|(i, inner)| {
        let start = starts[i];
        let end = starts[i + 1];
        let slice = &flat_output[start..end];
        inner
            .iter_mut()
            .enumerate()
            .for_each(|(j, r)| *r = slice[j]);
    });
    let total_elements = res.par_iter().flatten().count();
    let wrong_elements = res.par_iter().flatten().filter(|i| **i != 841.).count();

    let percentage_wrong = (wrong_elements as f64 / total_elements as f64) * 100.0;

    println!("Total number of elements: {}", total_elements);
    println!("Number of elements wrong: {}", wrong_elements);

    println!("Percentage of elements wrong: {:.2}%", percentage_wrong);
}

fn print_devices() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(print_devices_async());
}

async fn print_devices_async() {
    let instance = wgpu::Instance::default();
    println!("Available backends:");
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        println!("{:?}", a.get_info());
        println!(
            "{:?}",
            a.request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None
            )
            .await
            .unwrap()
            .0
            .global_id()
        );
    }
}

#[test]
fn test_feature_extraction() {
    use wgpu_test::WgpuContextError;
    // Data for computation
    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 14];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 4];
    let ex = Extractor::new();
    match ex {
        Ok(e) => match e.dot(&a, &b, (5, 4, 1), "parallel_dot.wgsl") {
            Ok(res) => {
                assert!(res.into_iter().eq([841.0; 4 * 14].iter().cloned()));
            }
            _ => assert!(false),
        },
        Err(e) => match e {
            // No adapter error is expected, for pipeline testing
            WgpuContextError::NoAdapterError => assert!(true),
            _ => assert!(false),
        },
    }
}
