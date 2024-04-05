use rayon::prelude::*;
use std::time::Instant;
use wgpu_test::Extractor;

fn main() {
    // TODO: initiate logger instead of print
    //print_devices();
    let now = Instant::now();
    test_simple_feature_extraction();
    let elapsed = now.elapsed();
    println!("Total elapsed time: {:.2?}", elapsed);
}

fn test_simple_feature_extraction() {
    // Data for computation
    println!("Initializing data..");
    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 3800];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 100_000];
    // Result buffer
    let mut res: Vec<Vec<f32>> = vec![vec![0.; b.len()]; a.len()];

    let now = Instant::now();
    println!("\nComputing..");
    let ex = Extractor::new().unwrap();
    let flat_output = ex
        //.get_features(&a, &b, chunk, filter_chunk)
        .dot(&a, &b, (238, 6_250, 1))
        .unwrap();
    //println!("Output: {:?}", flat_output);
    //println!("Output: {:?}", flat_output[100]);
    //println!("Flat output: {:?}", flat_output);
    //write_wgpu_res_to_file(&flat_output).unwrap();
    let elapsed = now.elapsed();
    println!("Elapsed computing time: {:.2?}", elapsed);
    
    println!("Finalizing..");
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
        inner.iter_mut().enumerate().for_each(|(j, r)| *r = slice[j]);
    });
    /*
    println!(
        "Total number of elements: {}",
        res.par_iter().flatten().count()
    );
    println!(
        "Number of elements wrong: {}",
        res.par_iter().flatten().filter(|i| **i != 841.).count()
    );
    */
    let total_elements = res.par_iter().flatten().count();
    let wrong_elements = res.par_iter().flatten().filter(|i| **i != 841.).count();

    let percentage_wrong = (wrong_elements as f64 / total_elements as f64) * 100.0;

    println!("Total number of elements: {}", total_elements);
    println!("Number of elements wrong: {}", wrong_elements);

    println!("Percentage of elements wrong: {:.2}%", percentage_wrong);

    //println!("Wrong elements:\n{:?}", res.iter().flatten().filter(|i| i != &&841.).collect::<Vec<&f32>>());
    //println!("Numbers wrong: {:?}", res.iter().flatten().filter(|i| i != &&841.).collect::<Vec<&f32>>());
    //println!("indexes of wrong: {:?}", res.iter().enumerate().map(|(i, inner)| inner.iter().enumerate().filter(|(j, r)| r != &&841.).map(|(j, r)| (i, j)).collect::<Vec<(usize, usize)>>()).collect::<Vec<Vec<(usize, usize)>>>());
    //println!("\nResult: {:?}", res);
    println!("res inner len {}", res[0].len());
}

use std::io::Write;

fn write_wgpu_res_to_file(out: &Vec<f32>) -> std::io::Result<()> {
    let path = std::path::Path::new("files/output_wgpu.csv");
    let mut file = std::fs::File::create(path)?;

    writeln!(file, "\n# WGPU")?;
    //for (_, result) in out.iter().enumerate() {
    writeln!(file, "{:?}", out)?;
    //}
    file.flush()?; // Ensure all data is written to disk

    Ok(())
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
        Ok(e) => match e.dot(&a, &b, (5, 4, 1)) {
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
