use wgpu_test::Extractor;

fn main() {
    // TODO: initiate logger instead of print
    print_devices();
    test_simple_feature_extraction();
}

fn test_simple_feature_extraction() {
    // Data for computation
    println!("Initializing data..");
    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 6000];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 65_535];
    // Result buffer
    let mut res: Vec<Vec<f32>> = vec![vec![0.; b.len()]; a.len()];

    println!("\nComputing..");
    let flat_output = Extractor::new()
        .unwrap()
        //.get_features(&a, &b, chunk, filter_chunk)
        .dot(&a, &b)
        .unwrap();
    println!("Output: {:?}", flat_output[0]);
    println!("Output: {:?}", flat_output[100]);
    //println!("Flat output: {:?}", flat_output);
    //write_wgpu_res_to_file(&flat_output).unwrap();

    println!("Finalizing..");
    let mut it = flat_output.into_iter();
    let _ = res
        .iter_mut()
        .map(|inner| inner.iter_mut().for_each(|r| *r = it.next().unwrap()))
        .collect::<Vec<_>>();

    println!(
        "Total number of elemnts: {}",
        res.iter().flatten().collect::<Vec<&f32>>().len()
    );
    println!(
        "Number of elements wrong: {}",
        res.iter().flatten().filter(|i| i != &&841.).count()
    );
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
    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 6000];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 65_535];
    let ex = Extractor::new();
    match ex {
        Ok(e) => match e.dot(&a, &b) {
            Ok(res) => {
                println!("Result: {:?}", res);
                assert!(res.into_iter().eq([841.0; 6000 * 65_535].iter().cloned()));
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
