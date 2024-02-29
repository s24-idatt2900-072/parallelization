use wgpu_test::Extractor;

fn main() {
    // TODO: initiate logger instead of print
    //print_devices();
    test_simple_feature_extraction();
}

fn test_simple_feature_extraction() {
    // Data for computation
    println!("Initializing data..");
    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 3];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 65_536];

    let filter_chunk = 2;
    let chunk = 5;
    // Result buffer
    let mut res: Vec<Vec<f32>> = vec![vec![0.; b.len() /*  filter_chunk*/]; a.len()];

    println!("\nComputing..");
    let flat_output = Extractor::new()
        .unwrap()
        .get_features(&a, &b, chunk, filter_chunk)
        .unwrap();
    let mut it = flat_output.into_iter();
    let _ = res
        .iter_mut()
        .map(|inner| inner.iter_mut().for_each(|r| *r = it.next().unwrap()))
        .collect::<Vec<_>>();

    println!("{}", res.iter().flatten().collect::<Vec<&f32>>().len());
    println!("{}", res.iter().flatten().filter(|i| i != &&841.).count());
    //println!("\nResult: {:?}", res);
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
    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 3];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 3];
    let filter_chunk = 2;
    let chunk = 5;
    let ex = Extractor::new();
    match ex {
        Ok(e) => {
            let ok = e.get_features(&a, &b, chunk, filter_chunk);
            match ok {
                Ok(res) => {
                    assert!(res.into_iter().eq([841.0; 3 * 3].iter().cloned()));
                }
                Err(_) => assert!(false),
            }
        }
        Err(e) => match e {
            // No adapter error is expected, for pipeline testing
            WgpuContextError::NoAdapterError => assert!(true),
            _ => assert!(false),
        },
    }
}
