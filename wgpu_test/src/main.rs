use wgpu_test::Extractor;

fn main() {
    // TODO: initiate logger instead of print
    print_devices();

    // Data for computation
    let mut a: Vec<Vec<f32>> = vec![
        vec![1., 2., 1.],
        vec![1., 1., 1.],
        vec![1., 2., 1.],
        vec![1., 1., 1.],
    ];
    let b: Vec<Vec<f32>> = vec![
        vec![1., 2., 4.],
        vec![1., 1., 2.],
        vec![3., 4., 5.],
        vec![1., 2., 1.],
    ];
    let filter_chunk = 2;
    let chunk = 2;
    // Result buffer
    let mut res: Vec<Vec<f32>> = vec![vec![0.; b.len() / filter_chunk]; a.len()];
    // Pad a with zeros
    a.push(vec![0.; 3]);
    a.push(vec![0.; 3]);

    println!("Computing..");
    Extractor::feature_extraction(&a, &b, &mut res, chunk, filter_chunk);
    println!("Result: {:?}", res);
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
    // Data for computation
    let mut a: Vec<Vec<f32>> = vec![
        vec![1., 2., 1.],
        vec![1., 1., 1.],
        vec![1., 2., 1.],
        vec![1., 1., 1.],
    ];
    let b: Vec<Vec<f32>> = vec![
        vec![1., 2., 4.],
        vec![1., 1., 2.],
        vec![3., 4., 5.],
        vec![1., 2., 1.],
    ];
    let filter_chunk = 2;
    let chunk = 2;
    // Result buffer
    let mut res = vec![vec![0.; b.len() / filter_chunk]; a.len()];
    a.push(vec![0.; 3]);
    a.push(vec![0.; 3]);
    Extractor::feature_extraction(&a, &b, &mut res, chunk, filter_chunk);
    assert!(res
        .into_iter()
        .flatten()
        .eq([9.0, 16.0, 7.0, 12.0, 9.0, 16.0, 7.0, 12.0].iter().cloned()));
}
