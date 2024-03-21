use wgpu_test::Extractor;

fn main() {
    // TODO: initiate logger instead of print
    //print_devices();
    test_simple_feature_extraction();
}

fn test_simple_feature_extraction() {
    // Data for computation
    println!("Initializing data..");
    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 36];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 65_535];

    let filter_chunk = 2;
    let chunk = 5;
    // Result buffer
    let mut res: Vec<Vec<f32>> = vec![vec![0.; b.len() /*  filter_chunk*/]; a.len()];

    println!("\nComputing..");
    let flat_output = Extractor::new()
        .unwrap()
        .get_features(&a, &b, chunk, filter_chunk)
        .unwrap();
    //println!("Flat output: {:?}", flat_output);
    /*let mut correct = 0;
    let mut t = false;
    for sum_prod in flat_output.chunks(421).into_iter(){
        let mut sum = 0.;
        for i in sum_prod.iter(){
            sum += i;
        }
        if sum != 841. {
            println!("Sum: {}", sum);
            println!("Sum prod: {:?}", sum_prod);
            if t {
                break;
            }
            t = true;
            continue;
        } else {
            correct += 1;
        }
    }
    println!("Correct / total\n {} / {}", correct, flat_output.chunks(421).len());*/

    let mut correct = 0;
    let mut counter = 0;
    let mut dis = 0;
    for (j, sum_prod) in flat_output.chunks(65_535).into_iter().enumerate() {
        if counter == 3 {
            counter = 0;
            dis += 1;
        }
        if sum_prod.iter().all(|i| i == &841.) {
            correct += 1;
        } else {
            /*for (i, r) in sum_prod.iter().enumerate(){
                if r != &841. && r != &0.{
                    println!("\n{}, {}, {}", sum_prod[i], sum_prod[i], sum_prod[i+1]);
                    let l = 65_535;
                    println!("{}, {}, {}", i, i, i+1);
                    println!("dispatch {}", dis);
                    println!("image {}", j);
                    println!("J: {}, {}, {}\n", i + j*l, i + j*l, i+1 + j*l);
                }
            }*/
        }
        counter += 1;
    }

    println!(
        "Correct / total\n {} / {}",
        correct,
        flat_output.chunks(65_535).len()
    );
    //write_wgpu_res_to_file(&flat_output).unwrap();

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
    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 3];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 3];
    let filter_chunk = 2;
    let chunk = 5;
    let ex = Extractor::new();
    match ex {
        Ok(e) => match e.get_features(&a, &b, chunk, filter_chunk) {
            Ok(res) => {
                assert!(res.into_iter().eq([841.0; 3 * 3].iter().cloned()));
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
