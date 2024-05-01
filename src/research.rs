use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::ops::Div;
use wgpu_test::Extractor;

const VARIANS_COMPUTING: usize = 30;
const MAX_DISPATCH: u32 = 65_535;
const FILE_PATH: &str = "src/files/results/";

#[derive(Debug, PartialEq, Eq)]
pub enum GPUShader {
    OneImgAllFilters,
    AllImgsAllFilters,
    AllImgsAllFiltersParallel,
}

pub fn run_research_cpu(
    images: &[Vec<f32>],
    abs: &[Vec<f32>],
    re: &[Vec<f32>],
    max_chunk: usize,
    filter_inc: usize,
    sequential: bool,
) {
    let uniqe = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let file_name = format!("CPU_img_{}_{}.csv", images.len(), uniqe);
    let mut file =
        File::create(format!("{}{}", FILE_PATH, file_name)).expect("Failed to create file");
    writeln!(file, "Filter, ID, Time_ms, Average_time").expect("Failed to write to file");
    let mut fi_len = filter_inc;
    let max = re.len();
    while fi_len <= max {
        let real = re[..fi_len].to_vec();
        let absolute = abs[..fi_len].to_vec();

        let comp = Computing {
            nr_of_filters: fi_len,
            elapsed: run_varians_computing_cpu(images, &real, &absolute, max_chunk, sequential),
        };
        comp.save(&mut file);
        fi_len += filter_inc;
    }
}

fn run_varians_computing_cpu(
    images: &[Vec<f32>],
    real: &[Vec<f32>],
    absolute: &[Vec<f32>],
    max_chunk: usize,
    sequential: bool,
) -> Vec<Elapsed> {
    let mut comps = Vec::new();
    for i in 1..=VARIANS_COMPUTING {
        let start = std::time::Instant::now();
        let _ = match sequential {
            true => compute_cpu_sequential(images, real, absolute, max_chunk),
            false => compute_cpu(images, real, absolute, max_chunk),
        };
        comps.push(Elapsed {
            id: i,
            time: start.elapsed().as_millis(),
        })
    }
    comps
}

pub fn compute_cpu_sequential(
    images: &[Vec<f32>],
    real: &[Vec<f32>],
    absolute: &[Vec<f32>],
    max_chunk: usize,
) -> Vec<Vec<f32>> {
    images
        .iter()
        // Cosine simularity calculations
        .map(|img| {
            real.iter()
                .zip(absolute)
                .map(|(re, abs)| {
                    let (dot, norm) = img.iter().zip(re.iter()).zip(abs.iter()).fold(
                        (0., 0.),
                        |(dot, norm), ((&i, &r), &a)| {
                            let d = i * a;
                            (dot + d * r, norm + d * d)
                        },
                    );
                    dot.div(norm.sqrt())
                })
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
        // Max pooling of values
        .iter()
        .map(|values| {
            values
                .chunks(max_chunk)
                .map(|chunk| {
                    *chunk
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(&0.)
                })
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
}

pub fn compute_cpu(
    images: &[Vec<f32>],
    real: &[Vec<f32>],
    absolute: &[Vec<f32>],
    max_chunk: usize,
) -> Vec<Vec<f32>> {
    images
        .par_iter()
        // Cosine simularity calculations
        .map(|img| {
            real.par_iter()
                .zip(absolute)
                .map(|(re, abs)| {
                    let (dot, norm) = img.iter().zip(re.iter()).zip(abs.iter()).fold(
                        (0., 0.),
                        |(dot, norm), ((&i, &r), &a)| {
                            let d = i * a;
                            (dot + d * r, norm + d * d)
                        },
                    );
                    dot.div(norm.sqrt())
                })
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
        // Max pooling of values
        .par_iter()
        .map(|values| {
            values
                .chunks(max_chunk)
                .map(|chunk| {
                    *chunk
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(&0.)
                })
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>()
}

pub fn run_research_gpu(
    name: &str,
    data: (&Vec<f32>, &[f32], &[f32]),
    shaders: (&str, &str),
    ilen: usize,
    max_chunk: u64,
    ex: &Extractor,
    config: (usize, &GPUShader),
) {
    let (images, re, abs) = data;
    let (cosine_shader, max_shader) = shaders;
    let (filter_inc, shader) = config;
    let uniqe = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let img_len = images.len() / ilen;
    let file_name = format!("GPU_{}_img_{}_{}.csv", img_len, name, uniqe);
    let mut file =
        File::create(format!("{}{}", FILE_PATH, file_name)).expect("Failed to create file");
    writeln!(file, "Filter, ID, Time_ms, Average_time").expect("Failed to write to file");
    let max_fi_len = re.len() / ilen;
    let mut fi_len = filter_inc;
    while fi_len <= max_fi_len {
        let real = re[..fi_len * ilen].to_vec();
        let absolute = abs[..fi_len * ilen].to_vec();

        let (cos_dis_x, cos_dis_y, max_dis_x) = get_dispatches(img_len as u32, fi_len, max_chunk);
        let cosine_dis = if shader == &GPUShader::AllImgsAllFilters {
            (cos_dis_x, cos_dis_y, 1)
        } else if shader == &GPUShader::AllImgsAllFiltersParallel {
            (fi_len as u32, img_len as u32, 1)
        } else {
            let x = if fi_len as u32 > MAX_DISPATCH {
                MAX_DISPATCH
            } else {
                fi_len as u32
            };
            (x, 1, 1)
        };
        let max_dis = (max_dis_x, 1, 1);

        let comp = Computing {
            nr_of_filters: fi_len,
            elapsed: run_varians_computing_gpu(
                (images, &real, &absolute),
                (cosine_shader, cosine_dis),
                (max_shader, max_dis),
                (fi_len * img_len, ilen),
                max_chunk,
                ex,
                shader,
            ),
        };
        if comp.elapsed.is_empty() {
            break;
        }
        comp.save(&mut file);
        fi_len += filter_inc;
    }
}

fn get_dispatches(imgs: u32, filters: usize, max_chunk: u64) -> (u32, u32, u32) {
    let mut max_dis_x = imgs * filters as u32 / max_chunk as u32;
    if max_dis_x > MAX_DISPATCH {
        max_dis_x = MAX_DISPATCH;
    }
    let mut imgs = imgs;
    if imgs > MAX_DISPATCH {
        imgs = MAX_DISPATCH;
    }
    let mut filters = filters as u32;
    if filters > MAX_DISPATCH {
        filters = MAX_DISPATCH;
    }
    (imgs, filters, max_dis_x)
}

fn run_varians_computing_gpu(
    data: (&Vec<f32>, &Vec<f32>, &Vec<f32>),
    cosine: (&str, (u32, u32, u32)),
    max: (&str, (u32, u32, u32)),
    lens: (usize, usize),
    max_chunk: u64,
    ex: &Extractor,
    shader: &GPUShader,
) -> Vec<Elapsed> {
    let (images, re, abs) = data;
    let (cosine_shader, cosine_dis) = cosine;
    let (max_shader, max_dis) = max;
    let (out_len, ilen) = lens;
    let mut comps = Vec::new();
    for i in 1..=VARIANS_COMPUTING {
        let start = std::time::Instant::now();
        let res = match shader {
            GPUShader::AllImgsAllFilters => ex.compute_cosine_simularity_max_pool_all_images(
                images,
                re,
                abs,
                (cosine_shader, cosine_dis),
                (max_shader, max_dis),
                (out_len, max_chunk),
            ),
            GPUShader::AllImgsAllFiltersParallel => ex.cosine_simularity_max_all_img_all_filters(
                images,
                (&re, &abs),
                (&cosine_shader, cosine_dis),
                (&max_shader, max_dis),
                max_chunk,
                ilen,
            ),
            GPUShader::OneImgAllFilters => ex.cosine_simularity_max_one_img_all_filters(
                images,
                (re, abs),
                (cosine_shader, cosine_dis),
                (max_shader, max_dis),
                max_chunk,
                ilen,
            ),
        };
        let time = start.elapsed().as_millis();
        match res {
            Ok(_) => comps.push(Elapsed { id: i, time }),
            Err(e) => {
                println!("Error: {:?}\n Exiting..", e);
                return comps;
            }
        }
    }
    comps
}
struct Computing {
    nr_of_filters: usize,
    elapsed: Vec<Elapsed>,
}

impl Computing {
    fn save(&self, file: &mut File) {
        if self.elapsed.is_empty() {
            return;
        }
        let mut sum = 0;
        for el in &self.elapsed {
            if el == self.elapsed.first().unwrap() {
                writeln!(file, "{}, {}, {}, 0", self.nr_of_filters, el.id, el.time)
                    .expect("Failed to write to file");
            } else {
                writeln!(file, "0, {}, {}, 0", el.id, el.time).expect("Failed to write to file");
            }
            sum += el.time;
        }
        let mut avg = sum / self.elapsed.len() as u128;
        if avg == 0 {
            avg = 1;
        }
        println!(
            "Run saved, filters: {}, avg: {} ms",
            self.nr_of_filters, avg
        );
        writeln!(file, "0, 0, 0, {}", avg).expect("Failed to write to file");
        file.flush().expect("Failed to flush file");
    }
}

#[derive(Eq, PartialEq)]
struct Elapsed {
    id: usize,
    time: u128,
}
