use std::fs::File;
use std::io::Write;
use wgpu_test::Extractor;

const VARIANS_COMPUTING: usize = 60;
const MAX_DISPATCH: u32 = 65_535;
const FILE_PATH: &str = "src/files/results/";

#[derive(Debug, PartialEq, Eq)]
pub enum GPUShader {
    OneImgAllFilters,
    AllImgsAllFilters,
    AllImgsAllFiltersParallel,
}

pub fn run_research_gpu(
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

    let mut file_cos = File::create(format!(
        "{}{}",
        FILE_PATH,
        format!("cosine_time_{}.csv", uniqe)
    ))
    .expect("Failed to create file");
    writeln!(file_cos, "Filter, ID, Time_us, Average_time").expect("Failed to write to file");
    let mut file_max = File::create(format!("{}{}", FILE_PATH, format!("max_time{}.csv", uniqe)))
        .expect("Failed to create file");
    writeln!(file_max, "Filter, ID, Time_us, Average_time").expect("Failed to write to file");
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
            ),
        };
        if comp.elapsed.is_empty() {
            break;
        }
        comp.save(&mut file_cos, &mut file_max);
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
) -> Vec<(Elapsed, Elapsed)> {
    let (images, re, abs) = data;
    let (cosine_shader, cosine_dis) = cosine;
    let (max_shader, max_dis) = max;
    let (out_len, _) = lens;
    let mut comps = Vec::new();
    for i in 1..=VARIANS_COMPUTING {
        let res = ex.compute_cosine_simularity_max_pool_all_images(
            images,
            re,
            abs,
            (cosine_shader, cosine_dis),
            (max_shader, max_dis),
            (out_len, max_chunk),
        );
        match res {
            Ok(time) => {
                let (cos_time, max_time) = time;
                comps.push((
                    Elapsed {
                        id: i,
                        time: cos_time,
                    },
                    Elapsed {
                        id: i,
                        time: max_time,
                    },
                ))
            }
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
    elapsed: Vec<(Elapsed, Elapsed)>,
}

impl Computing {
    fn save(&self, file_cos: &mut File, file_max: &mut File) {
        if self.elapsed.is_empty() {
            return;
        }
        let mut cos_sum = 0;
        let mut max_sum = 0;
        for (i, el) in self.elapsed.iter().enumerate() {
            let (el_cos, el_max) = el;
            if i == 0 {
                writeln!(
                    file_cos,
                    "{}, {}, {}, 0",
                    self.nr_of_filters, el_cos.id, el_cos.time
                )
                .expect("Failed to write to file");
                writeln!(
                    file_max,
                    "{}, {}, {}, 0",
                    self.nr_of_filters, el_max.id, el_max.time
                )
                .expect("Failed to write to file");
            } else {
                writeln!(file_cos, "0, {}, {}, 0", el_cos.id, el_cos.time)
                    .expect("Failed to write to file");
                writeln!(file_max, "0, {}, {}, 0", el_max.id, el_max.time)
                    .expect("Failed to write to file");
            }
            cos_sum += el_cos.time;
            max_sum += el_max.time;
        }
        let mut avg_cos = cos_sum / self.elapsed.len() as u128;
        let mut avg_max = max_sum / self.elapsed.len() as u128;
        if avg_cos == 0 {
            avg_cos = 1;
        }
        if avg_max == 0 {
            avg_max = 1;
        }
        println!(
            "Run saved: Filter: {}, Cosine avg: {}, Max avg: {}",
            self.nr_of_filters, avg_cos, avg_max
        );
        writeln!(file_cos, "0, 0, 0, {}", avg_cos).expect("Failed to write to file");
        writeln!(file_max, "0, 0, 0, {}", avg_max).expect("Failed to write to file");
        file_cos.flush().expect("Failed to flush file");
        file_max.flush().expect("Failed to flush file");
    }
}

#[derive(Eq, PartialEq)]
struct Elapsed {
    id: usize,
    time: u128,
}
