use std::fs::File;
use std::io::Write;
use wgpu_test::{extractor, Extractor};
use wgsl::*;

const VARIANS_COMPUTING: usize = 30;
const FILE_PATH: &str = "src/files/";

pub fn run_research(
    name: &str,
    images: &Vec<f32>,
    re: &Vec<f32>,
    abs: &Vec<f32>,
    cosine_shader: &str,
    max_shader: &str,
    ilen: usize,
    max_chunk: u64,
    ex: &Extractor,
) {
    let img_len = images.len() / ilen;
    let file_name = format!("{}_img_{}.csv", img_len, name);
    let mut file =
        File::create(format!("{}{}", FILE_PATH, file_name)).expect("Failed to create file");

    let max_fi_len = re.len() / ilen;
    let mut fi_len = 500;
    while fi_len <= max_fi_len {
        let real = re[..fi_len * ilen].to_vec();
        let absolute = abs[..fi_len * ilen].to_vec();

        let (cos_dis_x, cos_dis_y, max_dis_x) = get_dispatches(img_len as u32, fi_len, max_chunk);
        let cosine_dis = (cos_dis_x, cos_dis_y, 1);
        let max_dis = (max_dis_x, 1, 1);

        let comp = Computing {
            nr_of_filters: fi_len,
            elapsed: run_varians_computing(
                images,
                &real,
                &absolute,
                cosine_dis,
                max_dis,
                &cosine_shader,
                &max_shader,
                fi_len,
                max_chunk,
                ex,
            ),
        };
        comp.save(&mut file);
        fi_len += 500;
    }
    file.flush().expect("Failed to flush file");
}

fn get_dispatches(imgs: u32, filters: usize, max_chunk: u64) -> (u32, u32, u32) {
    let mut max_dis_x = imgs * filters as u32 / max_chunk as u32;
    if max_dis_x > 65_535 {
        max_dis_x = 65_535;
    }
    let mut imgs = imgs;
    if imgs > 65_535 {
        imgs = 65_535;
    }
    let mut filters = filters as u32;
    if filters > 65_535 {
        filters = 65_535;
    }
    (imgs, filters as u32, max_dis_x)
}

struct Computing {
    nr_of_filters: usize,
    elapsed: Vec<Elapsed>,
}

impl Computing {
    fn save(&self, file: &mut File) {
        let mut sum = 0;
        for el in &self.elapsed {
            if el == self.elapsed.first().unwrap() {
                writeln!(file, "{}, {}, {}, ", self.nr_of_filters, el.id, el.time)
                    .expect("Failed to write to file");
            } else {
                writeln!(file, ", {}, {},", el.id, el.time).expect("Failed to write to file");
            }
            sum += el.time;
        }
        let avg = sum / self.elapsed.len() as u128;
        writeln!(file, ", , , {}", avg).expect("Failed to write to file");
    }
}

#[derive(Eq, PartialEq)]
struct Elapsed {
    id: usize,
    time: u128,
}

fn run_varians_computing(
    image: &Vec<f32>,
    re: &Vec<f32>,
    abs: &Vec<f32>,
    cosine_dis: (u32, u32, u32),
    max_dis: (u32, u32, u32),
    cosine_shader: &str,
    max_shader: &str,
    out_len: usize,
    max_chunk: u64,
    ex: &Extractor,
) -> Vec<Elapsed> {
    let mut comps = Vec::new();
    for i in 0..VARIANS_COMPUTING {
        let start = std::time::Instant::now();
        match ex.compute_cosine_simularity_max_pool(
            image,
            re,
            abs,
            cosine_dis,
            max_dis,
            cosine_shader,
            max_shader,
            out_len,
            max_chunk,
        ) {
            Ok(res) => {
                let time = start.elapsed().as_millis();
                extractor::test_res(res, 29.);
                comps.push(Elapsed { id: i, time })
            }
            Err(e) => {
                println!("Error: {:?}\n continuing..", e);
                break;
            }
        }
    }
    comps
}
