use wgsl::*;
use wgpu_test::*;
use cpu_test::{utils::*, *};

fn main() {
    let _con = WgpuContext::new();
    _con.unwrap().get_limits();

    println!("file path: {}", config::settings::FILTERS_PATH);
    let filters = file_io::read_filters_from_file("cpu_test/files/filter.csv").expect("Failed to read filters from file.");
    println!("Filters: {:?}", filters);

    let a = Var::from("a");
    let b = Var::from("b");
    let out = Var::from("out");

    let blen = Var::from("blen");
    let alen = Var::from("alen");
    let tidx = Var::from("tidx");

    let vars = vec![
        (alen.clone(), Var::from_num(14 as u32)),
        (blen.clone(), Var::from_num(4 as u32)),
        (
            tidx.clone(),
            Var::WorkgroupIdX
                .multiply(&Var::WorkSizeX)
                .add(&Var::LocalInvocationIdX),
        ),
    ];
    let obj = ReturnType::Obj(Object::Array(Type::F32, None));
    let binds = vec![(&a, false), (&b, false), (&out, true)];
    let shader = ComputeShader::new(binds, &obj, (16, 16, 1))
        .add_variables(vars)
        .add_line(Line::from(Instruction::Set {
            lhs: out.index(&tidx),
            rhs: alen.multiply(&blen),
        }))
        .finish();
    println!("{}", shader.to_string());
}
