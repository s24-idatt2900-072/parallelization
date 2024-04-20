use wgpu_test::wgpu_context::WgpuContext;
use wgpu_test::WgpuContextError;
use wgsl::*;

fn main() {
    let a: Vec<Vec<f32>> = vec![vec![1.; 101 * 101]; 4];
    let b: Vec<Vec<f32>> = vec![vec![1.; 101 * 101]; 14];
    let ilen = a[0].len();
    let shader = get_par_shader(a.len(), b.len(), ilen, 40_u32, (256, 1, 1));
    //let shader = get_for_shader(a.len(), b.len(), ilen, (32, 32, 1));
    let shader = format!("{}", shader);
    println!("{}", shader);
    let res = dot(&a, &b, (2_000, 4_000, 1), shader).unwrap();
    println!("res: {:?}", res);
}

fn dot(
    a: &Vec<Vec<f32>>,
    b: &Vec<Vec<f32>>,
    dis: (u32, u32, u32),
    shader: String,
) -> Result<Vec<f32>, WgpuContextError> {
    let size = (a.len() * b.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let con = WgpuContext::new().unwrap();
    let buffers = [a, b]
        .iter()
        .map(|i| flatten_content(i))
        .map(|i| con.storage_buf(&i).expect("Failed to create buffer"))
        .collect::<Vec<wgpu::Buffer>>();
    let mut buffers = buffers.iter().map(|b| b).collect::<Vec<&wgpu::Buffer>>();
    let out_buf = con.read_write_buf(size)?;

    buffers.push(&out_buf);
    con.compute_gpu::<f32>(&shader, &mut buffers, dis, 1)?;
    con.get_data::<f32>(&out_buf)
}

fn flatten_content(content: &Vec<Vec<f32>>) -> Vec<f32> {
    content.iter().flatten().cloned().collect()
}

pub fn get_for_shader(
    length_a: usize,
    length_b: usize,
    length_inner: usize,
    workgroup_size: (u32, u32, u32),
) -> ComputeShader {
    let a = Var::from("a");
    let b = Var::from("b");
    let out = Var::from("out");
    let vdot = Var::from("dot");
    let blen = Var::from("blen");
    let ilen = Var::from("ilen");
    let alen = Var::from("alen");
    let tidx = Var::from("tidx");
    let tidy = Var::from("tidy");

    let vars = vec![
        (ilen.clone(), Var::from_num(length_inner as u32)),
        (blen.clone(), Var::from_num(length_b as u32)),
        (alen.clone(), Var::from_num(length_a as u32)),
        (
            tidx.clone(),
            Var::WorkgroupIdX
                .multiply(&Var::WorkSizeX)
                .add(&Var::LocalInvocationIdX),
        ),
        (
            tidy.clone(),
            Var::WorkgroupIdY
                .multiply(&Var::WorkSizeY)
                .add(&Var::LocalInvocationIdY),
        ),
    ];

    let obj = ReturnType::Obj(Object::Array(Type::F32, None));
    let binds = vec![(&a, false), (&b, false), (&out, true)];

    ComputeShader::new(binds, &obj, workgroup_size)
        .add_variables(vars)
        .add_line(Line::from(FlowControl::If(
            tidx.compare(&alen, Comparison::GreaterThenOrEqual).compare(
                &tidy.compare(&blen, Comparison::GreaterThenOrEqual),
                Comparison::Or,
            ),
            Body::new()
                .add_line(Line::from(FlowControl::Return(None)))
                .finish(),
        )))
        .add_line(Line::from(Instruction::DefineMutVar {
            lhs: vdot.clone(),
            rhs: Var::from_num(0.),
        }))
        .add_for_loop(
            "i",
            Var::from_num(0_u32),
            Comparison::LessThen,
            ilen.clone(),
            Var::from_num(1_u32),
            Body::new()
                .add_line(Line::from(Instruction::Set {
                    lhs: vdot.clone(),
                    rhs: vdot.add(
                        &a.index(&tidx.multiply(&ilen).add(&Var::from("i")))
                            .multiply(&b.index(&tidy.multiply(&ilen).add(&Var::from("i")))),
                    ),
                }))
                .finish(),
        )
        .add_line(Line::from(Instruction::Set {
            lhs: out.index(&tidx.multiply(&blen).add(&tidy)),
            rhs: vdot.clone(),
        }))
        .finish()
}

pub fn get_par_shader(
    length_a: usize,
    length_b: usize,
    length_inner: usize,
    chunk_size: u32,
    workgroup_size: (u32, u32, u32),
) -> ComputeShader {
    let a = Var::from("a");
    let b = Var::from("b");
    let i = Var::from("i");
    let end = Var::from("end");
    let out = Var::from("out");
    let blen = Var::from("blen");
    let ilen = Var::from("ilen");
    let alen = Var::from("alen");
    let tidx = Var::from("tidx");
    let tidy = Var::from("tidy");
    let temp = Var::from("temp");
    let chunk = Var::from("chunk");
    let start = Var::from("start");
    let next_ilen = Var::from("next_ilen");
    let work_size = Var::from("work_size");

    let next_inner_len = get_next_ilen(length_inner, chunk_size);
    println!("next_inner_len: {}", next_inner_len);
    let temp_size = (chunk_size * workgroup_size.0 / length_inner as u32) * next_inner_len;

    let vars = vec![
        (ilen.clone(), Var::from_num(length_inner as u32)),
        (blen.clone(), Var::from_num(length_b as u32)),
        (alen.clone(), Var::from_num(length_a as u32)),
        (chunk.clone(), Var::from_num(chunk_size)),
        (
            work_size.clone(),
            chunk.multiply(&Var::WorkSizeX).divide(&ilen),
        ),
        (tidx.clone(), Var::WorkgroupIdX.multiply(&work_size)),
        (tidy.clone(), Var::WorkgroupIdY),
        (next_ilen.clone(), Var::from_num(next_inner_len)),
    ];

    let obj = ReturnType::Obj(Object::Array(Type::F32, None));
    let binds = vec![(&a, false), (&b, false), (&out, true)];

    ComputeShader::new(binds, &obj, workgroup_size)
        .add_variables(vars)
        .add_line(Line::from(FlowControl::If(
            tidx.compare(&alen, Comparison::GreaterThenOrEqual).compare(
                &tidy.compare(&blen, Comparison::GreaterThenOrEqual),
                Comparison::Or,
            ),
            Body::new()
                .add_line(Line::from(FlowControl::Return(None)))
                .finish(),
        )))
        .add_line(Line::from(Instruction::DefineMutVar {
            lhs: start.clone(),
            rhs: Var::LocalInvocationIdX.multiply(&chunk),
        }))
        .add_line(Line::from(Instruction::DefineMutVar {
            lhs: end.clone(),
            rhs: start.add(&chunk),
        }))
        .add_line(Line::from(Instruction::DefineVar {
            lhs: Var::from("over"),
            rhs: Var::from("end % (ilen * work_size)"),
        }))
        .add_line(Line::from(Instruction::Set {
            lhs: end.clone(),
            rhs: Var::from("end - over * (end / (ilen * work_size))"),
        }))
        .add_for_loop(
            "i",
            start.clone(),
            Comparison::LessThen,
            end.clone(),
            Var::from_num(1_u32),
            Body::new()
                .add_line(Line::from(Instruction::Set {
                    lhs: temp.index(&Var::LocalInvocationIdX.add(&i.divide(&ilen))),
                    rhs: temp
                        .index(&Var::LocalInvocationIdX.add(&i.divide(&ilen)))
                        .add(
                            &a.index(&tidx.multiply(&ilen).add(&i))
                                .multiply(&b.index(&tidy.multiply(&ilen).add(&i.modulo(&ilen)))),
                        ),
                }))
                .finish(),
        )
        .add_line(Line::from(FlowControl::WorkgroupBarrier))
        .add_const(Line::from(Instruction::DefineConstant {
            vis: Visability::Workgroup,
            var: Var::TypedVar(
                Box::new(temp.clone()),
                ReturnType::Obj(Object::Array(Type::F32, Some(temp_size))),
            ),
        }))
        .add_line(Line::from(FlowControl::If(
            Var::LocalInvocationIdX.compare(&work_size, Comparison::LessThen),
            Body::new()
                .add_line(Line::from(Instruction::Set {
                    lhs: start.clone(),
                    rhs: Var::LocalInvocationIdX.multiply(&next_ilen),
                }))
                .add_line(Line::from(Instruction::Set {
                    lhs: end.clone(),
                    rhs: start.add(&next_ilen),
                }))
                .add_line(Line::from(FlowControl::For(
                    Instruction::DefineMutVar {
                        lhs: i.clone(),
                        rhs: start.clone(),
                    },
                    i.compare(&end, Comparison::LessThen),
                    Instruction::Set {
                        lhs: i.clone(),
                        rhs: i.add(&Var::from_num(1_u32)),
                    },
                    Body::new()
                        .add_line(Line::from(Instruction::Set {
                            lhs: out.index(
                                &tidx
                                    .multiply(&blen)
                                    .add(&Var::LocalInvocationIdX.multiply(&blen).add(&tidy)),
                            ),
                            rhs: out
                                .index(
                                    &tidx
                                        .multiply(&blen)
                                        .add(&Var::LocalInvocationIdX.multiply(&blen).add(&tidy)),
                                )
                                .add(&temp.index(&i)),
                        }))
                        .finish(),
                )))
                .finish(),
        )))
        .finish()
}

fn get_next_ilen(ilen: usize, chunk: u32) -> u32 {
    let mut next_ilen = ilen as u32;
    while next_ilen % chunk != 0 {
        next_ilen += 1;
    }
    next_ilen / chunk
}

#[test]
fn test_shader_construction() {
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
    let expected_shader = "@group(0)
@binding(0)
var<storage, read> a: array<f32>;

@group(0)
@binding(1)
var<storage, read> b: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> out: array<f32>;

const workgroup_size: vec3<u32> = vec3<u32>(16, 16, 1);
@workgroup_size(16, 16, 1)
@compute
fn main(
@builtin(local_invocation_id) lid: vec3<u32>,
@builtin(num_workgroups) num_wgs: vec3<u32>,
@builtin(workgroup_id) wid: vec3<u32>,
) {
var alen = 14u;
var blen = 4u;
var tidx = wid.x * workgroup_size.x + lid.x;
out[tidx] = alen * blen;

}";
    assert_eq!(shader.to_string(), expected_shader);
}
