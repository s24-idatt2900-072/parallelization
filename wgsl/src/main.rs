use wgpu_test::wgpu_context::WgpuContext;
use wgpu_test::WgpuContextError;
use wgsl::*;

fn main() {
    let a_data: Vec<Vec<f32>> = vec![vec![1.; 841]; 4];
    let b_data: Vec<Vec<f32>> = vec![vec![1.; 841]; 14];

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
        (ilen.clone(), Var::from_num(a_data[0].len() as u32)),
        (blen.clone(), Var::from_num(b_data.len() as u32)),
        (alen.clone(), Var::from_num(a_data.len() as u32)),
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
    let shader = ComputeShader::new(binds, &obj, (32, 32, 1))
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
        .finish();

    println!("\nHERE IS THE SHADER:\n\n{}\n", shader);
    let res = dot(&a_data, &b_data, (2_000, 4_000, 1), format!("{shader}")).unwrap();
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
    println!("shader: {}", shader);
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
