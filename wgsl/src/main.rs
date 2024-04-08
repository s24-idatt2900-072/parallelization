use wgpu_test::wgpu_context::WgpuContext;
use wgpu_test::WgpuContextError;
use wgsl::*;

fn main() {
    let a_data: Vec<Vec<f32>> = vec![vec![1.; 841]; 14];
    let b_data: Vec<Vec<f32>> = vec![vec![1.; 841]; 4];

    let a = Var::from("a");
    let b = Var::from("b");
    let out = Var::from("out");
    let blen = Var::from("blen");
    let ilen = Var::from("ilen");
    let alen = Var::from("alen");
    let vdot = Var::from("dot");
    let i = Var::from("i");
    let tidx = Var::WorkgroupIdX
        .multiply(&Var::WorkSizeX)
        .add(&Var::LocalInvocationIdX);
    let tidy = Var::WorkgroupIdY
        .multiply(&Var::WorkSizeY)
        .add(&Var::LocalInvocationIdY);

    let obj = ReturnType::Obj(Object::Array(Type::F32, None));
    let binds = vec![(&a, false), (&b, false), (&out, true)];
    let shader = ComputeShader::new(binds, &obj, (32, 32, 1))
        .add_line(Line::from(Instruction::DefineVar {
            lhs: ilen.clone(),
            rhs: Var::from_num(a_data[0].len() as u32),
        }))
        .add_line(Line::from(Instruction::DefineVar {
            lhs: blen.clone(),
            rhs: Var::from_num(b_data.len() as u32),
        }))
        .add_line(Line::from(Instruction::DefineVar {
            lhs: alen.clone(),
            rhs: Var::from_num(a_data.len() as u32),
        }))
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
        .add_line(Line::from(FlowControl::For(
            Instruction::DefineMutVar {
                lhs: i.clone(),
                rhs: Var::from_num(0_u32),
            },
            i.compare(&ilen, Comparison::LessThen),
            Instruction::Set {
                lhs: i.clone(),
                rhs: i.add(&Var::from_num(1_u32)),
            },
            Body::new()
                .add_line(Line::from(Instruction::Set {
                    lhs: vdot.clone(),
                    rhs: vdot.add(
                        &a.index(&tidx.multiply(&ilen).add(&i))
                            .multiply(&b.index(&tidy.multiply(&ilen).add(&i))),
                    ),
                }))
                .finish(),
        )))
        .add_line(Line::from(Instruction::Set {
            lhs: out.index(&tidx.multiply(&blen).add(&tidy)),
            rhs: vdot.clone(),
        }))
        .finish();

    println!("\nHERE IS MY SHADER:\n\n{}\n", shader);
    let res = dot(&a_data, &b_data, (100, 100, 1), format!("{shader}")).unwrap();
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
