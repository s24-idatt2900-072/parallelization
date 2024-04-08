use wgpu_test::wgpu_context::WgpuContext;
use wgpu_test::WgpuContextError;
use wgsl::*;

fn main() {
    let a = Var::StdVar("a".to_string());
    let b = Var::StdVar("b".to_string());
    let out = Var::StdVar("out".to_string());
    let blen = Var::StdVar("blen".to_string());
    let ilen = Var::StdVar("ilen".to_string());
    let alen = Var::StdVar("alen".to_string());
    let vdot = Var::StdVar("dot".to_string());
    let i = Var::StdVar("i".to_string());
    let tidx = Var::WorkgroupIdX
        .multiply(&Var::WorkSizeX)
        .add(&Var::LocalInvocationIdX);
    let tidy = Var::WorkgroupIdY
        .multiply(&Var::WorkSizeY)
       .add(&Var::LocalInvocationIdY);    

    let obj = ReturnType::Obj(Object::Array(Type::F32, None));
    let binds = vec![
        (&a, false),
        (&b, false),
        (&out, true),
    ];
    let shader = ComputeShader::new(binds, &obj, (32, 32, 1))
        .add_line(Line::Ins(Instruction::DefineVar {
            lhs: ilen.clone(),
            rhs: Var::StdVar("841u".to_string()),
        }))
        .add_line(Line::Ins(Instruction::DefineVar {
            lhs: blen.clone(),
            rhs: Var::StdVar("4u".to_string()),
        }))
        .add_line(Line::Ins(Instruction::DefineVar {
            lhs: alen.clone(),
            rhs: Var::StdVar("14u".to_string()),
        }))
        .add_line(Line::Flw(FlowControl::If(
            tidx.compare(&alen, Comparison::GreaterThenOrEqual)
            .compare(
                &tidy.compare(&blen, Comparison::GreaterThenOrEqual),
                Comparison::Or,
            ),
            Body::new()
                .add_line(Line::Flw(FlowControl::Return(None)))
                .finish(),
        )))
        .add_line(Line::Ins(Instruction::DefineMutVar {
            lhs: vdot.clone(),
            rhs: Var::StdVar("0.0".to_string()),
        }))
        .add_line(Line::Flw(FlowControl::For(
            Instruction::DefineMutVar {
                lhs: i.clone(),
                rhs: Var::StdVar("0u".to_string()),
            },
            i.compare(&ilen, Comparison::LessThen),
            Instruction::Set {
                lhs: i.clone(),
                rhs: i.add(&Var::StdVar("1u".to_string())),
            },
            Body::new()
                .add_line(Line::Ins(Instruction::Set {
                lhs: vdot.clone(),
                rhs: vdot.add(
                    &a.index(&tidx.multiply(&ilen).add(&i))
                        .multiply(
                            &b.index(&tidy.multiply(&ilen).add(&i)))),
                }))
                .finish(),
        )))
        .add_line(Line::Ins(Instruction::Set {
            lhs: out.index(&tidx.multiply(&blen).add(&tidy)),
            rhs: vdot.clone(),
        })) // maybe add a hashmap of vars to use
        .finish();

    println!("\nHERE IS MY SHADER:\n\n{}\n", shader);

    let a: Vec<Vec<f32>> = vec![vec![1.; 841]; 14];
    let b: Vec<Vec<f32>> = vec![vec![1.; 841]; 4];
    let res = dot(&a, &b, (100, 100, 1), format!("{shader}")).unwrap();
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
