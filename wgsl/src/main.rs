use wgsl::*;
use wgpu_test::wgpu_context::WgpuContext;
use wgpu_test::WgpuContextError;

fn main() {
    let a = Var::StdVar("a".to_string());
    let b = Var::StdVar("b".to_string());
    let out = Var::StdVar("out".to_string());

    let obj = Object::Array(Type::F32, None);
    let binds = vec![(Var::TypedVar(Box::new(a.clone()), obj), false), (Var::TypedVar(Box::new(b.clone()), obj), false), (Var::TypedVar(Box::new(out.clone()), obj), true)];
    
    let mut shader = ComputeShader::new(binds, (32, 32, 1));

    let tidx = Var::WorkgroupIdX.multiply(Var::WorkSizeX).add(Var::LocalInvocationIdX);
    let tidy = Var::WorkgroupIdY.multiply(Var::WorkSizeY).add(Var::LocalInvocationIdY);
    let blen = Var::StdVar("blen".to_string());
    let ilen = Var::StdVar("ilen".to_string());
    let alen = Var::StdVar("alen".to_string());
    let vdot = Var::StdVar("dor".to_string());
    let i = Var::StdVar("i".to_string());

    let mut if_body = Body::new();
    if_body.add_line(Line::Flw(FlowControl::Return(None)));

    let mut for_body = Body::new();
    for_body.add_line(Line::Ins(Instruction::Set { lhs: vdot.clone(), rhs: vdot.clone().add(a.clone().index(tidx.clone().multiply(ilen.clone()).add(i.clone())).multiply(b.clone().index(tidy.clone().multiply(ilen.clone()).add(i.clone())))) }));
    // implement copy so u dont ave to clone all the time...
    shader
    // Add stdvar number so you dont have to to string it. maybe a from method??
        .add_line(Line::Ins(Instruction::DefineVar { lhs: ilen.clone(), rhs: Var::StdVar("841u".to_string()) }))
        .add_line(Line::Ins(Instruction::DefineVar { lhs: blen.clone(), rhs: Var::StdVar("4u".to_string()) }))
        .add_line(Line::Ins(Instruction::DefineVar { lhs: alen.clone(), rhs: Var::StdVar("14u".to_string()) }))
        .add_line(Line::Flw(FlowControl::If(tidx.clone().compare(alen.clone(), Comparison::GreaterThenOrEqual).compare(tidy.clone().compare(blen.clone(), Comparison::GreaterThenOrEqual), Comparison::Or), if_body)))
        .add_line(Line::Ins(Instruction::DefineMutVar { lhs: vdot.clone(), rhs: Var::StdVar("0.0".to_string()) }))
        .add_line(Line::Flw(FlowControl::For(Instruction::DefineMutVar { lhs: i.clone(), rhs: Var::StdVar("0u".to_string()) }, i.clone().compare(ilen.clone(), Comparison::LessThen), Instruction::Set { lhs: i.clone(), rhs: i.clone().add(Var::StdVar("1u".to_string())) }, for_body)))
        .add_line(Line::Ins(Instruction::Set { lhs: out.index(tidx.multiply(blen).add(tidy)), rhs: vdot })); // maybe add a hashmap of vars to use
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