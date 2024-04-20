pub mod errors;
pub mod shader;
pub mod utils;
pub mod variabel;

pub use errors::*;
pub use shader::*;
pub use utils::*;
pub use variabel::*;

pub fn get_for_loop_cosine_similarity_shader(
    image_len: usize,
    filter_len: usize,
    inner_len: usize,
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
        (ilen.clone(), Var::from_num(inner_len as u32)),
        (blen.clone(), Var::from_num(filter_len as u32)),
        (alen.clone(), Var::from_num(image_len as u32)),
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

fn get_par_shader(
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
