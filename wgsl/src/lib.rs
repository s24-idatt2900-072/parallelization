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
    let i = Var::from("i");
    let re = Var::from("re");
    let abs = Var::from("abs");
    let out = Var::from("out");
    let dot = Var::from("dot");
    let norm = Var::from("norm");
    let ilen = Var::from("ilen");
    let tidx = Var::from("tidx");
    let tidy = Var::from("tidy");
    let temp = Var::from("temp");
    let image = Var::from("image");
    let im_len = Var::from("image_len");
    let fi_len = Var::from("filter_len");

    let vars = vec![
        (ilen.clone(), Var::from_num(inner_len as u32)),
        (fi_len.clone(), Var::from_num(filter_len as u32)),
        (im_len.clone(), Var::from_num(image_len as u32)),
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
    let binds = vec![(&image, false), (&re, false), (&abs, false), (&out, true)];

    ComputeShader::new(binds, &obj, workgroup_size)
        .add_variables(vars)
        .add_line(Line::from(FlowControl::If(
            tidx.compare(&im_len, Comparison::GreaterThenOrEqual)
                .compare(
                    &tidy.compare(&fi_len, Comparison::GreaterThenOrEqual),
                    Comparison::Or,
                ),
            Body::new()
                .add_line(Line::from(FlowControl::Return(None)))
                .finish(),
        )))
        .add_line(Line::from(Instruction::DefineMutVar {
            lhs: dot.clone(),
            rhs: Var::from_num(0.),
        }))
        .add_line(Line::from(Instruction::DefineMutVar {
            lhs: norm.clone(),
            rhs: Var::from_num(0.),
        }))
        .add_for_loop(
            "i",
            Var::from_num(0_u32),
            Comparison::LessThen,
            ilen.clone(),
            Var::from_num(1_u32),
            Body::new()
                .add_line(Line::from(Instruction::DefineVar {
                    lhs: temp.clone(),
                    rhs: image
                        .index(&&tidx.multiply(&ilen).add(&i))
                        .multiply(&abs.index(&&tidy.multiply(&ilen).add(&i))),
                }))
                .add_line(Line::from(Instruction::Set {
                    lhs: dot.clone(),
                    rhs: dot.add(&temp.multiply(&re.index(&tidy.multiply(&ilen).add(&i)))),
                }))
                .add_line(Line::from(Instruction::Set {
                    lhs: norm.clone(),
                    rhs: norm.add(&temp.multiply(&temp)),
                }))
                .finish(),
        )
        .add_line(Line::from(Instruction::Set {
            lhs: out.index(&tidx.multiply(&fi_len).add(&tidy)),
            rhs: dot.divide(&norm.sqrt()),
        }))
        .finish()
}

pub fn get_for_loop_max_pool_shader(
    inner_len: u64,
    elements: usize,
    workgroup_size: (u32, u32, u32),
) -> ComputeShader {
    let i = Var::from("i");
    let out = Var::from("out");
    let tid = Var::from("tid");
    let max = Var::from("max");
    let ilen = Var::from("ilen");
    let elems = Var::from("elements");
    let feat = Var::from("features");
    let vars = vec![
        (elems.clone(), Var::from_num(elements as u32)),
        (ilen.clone(), Var::from_num(inner_len as u32)),
        (
            tid.clone(),
            Var::WorkgroupIdX
                .multiply(&Var::WorkSizeX)
                .add(&Var::LocalInvocationIdX),
        ),
    ];

    let obj = ReturnType::Obj(Object::Array(Type::F32, None));
    let binds = vec![(&feat, false), (&out, true)];

    ComputeShader::new(binds, &obj, workgroup_size)
        .add_variables(vars)
        .add_line(Line::from(FlowControl::If(
            tid.compare(&elems, Comparison::GreaterThenOrEqual),
            Body::new()
                .add_line(Line::from(FlowControl::Return(None)))
                .finish(),
        )))
        .add_line(Line::from(Instruction::DefineMutVar {
            lhs: max.clone(),
            rhs: Var::from_num(0.),
        }))
        .add_for_loop(
            i.to_string().as_str(),
            Var::from_num(0_u32),
            Comparison::LessThen,
            ilen.clone(),
            Var::from_num(1_u32),
            Body::new()
                .add_line(Line::from(FlowControl::If(
                    feat.index(&tid.multiply(&ilen).add(&i))
                        .compare(&max, Comparison::GreaterThen),
                    Body::new()
                        .add_line(Line::from(Instruction::Set {
                            lhs: max.clone(),
                            rhs: feat.index(&tid.multiply(&ilen).add(&i)),
                        }))
                        .finish(),
                )))
                .finish(),
        )
        .add_line(Line::from(Instruction::Set {
            lhs: out.index(&tid),
            rhs: max,
        }))
        .finish()
}

pub fn get_parallel_cosine_similarity_shader(
    image_len: usize,
    filter_len: usize,
    inner_len: usize,
    chunk_size: u32,
    workgroup_size: (u32, u32, u32),
) -> ComputeShader {
    let i = Var::from("i");
    let re = Var::from("re");
    let abs = Var::from("abs");
    let end = Var::from("end");
    let out = Var::from("out");
    let dot = Var::from("dot");
    let norm = Var::from("norm");
    let ilen = Var::from("ilen");
    let tidx = Var::from("tidx");
    let tidy = Var::from("tidy");
    let over = Var::from("over");
    let temp = Var::from("temp");
    let chunk = Var::from("chunk");
    let start = Var::from("start");
    let image = Var::from("image");
    let im_len = Var::from("image_len");
    let temp_dot = Var::from("temp_dot");
    let norm_dot = Var::from("norm_dot");
    let fi_len = Var::from("filter_len");
    let next_ilen = Var::from("next_ilen");
    let work_size = Var::from("work_size");

    let next_inner_len = get_next_ilen(inner_len, chunk_size);
    let buffer_size = (chunk_size * workgroup_size.0 / inner_len as u32) * next_inner_len;

    let vars = vec![
        (ilen.clone(), Var::from_num(inner_len as u32)),
        (fi_len.clone(), Var::from_num(filter_len as u32)),
        (im_len.clone(), Var::from_num(image_len as u32)),
        (chunk.clone(), Var::from_num(chunk_size)),
        (
            work_size.clone(),
            chunk.multiply(&Var::WorkSizeX).divide(&ilen),
        ),
        (tidx.clone(), Var::WorkgroupIdX),
        (tidy.clone(), Var::WorkgroupIdY.multiply(&work_size)),
        (next_ilen.clone(), Var::from_num(next_inner_len)),
    ];

    let obj = ReturnType::Obj(Object::Array(Type::F32, None));
    let binds = vec![(&image, false), (&re, false), (&abs, false), (&out, true)];

    ComputeShader::new(binds, &obj, workgroup_size)
        .add_variables(vars)
        .add_line(Line::from(FlowControl::If(
            tidx.compare(&im_len, Comparison::GreaterThenOrEqual)
                .compare(
                    &tidy.compare(&fi_len, Comparison::GreaterThenOrEqual),
                    Comparison::Or,
                ),
            Body::new()
                .add_line(Line::from(FlowControl::Return(None)))
                .finish(),
        )))
        .add_line(Line::from(Instruction::DefineVar {
            lhs: start.clone(),
            rhs: Var::LocalInvocationIdX.multiply(&chunk),
        }))
        .add_line(Line::from(Instruction::DefineMutVar {
            lhs: end.clone(),
            rhs: start.add(&chunk),
        }))
        .add_line(Line::from(Instruction::DefineVar {
            lhs: over.clone(),
            rhs: end.modulo(&ilen.multiply(&work_size).parenthesis()),
        }))
        .add_line(Line::from(Instruction::Set {
            lhs: end.clone(),
            rhs: end.sub(
                &over.multiply(
                    &end.divide(&ilen.multiply(&work_size).parenthesis())
                        .parenthesis(),
                ),
            ),
        }))
        .add_for_loop(
            "i",
            start.clone(),
            Comparison::LessThen,
            end.clone(),
            Var::from_num(1_u32),
            Body::new()
                .add_line(Line::from(Instruction::DefineVar {
                    lhs: temp.clone(),
                    rhs: image
                        .index(&tidx.multiply(&ilen).add(&i.modulo(&ilen).parenthesis()))
                        .multiply(&abs.index(&tidy.multiply(&ilen).add(&i))),
                }))
                .add_line(Line::from(Instruction::Set {
                    lhs: temp_dot.index(&Var::LocalInvocationIdX.add(&i.divide(&ilen))),
                    rhs: temp_dot
                        .index(&Var::LocalInvocationIdX.add(&i.divide(&ilen)))
                        .add(&temp.multiply(&re.index(&tidy.multiply(&ilen).add(&i)))),
                }))
                .add_line(Line::from(Instruction::Set {
                    lhs: norm_dot.index(&Var::LocalInvocationIdX.add(&i.divide(&ilen))),
                    rhs: norm_dot
                        .index(&Var::LocalInvocationIdX.add(&i.divide(&ilen)))
                        .add(&temp.multiply(&temp)),
                }))
                .finish(),
        )
        .add_line(Line::from(FlowControl::WorkgroupBarrier))
        .add_const(Line::from(Instruction::DefineConstant {
            vis: Visability::Workgroup,
            var: Var::TypedVar(
                Box::new(temp_dot.clone()),
                ReturnType::Obj(Object::Array(Type::F32, Some(buffer_size))),
            ),
        }))
        .add_const(Line::from(Instruction::DefineConstant {
            vis: Visability::Workgroup,
            var: Var::TypedVar(
                Box::new(norm_dot.clone()),
                ReturnType::Obj(Object::Array(Type::F32, Some(buffer_size))),
            ),
        }))
        .add_line(Line::from(FlowControl::If(
            Var::LocalInvocationIdX
                .compare(&work_size, Comparison::LessThen)
                .compare(
                    &Var::LocalInvocationIdX
                        .add(&tidy)
                        .compare(&fi_len, Comparison::LessThen),
                    Comparison::And,
                ),
            Body::new()
                .add_line(Line::from(Instruction::DefineVar {
                    lhs: start.clone(),
                    rhs: Var::LocalInvocationIdX.multiply(&next_ilen),
                }))
                .add_line(Line::from(Instruction::Set {
                    lhs: end.clone(),
                    rhs: start.add(&next_ilen),
                }))
                .add_line(Line::from(Instruction::DefineMutVar {
                    lhs: dot.clone(),
                    rhs: Var::from_num(0.),
                }))
                .add_line(Line::from(Instruction::DefineMutVar {
                    lhs: norm.clone(),
                    rhs: Var::from_num(0.),
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
                            lhs: dot.clone(),
                            rhs: dot.add(&temp_dot.index(&i)),
                        }))
                        .add_line(Line::from(Instruction::Set {
                            lhs: norm.clone(),
                            rhs: norm.add(&norm_dot.index(&i)),
                        }))
                        .finish(),
                )))
                .add_line(Line::from(Instruction::Set {
                    lhs: out.index(
                        &tidx
                            .multiply(&fi_len)
                            .add(&Var::LocalInvocationIdX)
                            .add(&tidy),
                    ),
                    rhs: dot.divide(&norm.sqrt()),
                }))
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

pub fn get_parallel_max_pool_shader(
    inner_len: u64,
    chunk_size: u32,
    workgroup_size: (u32, u32, u32),
) -> ComputeShader {
    let i = Var::from("i");
    let out = Var::from("out");
    let tid = Var::from("tid");
    let end = Var::from("end");
    let ilen = Var::from("ilen");
    let over = Var::from("over");
    let temp = Var::from("temp");
    let start = Var::from("start");
    let chunk = Var::from("chunk");
    let feat = Var::from("features");
    let elems = Var::from("elements");
    let out_index = Var::from("out_index");
    let work_size = Var::from("work_size");
    let next_ilen = Var::from("next_ilen");

    let next_inner_len = get_next_ilen(inner_len as usize, chunk_size);
    let vars = vec![
        (chunk.clone(), Var::from_num(chunk_size)),
        (ilen.clone(), Var::from_num(inner_len as u32)),
        (elems.clone(), feat.array_length().divide(&ilen)),
        (next_ilen.clone(), Var::from_num(next_inner_len)),
        (
            work_size.clone(),
            chunk.multiply(&Var::WorkSizeX).divide(&ilen),
        ),
        (tid.clone(), Var::WorkgroupIdX.multiply(&Var::WorkSizeX)),
        (start.clone(), Var::LocalInvocationIdX.multiply(&chunk)),
    ];
    let buffer_size = (chunk_size * workgroup_size.0 / inner_len as u32) * next_inner_len;
    let obj = ReturnType::Obj(Object::Array(Type::F32, None));
    let binds = vec![(&feat, false), (&out, true)];

    ComputeShader::new(binds, &obj, workgroup_size)
        .add_variables(vars)
        .add_const(Line::from(Instruction::DefineConstant {
            vis: Visability::Workgroup,
            var: Var::TypedVar(
                Box::new(temp.clone()),
                ReturnType::Obj(Object::Array(Type::F32, Some(buffer_size))),
            ),
        }))
        .add_line(Line::from(FlowControl::If(
            tid.compare(&elems, Comparison::GreaterThenOrEqual),
            Body::new()
                .add_line(Line::from(FlowControl::Return(None)))
                .finish(),
        )))
        .add_line(Line::from(Instruction::DefineMutVar {
            lhs: end.clone(),
            rhs: start.add(&chunk),
        }))
        .add_line(Line::from(Instruction::DefineVar {
            lhs: over.clone(),
            rhs: end.modulo(&ilen.multiply(&work_size).parenthesis()),
        }))
        .add_line(Line::from(Instruction::Set {
            lhs: end.clone(),
            rhs: end.sub(
                &over.multiply(
                    &end.divide(&ilen.multiply(&work_size).parenthesis())
                        .parenthesis(),
                ),
            ),
        }))
        .add_for_loop(
            i.to_string().as_str(),
            start.clone(),
            Comparison::LessThen,
            end.clone(),
            Var::from_num(1_u32),
            Body::new()
                .add_line(Line::from(FlowControl::If(
                    temp.index(&Var::LocalInvocationIdX.add(&i.divide(&ilen)))
                        .compare(
                            &feat.index(&tid.multiply(&ilen).add(&i)),
                            Comparison::LessThen,
                        ),
                    Body::new()
                        .add_line(Line::from(Instruction::Set {
                            lhs: temp.index(&Var::LocalInvocationIdX.add(&i.divide(&ilen))),
                            rhs: feat.index(&tid.multiply(&ilen).add(&i)),
                        }))
                        .finish(),
                )))
                .finish(),
        )
        .add_line(Line::from(FlowControl::WorkgroupBarrier))
        .add_line(Line::from(FlowControl::If(
            Var::LocalInvocationIdX.compare(&work_size, Comparison::LessThen),
            Body::new()
                .add_line(Line::from(Instruction::DefineVar {
                    lhs: start.clone(),
                    rhs: Var::LocalInvocationIdX.multiply(&next_ilen),
                }))
                .add_line(Line::from(Instruction::Set {
                    lhs: end.clone(),
                    rhs: start.add(&next_ilen),
                }))
                .add_line(Line::from(Instruction::DefineVar {
                    lhs: out_index.clone(),
                    rhs: tid.add(&Var::LocalInvocationIdX),
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
                        .add_line(Line::from(FlowControl::If(
                            out.index(&out_index)
                                .compare(&temp.index(&i), Comparison::LessThen),
                            Body::new()
                                .add_line(Line::from(Instruction::Set {
                                    lhs: out.index(&out_index),
                                    rhs: temp.index(&i),
                                }))
                                .finish(),
                        )))
                        .finish(),
                )))
                .finish(),
        )))
        .finish()
}
