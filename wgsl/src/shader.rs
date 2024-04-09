use crate::{errors::*, utils::*, variabel::*};
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct ComputeShader {
    pub bindings: Vec<Binding>,
    pub work_size: WorkgroupSize,
    pub bins: Vec<BuiltIns>,
    pub body: Body,
    // pub extensions: Vec<Extension>, functions and stuff
}

impl ComputeShader {
    pub fn new(binds: Vec<(&Var, bool)>, obj: &ReturnType, work_size: (u32, u32, u32)) -> Self {
        let bindings = Binding::new(binds, obj);
        let (x, y, z) = work_size;
        let work_size = WorkgroupSize::new(x, y, z).unwrap();
        let bins = BuiltIns::all();
        let body = Body::new();
        Self {
            bindings,
            work_size,
            bins,
            body,
        }
    }

    pub fn add_line(&mut self, ins: Line) -> &mut Self {
        self.body.add_line(ins);
        self
    }

    pub fn add_for_loop(
        &mut self,
        i: &str,
        initial: Var,
        comp: Comparison,
        other: Var,
        step: Var,
        body: Body,
    ) -> &mut Self {
        let i = Var::from(i);
        self.body.add_line(Line::from(FlowControl::For(
            Instruction::DefineMutVar {
                lhs: i.clone(),
                rhs: initial,
            },
            i.compare(&other, comp),
            Instruction::Set {
                lhs: i.clone(),
                rhs: i.add(&step),
            },
            body,
        )));
        self
    }

    pub fn add_variables(&mut self, vars: Vec<(Var, Var)>) -> &mut Self {
        for (v, n) in vars {
            self.body.add_line(Line::from(Instruction::DefineMutVar {
                lhs: v.clone(),
                rhs: n,
            }));
        }
        self
    }

    pub fn finish(&mut self) -> Self {
        self.clone()
    }
}

impl Display for ComputeShader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.bindings {
            f.write_fmt(format_args!("{}\n", b))?;
        }
        f.write_fmt(format_args!("{}", self.work_size))?;
        f.write_fmt(format_args!("@compute\nfn main(\n"))?;
        for bin in &self.bins {
            f.write_fmt(format_args!("{}", bin))?;
        }
        f.write_fmt(format_args!(") {{\n{}\n}}", self.body))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Binding {
    vis: Visability,
    gid: u32,
    bid: u32,
    acc: Access,
    var: Var,
}

impl Binding {
    fn new(binds: Vec<(&Var, bool)>, obj: &ReturnType) -> Vec<Self> {
        binds
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let (var, write) = r;
                let var = Var::TypedVar(Box::new(var.clone()), obj.clone());
                let vis = Visability::Storage;
                let gid = 0_u32;
                let bid = i as u32;
                let acc = match write {
                    true => Access::ReadWrite,
                    false => Access::Read,
                };
                Self {
                    vis,
                    gid,
                    bid,
                    acc,
                    var,
                }
            })
            .collect::<Vec<Self>>()
    }
}

impl Display for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("@group({})\n", self.gid))?;
        f.write_fmt(format_args!("@binding({})\n", self.bid))?;
        f.write_fmt(format_args!(
            "var<{}, {}> {};\n",
            self.vis, self.acc, self.var
        ))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum Access {
    Read,
    ReadWrite,
}

impl Display for Access {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let display = match *self {
            Self::Read => format!("{:?}", *self).to_lowercase(),
            Self::ReadWrite => String::from("read_write"),
        };
        f.write_fmt(format_args!("{}", display))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum Visability {
    Storage,
    Workgroup,
}

impl Display for Visability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let display = format!("{:?}", self).to_lowercase();
        f.write_fmt(format_args!("{}", display))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct WorkgroupSize {
    x: u32,
    y: u32,
    z: u32,
}

impl WorkgroupSize {
    fn new(x: u32, y: u32, z: u32) -> Result<Self, WgslError> {
        let max_workgroup_size = 1024;
        if x * y * z > max_workgroup_size {
            return Err(WgslError::WorkgroupSizeError(format!(
                "Error creating workgroup. Workgroup size can't be larger than {}",
                max_workgroup_size
            )));
        }
        let max_size_z = 64;
        if z > max_size_z {
            return Err(WgslError::WorkgroupSizeError(format!(
                "Error creating workgroup. Workgroup size z can't be larger than {}",
                max_size_z
            )));
        }
        Ok(Self { x, y, z })
    }
}

impl Display for WorkgroupSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (x, y, z) = (self.x, self.y, self.z);
        f.write_fmt(format_args!("const workgroup_size: vec3<u32> = vec3<u32>({x}, {y}, {z});\n@workgroup_size({x}, {y}, {z})\n"))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum BuiltIns {
    LocalInvocationId,
    WorkgroupId,
    NumWorkgroups,
}

impl BuiltIns {
    fn all() -> Vec<Self> {
        vec![
            BuiltIns::LocalInvocationId,
            BuiltIns::WorkgroupId,
            BuiltIns::NumWorkgroups,
        ]
    }
}

impl Display for BuiltIns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (call, name) = match *self {
            Self::LocalInvocationId => (String::from("local_invocation_id"), String::from("lid")),
            Self::NumWorkgroups => (String::from("workgroup_id"), String::from("wid")),
            Self::WorkgroupId => (String::from("num_workgroups"), String::from("num_wgs")),
        };
        let obj = Object::Vec3(Type::U32);
        let obj = ReturnType::Obj(obj);
        let v = Var::StdVar(name);
        let v = Var::TypedVar(Box::new(v), obj);
        f.write_fmt(format_args!("@builtin({}) {},\n", call, v))?;
        Ok(())
    }
}
