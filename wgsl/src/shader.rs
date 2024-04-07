use std::fmt::Display;

pub struct ComputeShader {
    pub bindings: Vec<Binding>,
    pub work_size: WorkgroupSize,
    pub bins: Vec<BuiltIns>,
    pub body: Body,
    // pub extensions: Vec<Extension>, functions and stuff
}

impl ComputeShader {
    pub fn new(binds: Vec<(Var, bool)>, work_size: (u32, u32, u32)) -> Self {
        let bindings = Binding::from(binds);
        let (x, y, z) = work_size;
        let work_size = WorkgroupSize::new(x, y, z).unwrap();
        let bins = BuiltIns::all();
        let body = Body::new();            
        Self { bindings, work_size, bins, body }
    }

    pub fn add_line(&mut self, ins: Line) -> &mut Self {
        self.body.add_line(ins);
        self
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
pub struct Body {
    ins: Vec<Line>
}

impl Body {
    pub fn new() -> Self {
        Self { ins: Vec::new() }
    }

    pub fn add_line(&mut self, i: Line) -> &mut Self {
        self.ins.push(i);
        self
    }
}

impl Display for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ins.iter().for_each(|i|  { f.write_fmt(format_args!("{i};\n")).unwrap(); });
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Line {
    Ins(Instruction),
    Flw(FlowControl),
}

impl Display for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Line::Flw(inner) => f.write_fmt(format_args!("{inner}"))?,
            Line::Ins(inner) => f.write_fmt(format_args!("{inner}"))?,
        };
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum FlowControl {
    If(Var, Body),
    ElseIf(Var, Body),
    Else(Body),
    For(Instruction, Var, Instruction, Body),
    Return(Option<Var>),
    WorkgroupBarrier,
    Break,
}

impl Display for FlowControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let display = match self {
            FlowControl::If(i, b) => format!("if {} {{\n {}\n}}", i, b),
            FlowControl::ElseIf(i, b) => format!("else if {} {{\n{}\n}}", i, b),
            FlowControl::Else(b) => format!("else {{\n{}\n}}", b),
            FlowControl::For(v, cond, i, b) => format!("for ({v}; {cond}; {i}) {{\n{b}\n}}"),
            FlowControl::Return(o) => {
                match o {
                    Some(v) => format!("return {};", v),
                    None => String::from("return;"),
                }
            },
            FlowControl::Break => String::from("break;"),
            FlowControl::WorkgroupBarrier => String::from("WorkgroupBarrier()"),
        };
        f.write_fmt(format_args!("{display}"))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Multiply{lhs: Var, rhs: Var, out: Var},
    Subtract{lhs: Var, rhs: Var, out: Var},
    Divide{lhs: Var, rhs: Var, out: Var},
    Modulo{lhs: Var, rhs: Var, out: Var},
    Add{lhs: Var, rhs: Var, out: Var},
    DefineMutVar{lhs: Var, rhs: Var},
    DefineVar{lhs: Var, rhs: Var},
    Set{lhs: Var, rhs: Var},
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let display = match self {
            Instruction::Subtract { lhs, rhs, out } => format!("{} = {} - {}", out, lhs, rhs),
            Instruction::Multiply { lhs, rhs, out } => format!("{} = {} * {}", out, lhs, rhs),
            Instruction::Divide { lhs, rhs, out } => format!("{} = {} / {}", out, lhs, rhs),
            Instruction::Modulo { lhs, rhs, out } => format!("{} = {} % {}", out, lhs, rhs),
            Instruction::Add { lhs, rhs, out } => format!("{} = {} + {}", out, lhs, rhs),
            Instruction::DefineMutVar { lhs, rhs } => format!("var {} = {}", lhs, rhs),
            Instruction::DefineVar { lhs, rhs } => format!("let {} = {}", lhs, rhs),
            Instruction::Set { lhs, rhs } => format!("{} = {}", lhs, rhs),
        };
        f.write_fmt(format_args!("{display}"))?;
        Ok(())
    }
}

pub enum ReturnType {
    Type,
    Object,
}

impl Display for ReturnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self))?;
        Ok(())
    }
}

pub enum Comparison {
    Or,
    And,
    Equals,
    LessThen,
    GreaterThen,
    NotEqual,
    GreaterThenOrEqual,
    LessThenOrWqual,
}

impl Display for Comparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let display = match self {
            Comparison::Or => " || ",
            Comparison::And => " && ",
            Comparison::Equals => " == ",
            Comparison::LessThen => " < ",
            Comparison::NotEqual => " != ",
            Comparison::GreaterThen => " > ",
            Comparison::LessThenOrWqual => " <= ",
            Comparison::GreaterThenOrEqual => " >= ",
        };
        f.write_fmt(format_args!("{}", display))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Type {
    F32,
    U32,
    I32,
    Bool,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let display = format!("{:?}", *self);
        f.write_fmt(format_args!("{}", display))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Object {
    Vec2(Type),
    Vec3(Type),
    Vec4(Type),
    Array(Type, Option<u32>),
}

impl Display for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stand = format!("{:?}", self).to_lowercase().replace('(', "<").replace(')', ">");
        let display = match self {
            Object::Array(_, s) => {
                match s {
                    Some(_) => {
                        let temp = stand.replace("some<", "");
                        let mut disp = String::new();
                        let i = temp.find(">").unwrap();
                        disp.push_str(&temp[..i]);
                        disp.push_str(&temp[i+1..]);
                        disp
                    },
                    None => stand.replace(", none", ""),
                }
            }
            _ => stand,
        };
        f.write_fmt(format_args!("{}", display))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Var {
    WorkSizeX,
    WorkSizeY,
    WorkSizeZ,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    NumWorkgroupsX,
    NumWorkgroupsY,
    NumWorkgroupsZ,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    StdVar(String),
    TypedVar(Box<Var>, Object),
    IndexedVar(Box<Var>, Box<Var>),
}

impl Var {
    pub fn index(&self, i: Var) -> Var {
        match self {
            Var::TypedVar(s, _) => Var::IndexedVar(s.clone(), Box::new(i)),
            Var::StdVar(s) => Var::IndexedVar(Box::new(Var::StdVar(s.clone())), Box::new(i)),
            _ => panic!("Non indexable variable"),
        }
    }

    pub fn add(&self, o: Var) -> Var {
        Var::StdVar(format!("{} + {}", self, o))
    }

    pub fn multiply(&self, o: Var) -> Var {
        Var::StdVar(format!("{} * {}", self, o))
    }

    pub fn divide(&self, o: Var) -> Var {
        Var::StdVar(format!("{} / {}", self, o))
    }

    pub fn sub(&self, o: Var) -> Var {
        Var::StdVar(format!("{} - {}", self, o))
    }

    pub fn modulo(&self, o: Var) -> Var {
        Var::StdVar(format!("{} % {}", self, o))
    }

    pub fn compare(&self, o: Var, c: Comparison) -> Var {
        Var::StdVar(format!("{self} {c} {o}"))
    }
}

impl Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Var::WorkSizeX => f.write_fmt(format_args!("workgroup_size.x")),
            Var::WorkSizeY => f.write_fmt(format_args!("workgroup_size.y")),
            Var::WorkSizeZ => f.write_fmt(format_args!("workgroup_size.z")),
            Var::WorkgroupIdX => f.write_fmt(format_args!("wid.x")),
            Var::WorkgroupIdY => f.write_fmt(format_args!("wid.y")),
            Var::WorkgroupIdZ => f.write_fmt(format_args!("wid.z")),
            Var::NumWorkgroupsX => f.write_fmt(format_args!("num_wgs.x")),
            Var::NumWorkgroupsY => f.write_fmt(format_args!("num_wgs.y")),
            Var::NumWorkgroupsZ => f.write_fmt(format_args!("num_wgs.z")),
            Var::LocalInvocationIdX => f.write_fmt(format_args!("lid.x")),
            Var::LocalInvocationIdY => f.write_fmt(format_args!("lid.y")),
            Var::LocalInvocationIdZ => f.write_fmt(format_args!("lid.z")),
            Var::TypedVar(s, o) => f.write_fmt(format_args!("{}: {}", s, o)),
            Var::StdVar(s) => f.write_fmt(format_args!("{}", s)),
            Var::IndexedVar(s, i) => f.write_fmt(format_args!("{}[{}]", s, i)),
        }?;
        Ok(())
    }
}

pub struct Binding {
    vis: Visability,
    gid: u32,
    bid: u32,
    acc: Access,
    var: Var,
}

impl Binding {
    fn from(binds: Vec<(Var, bool)>) -> Vec<Self> {
        binds
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let (var, write) = r;
                let vis = Visability::Storage;
                let gid = 0_u32;
                let bid = i as u32;
                let acc = match write {
                    true => Access::ReadWrite,
                    false => Access::Read,
                };
                Self { vis, gid, bid, acc, var }
            })
            .collect::<Vec<Self>>()
    }
}    

impl Display for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("@group({})\n", self.gid))?;
        f.write_fmt(format_args!("@binding({})\n", self.bid))?;
        f.write_fmt(format_args!("var<{}, {}> {};\n", self.vis, self.acc, self.var))?;
        Ok(())
    }
}

pub enum BuiltIns {
    LocalInvocationId,
    WorkgroupId,
    NumWorkgroups,
}

impl BuiltIns {
    fn all() -> Vec<Self> {
        vec![BuiltIns::LocalInvocationId, BuiltIns::WorkgroupId, BuiltIns::NumWorkgroups]
    }
}

impl Display for BuiltIns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (call, name) = match *self {
            Self::LocalInvocationId => (String::from("local_invocation_id"), String::from("lid")),
            Self::NumWorkgroups => (String::from("workgroup_id"), String::from("wid")),
            Self::WorkgroupId => (String::from("num_workgroups"), String::from("num_wgs")),
        };
        let v = Var::TypedVar(Box::new(Var::StdVar(name)), Object::Vec3(Type::U32));
        f.write_fmt(format_args!("@builtin({}) {},\n", call, v ))?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum Access {
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

#[derive(Debug)]
pub enum Visability {
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

pub struct WorkgroupSize {
    x: u32,
    y: u32,
    z: u32,
}

impl WorkgroupSize {
    fn new(x: u32, y: u32, z:u32) -> Result<Self, WgslError >{
        let max_workgroup_size = 1024; //256;
        if x*y*z > max_workgroup_size {
            return Err(WgslError::WorkgroupSizeError(format!("Error creating workgroup. Workgroup size can't be larger than {}", max_workgroup_size)));
        }
        let max_size_z = 64;
        if z > max_size_z {
            return Err(WgslError::WorkgroupSizeError(format!("Error creating workgroup. Workgroup size z can't be larger than {}", max_size_z)));
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

pub enum WgslError {
    WorkgroupSizeError(String),
    BindingError,
}

impl std::fmt::Display for WgslError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            WgslError::WorkgroupSizeError(msg) => write!(f, "{}", msg),
            WgslError::BindingError => write!(f, "Binding error. Writers can't be larger than the number of binds"),
            //WgpuContextError::RuntimeError(err) => write!(f, "Runtime error: {}", err),
        }
    }
}

impl std::fmt::Debug for WgslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for WgslError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            //WgslError::AsynchronouslyRecievedError(ref err) => Some(err),
            _ => None,
        }
    }
}