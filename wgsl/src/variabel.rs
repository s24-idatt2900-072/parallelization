use crate::utils::*;
use std::fmt::Display;

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
    TypedVar(Box<Var>, ReturnType),
}

impl Var {
    pub fn index(&self, i: &Var) -> Var {
        match self {
            Var::StdVar(s) => Var::StdVar(format!("{}[{}]", s.clone(), i.clone())),
            _ => panic!("Non indexable variable"),
        }
    }

    pub fn array_length(&self) -> Var {
        Var::StdVar(format!("arrayLength(&{})", self))
    }

    pub fn parenthesis(&self) -> Var {
        Var::StdVar(format!("({})", self))
    }

    pub fn sqrt(&self) -> Var {
        Var::StdVar(format!("sqrt({})", self))
    }

    pub fn add(&self, o: &Var) -> Var {
        Var::StdVar(format!("{} + {}", self, o))
    }

    pub fn multiply(&self, o: &Var) -> Var {
        Var::StdVar(format!("{} * {}", self, o))
    }

    pub fn divide(&self, o: &Var) -> Var {
        Var::StdVar(format!("{} / {}", self, o))
    }

    pub fn sub(&self, o: &Var) -> Var {
        Var::StdVar(format!("{} - {}", self, o))
    }

    pub fn modulo(&self, o: &Var) -> Var {
        Var::StdVar(format!("{} % {}", self, o))
    }

    pub fn compare(&self, o: &Var, c: Comparison) -> Var {
        Var::StdVar(format!("{self} {c} {o}"))
    }

    pub fn from_num<T: std::string::ToString>(num: T) -> Var {
        let num_str = match std::any::type_name::<T>() {
            t if t == std::any::type_name::<u16>() => format!("{}u", num.to_string()),
            t if t == std::any::type_name::<u32>() => format!("{}u", num.to_string()),
            t if t == std::any::type_name::<u64>() => format!("{}u", num.to_string()),
            t if t == std::any::type_name::<u128>() => format!("{}u", num.to_string()),
            t if t == std::any::type_name::<i16>() => format!("{}i", num.to_string()),
            t if t == std::any::type_name::<i32>() => format!("{}i", num.to_string()),
            t if t == std::any::type_name::<i64>() => format!("{}i", num.to_string()),
            t if t == std::any::type_name::<i128>() => format!("{}i", num.to_string()),
            t if t == std::any::type_name::<f32>() => {
                let num_str = num.to_string();
                if num_str.contains('.') {
                    num_str
                } else {
                    format!("{num_str}.0")
                }
            }
            t if t == std::any::type_name::<f64>() => {
                let num_str = num.to_string();
                if num_str.contains('.') {
                    num_str
                } else {
                    format!("{num_str}.0")
                }
            }
            _ => num.to_string(),
        };
        Var::StdVar(num_str)
    }
}

impl From<&str> for Var {
    fn from(value: &str) -> Self {
        Var::StdVar(value.to_string())
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
            Var::TypedVar(s, t) => f.write_fmt(format_args!("{}: {}", s, t)),
            Var::StdVar(s) => f.write_fmt(format_args!("{}", s)),
        }?;
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
        let stand = format!("{:?}", self)
            .to_lowercase()
            .replace('(', "<")
            .replace(')', ">");
        let display = match self {
            Object::Array(_, s) => match s {
                Some(_) => {
                    let temp = stand.replace("some<", "");
                    let mut disp = String::new();
                    let i = temp.find('>').unwrap();
                    disp.push_str(&temp[..i]);
                    disp.push_str(&temp[i + 1..]);
                    disp
                }
                None => stand.replace(", none", ""),
            },
            _ => stand,
        };
        f.write_fmt(format_args!("{}", display))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ReturnType {
    T(Type),
    Obj(Object),
}

impl Display for ReturnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReturnType::T(inner) => f.write_fmt(format_args!("{inner}"))?,
            ReturnType::Obj(inner) => f.write_fmt(format_args!("{inner}"))?,
        };
        Ok(())
    }
}
