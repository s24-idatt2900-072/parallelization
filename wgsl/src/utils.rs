use crate::shader::Visability;
use crate::variabel::*;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct Body {
    ins: Vec<Line>,
}

impl Body {
    pub fn new() -> Self {
        Self { ins: Vec::new() }
    }

    pub fn add_line(&mut self, i: Line) -> &mut Self {
        self.ins.push(i);
        self
    }

    pub fn finish(&mut self) -> Self {
        self.clone()
    }
}

impl Display for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ins.iter().for_each(|i| {
            f.write_fmt(format_args!("{i}\n")).unwrap();
        });
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Line {
    Ins(Instruction),
    Flw(FlowControl),
}

impl From<FlowControl> for Line {
    fn from(value: FlowControl) -> Self {
        Line::Flw(value)
    }
}

impl From<Instruction> for Line {
    fn from(value: Instruction) -> Self {
        Line::Ins(value)
    }
}

impl Display for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Line::Flw(inner) => f.write_fmt(format_args!("{inner};"))?,
            Line::Ins(inner) => f.write_fmt(format_args!("{inner};"))?,
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
            FlowControl::Return(o) => match o {
                Some(v) => format!("return {}", v),
                None => String::from("return"),
            },
            FlowControl::Break => String::from("break"),
            FlowControl::WorkgroupBarrier => String::from("workgroupBarrier()"),
        };
        f.write_fmt(format_args!("{display}"))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    DefineConstant { vis: Visability, var: Var },
    Multiply { lhs: Var, rhs: Var, out: Var },
    Subtract { lhs: Var, rhs: Var, out: Var },
    Divide { lhs: Var, rhs: Var, out: Var },
    Modulo { lhs: Var, rhs: Var, out: Var },
    Add { lhs: Var, rhs: Var, out: Var },
    DefineMutVar { lhs: Var, rhs: Var },
    DefineVar { lhs: Var, rhs: Var },
    Set { lhs: Var, rhs: Var },
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let display = match self {
            Instruction::Subtract { lhs, rhs, out } => format!("{} = {} - {}", out, lhs, rhs),
            Instruction::Multiply { lhs, rhs, out } => format!("{} = {} * {}", out, lhs, rhs),
            Instruction::Divide { lhs, rhs, out } => format!("{} = {} / {}", out, lhs, rhs),
            Instruction::Modulo { lhs, rhs, out } => format!("{} = {} % {}", out, lhs, rhs),
            Instruction::DefineConstant { vis, var } => format!("var<{}> {}", vis, var),
            Instruction::Add { lhs, rhs, out } => format!("{} = {} + {}", out, lhs, rhs),
            Instruction::DefineMutVar { lhs, rhs } => format!("var {} = {}", lhs, rhs),
            Instruction::DefineVar { lhs, rhs } => format!("let {} = {}", lhs, rhs),
            Instruction::Set { lhs, rhs } => format!("{} = {}", lhs, rhs),
        };
        f.write_fmt(format_args!("{display}"))?;
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
