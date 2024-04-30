pub enum WgslError {
    WorkgroupSizeError(String),
    BindingError,
}

impl std::fmt::Display for WgslError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            WgslError::WorkgroupSizeError(msg) => write!(f, "{}", msg),
            WgslError::BindingError => write!(
                f,
                "Binding error. Writers can't be larger than the number of binds"
            ),
        }
    }
}

impl std::fmt::Debug for WgslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
