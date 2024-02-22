pub enum WgpuContextError {
    NoAdapterError,
    BindGroupError,
    ExceededBufferSizeError,
    RuntimeError(std::io::Error),
    NoDeviceError(wgpu::RequestDeviceError),
    AsynchronouslyRecievedError(flume::RecvError),
}

#[derive(Debug)]
struct ExceededBufferSizeError;

#[derive(Debug)]
struct NoDeviceError;

#[derive(Debug)]
struct RuntimeError;

#[derive(Debug)]
struct NoAdapterError;

#[derive(Debug)]
struct AsynchronouslyRecievedError;

#[derive(Debug)]
struct BindGroupError;

impl From<wgpu::RequestDeviceError> for WgpuContextError {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        WgpuContextError::NoDeviceError(err)
    }
}

impl From<std::io::Error> for WgpuContextError {
    fn from(err: std::io::Error) -> Self {
        WgpuContextError::RuntimeError(err)
    }
}

impl From<flume::RecvError> for WgpuContextError {
    fn from(err: flume::RecvError) -> Self {
        WgpuContextError::AsynchronouslyRecievedError(err)
    }
}

impl std::fmt::Display for WgpuContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            WgpuContextError::NoAdapterError => write!(f, "No adapter found"),
            WgpuContextError::ExceededBufferSizeError => write!(f, "Buffer size exceeded"),
            WgpuContextError::BindGroupError => write!(f, "Bind group error, to many writers"),
            WgpuContextError::RuntimeError(err) => write!(f, "Runtime error: {}", err),
            WgpuContextError::NoDeviceError(err) => write!(f, "No device found: {}", err),
            WgpuContextError::AsynchronouslyRecievedError(err) => {
                write!(f, "Asynchronously recieved error: {}", err)
            }
        }
    }
}

impl std::fmt::Debug for WgpuContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for WgpuContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            WgpuContextError::AsynchronouslyRecievedError(ref err) => Some(err),
            WgpuContextError::NoDeviceError(ref err) => Some(err),
            WgpuContextError::RuntimeError(ref err) => Some(err),
            _ => None,
        }
    }
}
