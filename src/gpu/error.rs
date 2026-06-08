#[derive(Debug)]
pub enum GpuError {
    Driver(cudarc::driver::DriverError),
    Cublas(cudarc::cublas::result::CublasError),
    Compile(cudarc::nvrtc::CompileError),
    Other(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Driver(e) => write!(f, "Driver error: {}", e),
            GpuError::Cublas(e) => write!(f, "cuBLAS error: {:?}", e),
            GpuError::Compile(e) => write!(f, "Compile error: {:?}", e),
            GpuError::Other(s) => write!(f, "GPU error: {}", s),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<cudarc::driver::DriverError> for GpuError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        GpuError::Driver(e)
    }
}

impl From<cudarc::cublas::result::CublasError> for GpuError {
    fn from(e: cudarc::cublas::result::CublasError) -> Self {
        GpuError::Cublas(e)
    }
}

impl From<cudarc::nvrtc::CompileError> for GpuError {
    fn from(e: cudarc::nvrtc::CompileError) -> Self {
        GpuError::Compile(e)
    }
}
