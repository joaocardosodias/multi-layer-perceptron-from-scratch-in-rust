use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use std::sync::Arc;

use crate::error::GpuError;

pub struct BlasHandle {
    pub inner: CudaBlas,
}

impl BlasHandle {
    pub fn new(dev: Arc<CudaDevice>) -> Result<Self, GpuError> {
        let inner = CudaBlas::new(dev)?;
        Ok(BlasHandle { inner })
    }

    pub fn gemm<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        m: usize, k: usize, n: usize,
        alpha: f32,
        a: &A,
        b: &B,
        beta: f32,
        c: &mut C,
    ) -> Result<(), GpuError> {
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: k as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,
        };
        unsafe {
            self.inner.gemm(cfg, b, a, c)?;
        }
        Ok(())
    }

    pub fn gemm_ta<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        m: usize, k: usize, n: usize,
        alpha: f32,
        a: &A,
        b: &B,
        beta: f32,
        c: &mut C,
    ) -> Result<(), GpuError> {
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: n as i32,
            ldb: m as i32,
            beta,
            ldc: n as i32,
        };
        unsafe {
            self.inner.gemm(cfg, b, a, c)?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn gemm_tb<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        m: usize, k: usize, n: usize,
        alpha: f32,
        a: &A,
        b: &B,
        beta: f32,
        c: &mut C,
    ) -> Result<(), GpuError> {
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: k as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,
        };
        unsafe {
            self.inner.gemm(cfg, b, a, c)?;
        }
        Ok(())
    }
}
