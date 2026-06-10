use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
use std::sync::Arc;

use crate::error::GpuError;

/// Wrapper seguro para a biblioteca cuBLAS (CUDA Basic Linear Algebra Subprograms),
/// encapsulando a handle usada para executar operações de Álgebra Linear otimizadas na GPU.
pub struct BlasHandle {
    pub inner: CudaBlas,
}

impl BlasHandle {
    /// Inicializa um novo contexto do cuBLAS e o vincula ao dispositivo CUDA fornecido.
    pub fn new(dev: Arc<CudaDevice>) -> Result<Self, GpuError> {
        let inner = CudaBlas::new(dev)?;
        Ok(BlasHandle { inner })
    }

    /// Multiplicação Geral de Matrizes (GEMM): `C = α * A * B + β * C`.
    /// Como C/C++/Rust usam Row-Major e cuBLAS usa Column-Major por padrão, os argumentos
    /// `A` e `B` são trocados na chamada interna para compensar essa diferença sem alocar nova memória.
    pub fn gemm<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        a: &A,
        b: &B,
        beta: f32,
        c: &mut C,
    ) -> Result<(), GpuError> {
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha,
            lda: n as i32,
            ldb: k as i32,
            beta,
            ldc: n as i32,
        };
        unsafe {
            self.inner.gemm(cfg, b, a, c)?;
        }
        Ok(())
    }

    /// Multiplicação Geral de Matrizes com matriz A transposta: `C = α * A^T * B + β * C`.
    /// Útil para o cálculo do backpropagation (propagação do erro para trás nas camadas).
    pub fn gemm_ta<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        m: usize,
        k: usize,
        n: usize,
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

    /// Multiplicação Geral de Matrizes com matriz B transposta: `C = α * A * B^T + β * C`.
    /// Útil para o cálculo dos gradientes dos pesos durante o backpropagation.
    #[allow(dead_code)]
    pub fn gemm_tb<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        m: usize,
        k: usize,
        n: usize,
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
