use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use std::os::raw::c_int;

#[inline]
/// Realiza a multiplicação de matrizes de forma eficiente (GEMM - General Matrix Multiply) usando CBLAS (OpenBLAS/Accelerate).
/// Computa: `C = alpha * A * B + beta * C`.
/// `rsa/csa` (row/column strides) definem se a matriz deve ser considerada transposta ou não na memória.
pub unsafe fn gemm(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: *const f32,
    rsa: isize,
    csa: isize,
    b: *const f32,
    rsb: isize,
    csb: isize,
    beta: f32,
    c: *mut f32,
    rsc: isize,
    _csc: isize,
) {
    let layout = CBLAS_LAYOUT::CblasRowMajor;

    let trans_a = if csa == 1 {
        CBLAS_TRANSPOSE::CblasNoTrans
    } else {
        CBLAS_TRANSPOSE::CblasTrans
    };
    let lda: c_int = if csa == 1 { rsa as c_int } else { csa as c_int };

    let trans_b = if csb == 1 {
        CBLAS_TRANSPOSE::CblasNoTrans
    } else {
        CBLAS_TRANSPOSE::CblasTrans
    };
    let ldb: c_int = if csb == 1 { rsb as c_int } else { csb as c_int };

    let ldc: c_int = rsc as c_int;

    unsafe {
        cblas_sys::cblas_sgemm(
            layout, trans_a, trans_b, m as c_int, n as c_int, k as c_int, alpha, a, lda, b, ldb,
            beta, c, ldc,
        );
    }
}
