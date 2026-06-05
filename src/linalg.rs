use matrixmultiply::sgemm;

#[inline]
pub unsafe fn gemm(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: *const f32, rsa: isize, csa: isize,
    b: *const f32, rsb: isize, csb: isize,
    beta: f32,
    c: *mut f32, rsc: isize, csc: isize,
) {
    unsafe {
        sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
    }
}

#[inline]
pub fn mat_vec_mul(matrix: &[f32], rows: usize, cols: usize, vector: &[f32], out: &mut [f32]) {
    unsafe {
        for r in 0..rows {
            let offset = r * cols;
            let mut sum = 0.0;
            for c in 0..cols {
                sum += *matrix.get_unchecked(offset + c) * *vector.get_unchecked(c);
            }
            *out.get_unchecked_mut(r) = sum;
        }
    }
}

#[inline]
pub fn add_outer_product(a: &[f32], b: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    unsafe {
        for r in 0..rows {
            let offset = r * cols;
            let val_a = *a.get_unchecked(r);
            for c in 0..cols {
                let val_b = *b.get_unchecked(c);
                *out.get_unchecked_mut(offset + c) += val_a * val_b;
            }
        }
    }
}

#[inline]
pub fn transpose_mul_vec(matrix: &[f32], rows: usize, cols: usize, vector: &[f32], out: &mut [f32]) {
    unsafe {
        for c in 0..cols {
            let mut sum = 0.0;
            for r in 0..rows {
                sum += *matrix.get_unchecked(r * cols + c) * *vector.get_unchecked(r);
            }
            *out.get_unchecked_mut(c) = sum;
        }
    }
}
