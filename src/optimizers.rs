use crate::network::Gradients;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn sgd_update(
    weights: &mut Vec<Vec<f32>>,
    biases: &mut Vec<Vec<f32>>,
    gradients: &Gradients,
    learning_rate: f32,
) {
    for layer_idx in 0..weights.len() {
        let w = &mut weights[layer_idx];
        let b = &mut biases[layer_idx];
        let dw = &gradients.dw[layer_idx];
        let db = &gradients.db[layer_idx];

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let lr_vec = _mm256_set1_ps(learning_rate);

            let w_len = w.len();
            let w_ptr = w.as_mut_ptr();
            let dw_ptr = dw.as_ptr();
            for i in (0..w_len).step_by(8) {
                let w_vec = _mm256_loadu_ps(w_ptr.add(i));
                let dw_vec = _mm256_loadu_ps(dw_ptr.add(i));
                let update = _mm256_mul_ps(lr_vec, dw_vec);
                let result = _mm256_sub_ps(w_vec, update);
                _mm256_storeu_ps(w_ptr.add(i), result);
            }

            let b_len = b.len();
            let b_ptr = b.as_mut_ptr();
            let db_ptr = db.as_ptr();
            let simd_len = b_len - (b_len % 8);
            for i in (0..simd_len).step_by(8) {
                let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                let db_vec = _mm256_loadu_ps(db_ptr.add(i));
                let update = _mm256_mul_ps(lr_vec, db_vec);
                let result = _mm256_sub_ps(b_vec, update);
                _mm256_storeu_ps(b_ptr.add(i), result);
            }
            for i in simd_len..b_len {
                b[i] -= learning_rate * db[i];
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..w.len() {
                w[i] -= learning_rate * dw[i];
            }
            for i in 0..b.len() {
                b[i] -= learning_rate * db[i];
            }
        }
    }
}
