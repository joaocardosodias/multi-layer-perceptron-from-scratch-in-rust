use crate::network::{Gradients, MLP};

pub fn sgd_update(mlp: &mut MLP, grads: &Gradients, lr: f32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let lr_vec = std::arch::x86_64::_mm256_set1_ps(lr);
        
        let w_len = mlp.weights.len();
        let w_simd_len = w_len - (w_len % 8);
        for i in (0..w_simd_len).step_by(8) {
            let w_val = std::arch::x86_64::_mm256_loadu_ps(mlp.weights.as_ptr().add(i));
            let gw_val = std::arch::x86_64::_mm256_loadu_ps(grads.dw.as_ptr().add(i));
            let new_w = std::arch::x86_64::_mm256_fnmadd_ps(gw_val, lr_vec, w_val);
            std::arch::x86_64::_mm256_storeu_ps(mlp.weights.as_mut_ptr().add(i), new_w);
        }
        for i in w_simd_len..w_len {
            *mlp.weights.as_mut_ptr().add(i) -= lr * *grads.dw.as_ptr().add(i);
        }

        let b_len = mlp.biases.len();
        let b_simd_len = b_len - (b_len % 8);
        for i in (0..b_simd_len).step_by(8) {
            let b_val = std::arch::x86_64::_mm256_loadu_ps(mlp.biases.as_ptr().add(i));
            let gb_val = std::arch::x86_64::_mm256_loadu_ps(grads.db.as_ptr().add(i));
            let new_b = std::arch::x86_64::_mm256_fnmadd_ps(gb_val, lr_vec, b_val);
            std::arch::x86_64::_mm256_storeu_ps(mlp.biases.as_mut_ptr().add(i), new_b);
        }
        for i in b_simd_len..b_len {
            *mlp.biases.as_mut_ptr().add(i) -= lr * *grads.db.as_ptr().add(i);
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..mlp.weights.len() {
            mlp.weights[i] -= lr * grads.dw[i];
        }
        for i in 0..mlp.biases.len() {
            mlp.biases[i] -= lr * grads.db[i];
        }
    }
}
