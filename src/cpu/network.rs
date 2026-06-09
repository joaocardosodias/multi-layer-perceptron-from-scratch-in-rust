use crate::linalg::gemm;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

pub struct MLP {
    pub weights: Vec<f32>,
    pub w_offsets: Vec<usize>,
    pub biases: Vec<f32>,
    pub b_offsets: Vec<usize>,
    pub dims: Vec<(usize, usize)>,
}

pub struct BatchCache {
    pub pre_activations: Vec<f32>,
    pub p_offsets: Vec<usize>,
    pub activations: Vec<f32>,
    pub a_offsets: Vec<usize>,
    pub deltas: Vec<f32>,
    pub d_offsets: Vec<usize>,
    pub dropout_masks: Vec<f32>,
    pub dm_offsets: Vec<usize>,
}

impl BatchCache {
    pub fn new(dims: &[(usize, usize)], batch_size: usize) -> Self {
        let mut pre_activations = Vec::new();
        let mut p_offsets = Vec::new();
        let mut activations = Vec::new();
        let mut a_offsets = Vec::new();
        let mut deltas = Vec::new();
        let mut d_offsets = Vec::new();
        let mut dropout_masks = Vec::new();
        let mut dm_offsets = Vec::new();

        a_offsets.push(0);
        activations.resize(batch_size * dims[0].1, 0.0);

        for &(r, _) in dims {
            p_offsets.push(pre_activations.len());
            pre_activations.resize(pre_activations.len() + batch_size * r, 0.0);

            a_offsets.push(activations.len());
            activations.resize(activations.len() + batch_size * r, 0.0);

            d_offsets.push(deltas.len());
            deltas.resize(deltas.len() + batch_size * r, 0.0);

            dm_offsets.push(dropout_masks.len());
            dropout_masks.resize(dropout_masks.len() + batch_size * r, 1.0);
        }
        p_offsets.push(pre_activations.len());
        a_offsets.push(activations.len());
        d_offsets.push(deltas.len());
        dm_offsets.push(dropout_masks.len());

        BatchCache {
            pre_activations,
            p_offsets,
            activations,
            a_offsets,
            deltas,
            d_offsets,
            dropout_masks,
            dm_offsets,
        }
    }
}

pub struct Gradients {
    pub dw: Vec<f32>,
    pub w_offsets: Vec<usize>,
    pub db: Vec<f32>,
    pub b_offsets: Vec<usize>,
}

impl Gradients {
    pub fn new(mlp: &MLP) -> Self {
        Gradients {
            dw: vec![0.0; mlp.weights.len()],
            w_offsets: mlp.w_offsets.clone(),
            db: vec![0.0; mlp.biases.len()],
            b_offsets: mlp.b_offsets.clone(),
        }
    }

    pub fn zero(&mut self) {
        self.dw.fill(0.0);
        self.db.fill(0.0);
    }

    pub fn accumulate(&mut self, src: &Gradients) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let wl = self.dw.len();
            let sw = wl - (wl % 8);
            for i in (0..sw).step_by(8) {
                let a = _mm256_loadu_ps(self.dw.as_ptr().add(i));
                let b = _mm256_loadu_ps(src.dw.as_ptr().add(i));
                _mm256_storeu_ps(self.dw.as_mut_ptr().add(i), _mm256_add_ps(a, b));
            }
            for i in sw..wl {
                self.dw[i] += src.dw[i];
            }
            let bl = self.db.len();
            let sb = bl - (bl % 8);
            for i in (0..sb).step_by(8) {
                let a = _mm256_loadu_ps(self.db.as_ptr().add(i));
                let b = _mm256_loadu_ps(src.db.as_ptr().add(i));
                _mm256_storeu_ps(self.db.as_mut_ptr().add(i), _mm256_add_ps(a, b));
            }
            for i in sb..bl {
                self.db[i] += src.db[i];
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..self.dw.len() {
                self.dw[i] += src.dw[i];
            }
            for i in 0..self.db.len() {
                self.db[i] += src.db[i];
            }
        }
    }
}

impl MLP {
    pub fn new(architecture: &[usize]) -> Self {
        let mut weights = Vec::new();
        let mut w_offsets = Vec::new();
        let mut biases = Vec::new();
        let mut b_offsets = Vec::new();
        let mut dims = Vec::new();

        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..(architecture.len() - 1) {
            let n_in = architecture[i];
            let n_out = architecture[i + 1];
            let std_dev = (2.0 / n_in as f32).sqrt();
            let normal = Normal::new(0.0, std_dev).unwrap();

            w_offsets.push(weights.len());
            for _ in 0..(n_out * n_in) {
                weights.push(normal.sample(&mut rng) as f32);
            }

            b_offsets.push(biases.len());
            for _ in 0..n_out {
                biases.push(0.0);
            }

            dims.push((n_out, n_in));
        }
        w_offsets.push(weights.len());
        b_offsets.push(biases.len());

        MLP {
            weights,
            w_offsets,
            biases,
            b_offsets,
            dims,
        }
    }

    pub fn forward_batch(
        &self,
        input_batch: &[f32],
        cache: &mut BatchCache,
        bs: usize,
        is_training: bool,
        rng: &mut StdRng,
    ) {
        use rand::Rng;

        let a_start = cache.a_offsets[0];
        let a_end = cache.a_offsets[1];
        cache.activations[a_start..a_end].copy_from_slice(input_batch);

        for i in 0..self.dims.len() {
            let (rows, cols) = self.dims[i];
            let w_ptr = self.weights[self.w_offsets[i]..].as_ptr();
            let b_ptr = self.biases[self.b_offsets[i]..].as_ptr();
            let inp_ptr = cache.activations[cache.a_offsets[i]..].as_ptr();
            let z_ptr = cache.pre_activations[cache.p_offsets[i]..].as_mut_ptr();
            let a_ptr = cache.activations[cache.a_offsets[i + 1]..].as_mut_ptr();

            unsafe {
                gemm(
                    bs,
                    cols,
                    rows,
                    1.0,
                    inp_ptr,
                    cols as isize,
                    1,
                    w_ptr,
                    1,
                    cols as isize,
                    0.0,
                    z_ptr,
                    rows as isize,
                    1,
                );
            }

            #[cfg(target_arch = "x86_64")]
            unsafe {
                let simd_len = rows - (rows % 8);
                for s in 0..bs {
                    let offset = s * rows;
                    for r in (0..simd_len).step_by(8) {
                        let z_vec = std::arch::x86_64::_mm256_loadu_ps(z_ptr.add(offset + r));
                        let b_vec = std::arch::x86_64::_mm256_loadu_ps(b_ptr.add(r));
                        let result = std::arch::x86_64::_mm256_add_ps(z_vec, b_vec);
                        std::arch::x86_64::_mm256_storeu_ps(z_ptr.add(offset + r), result);
                    }
                    for r in simd_len..rows {
                        *z_ptr.add(offset + r) += *b_ptr.add(r);
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for s in 0..bs {
                    let offset = s * rows;
                    for r in 0..rows {
                        unsafe {
                            *z_ptr.add(offset + r) += *b_ptr.add(r);
                        }
                    }
                }
            }

            if i == self.dims.len() - 1 {
                for s in 0..bs {
                    let offset = s * rows;
                    let mut max_val = f32::NEG_INFINITY;
                    for r in 0..rows {
                        let val = unsafe { *z_ptr.add(offset + r) };
                        if val > max_val {
                            max_val = val;
                        }
                    }

                    let mut sum_exp = 0.0;
                    for r in 0..rows {
                        let e = (unsafe { *z_ptr.add(offset + r) } - max_val).exp();
                        unsafe {
                            *a_ptr.add(offset + r) = e;
                        }
                        sum_exp += e;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for r in 0..rows {
                        unsafe {
                            *a_ptr.add(offset + r) *= inv_sum;
                        }
                    }
                }
            } else {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    let zero = std::arch::x86_64::_mm256_set1_ps(0.0);
                    let simd_len = rows - (rows % 8);
                    for s in 0..bs {
                        let offset = s * rows;
                        for r in (0..simd_len).step_by(8) {
                            let z_vec = std::arch::x86_64::_mm256_loadu_ps(z_ptr.add(offset + r));
                            let result = std::arch::x86_64::_mm256_max_ps(z_vec, zero);
                            std::arch::x86_64::_mm256_storeu_ps(a_ptr.add(offset + r), result);
                        }
                        for r in simd_len..rows {
                            let val = *z_ptr.add(offset + r);
                            *a_ptr.add(offset + r) = if val > 0.0 { val } else { 0.0 };
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for s in 0..bs {
                        let offset = s * rows;
                        for r in 0..rows {
                            unsafe {
                                let val = *z_ptr.add(offset + r);
                                *a_ptr.add(offset + r) = if val > 0.0 { val } else { 0.0 };
                            }
                        }
                    }
                }

                if is_training {
                    let p_keep = 0.9f32;
                    let scale = 1.0 / p_keep;
                    let start = cache.dm_offsets[i];
                    let end = cache.dm_offsets[i + 1];
                    for k in start..end {
                        cache.dropout_masks[k] = if rng.gen_range(0.0..1.0) < p_keep {
                            scale
                        } else {
                            0.0
                        };
                    }

                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        use std::arch::x86_64::*;
                        let total_len = bs * rows;
                        let simd_len = total_len - (total_len % 8);
                        let m_ptr = cache.dropout_masks[start..].as_ptr();
                        for r in (0..simd_len).step_by(8) {
                            let a_vec = _mm256_loadu_ps(a_ptr.add(r));
                            let m_vec = _mm256_loadu_ps(m_ptr.add(r));
                            let res = _mm256_mul_ps(a_vec, m_vec);
                            _mm256_storeu_ps(a_ptr.add(r), res);
                        }
                        for r in simd_len..total_len {
                            *a_ptr.add(r) *= *m_ptr.add(r);
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        let start_a = cache.a_offsets[i + 1];
                        for r in 0..(bs * rows) {
                            cache.activations[start_a + r] *= cache.dropout_masks[start + r];
                        }
                    }
                }
            }
        }
    }

    pub fn backward_batch(
        &self,
        cache: &mut BatchCache,
        targets: &[usize],
        acc_grads: &mut Gradients,
        bs: usize,
    ) {
        let num_layers = self.dims.len();
        let inv_bs = 1.0 / bs as f32;

        let out_dim = self.dims[num_layers - 1].0;
        let delta_out_ptr = cache.deltas[cache.d_offsets[num_layers - 1]..].as_mut_ptr();
        let probs_ptr = cache.activations[cache.a_offsets[num_layers]..].as_ptr();

        for s in 0..bs {
            let offset = s * out_dim;
            let target = targets[s];
            for r in 0..out_dim {
                unsafe {
                    *delta_out_ptr.add(offset + r) = if r == target {
                        *probs_ptr.add(offset + r) - 1.0
                    } else {
                        *probs_ptr.add(offset + r)
                    };
                }
            }
        }

        let (rows, cols) = self.dims[num_layers - 1];
        unsafe {
            gemm(
                rows,
                bs,
                cols,
                inv_bs,
                delta_out_ptr,
                1,
                out_dim as isize,
                cache.activations[cache.a_offsets[num_layers - 1]..].as_ptr(),
                cols as isize,
                1,
                1.0,
                acc_grads.dw[acc_grads.w_offsets[num_layers - 1]..].as_mut_ptr(),
                cols as isize,
                1,
            );
        }

        let db_last_ptr = acc_grads.db[acc_grads.b_offsets[num_layers - 1]..].as_mut_ptr();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let inv_bs_vec = std::arch::x86_64::_mm256_set1_ps(inv_bs);
            let simd_len = out_dim - (out_dim % 8);
            for r in (0..simd_len).step_by(8) {
                let mut acc = std::arch::x86_64::_mm256_setzero_ps();
                for s in 0..bs {
                    let val =
                        std::arch::x86_64::_mm256_loadu_ps(delta_out_ptr.add(s * out_dim + r));
                    acc = std::arch::x86_64::_mm256_add_ps(acc, val);
                }
                let existing = std::arch::x86_64::_mm256_loadu_ps(db_last_ptr.add(r));
                let result = std::arch::x86_64::_mm256_fmadd_ps(acc, inv_bs_vec, existing);
                std::arch::x86_64::_mm256_storeu_ps(db_last_ptr.add(r), result);
            }
            for r in simd_len..out_dim {
                let mut sum = 0.0;
                for s in 0..bs {
                    sum += *delta_out_ptr.add(s * out_dim + r);
                }
                *db_last_ptr.add(r) += sum * inv_bs;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for r in 0..out_dim {
                let mut sum = 0.0;
                for s in 0..bs {
                    unsafe {
                        sum += *delta_out_ptr.add(s * out_dim + r);
                    }
                }
                unsafe {
                    *db_last_ptr.add(r) += sum * inv_bs;
                }
            }
        }

        for l in (0..num_layers - 1).rev() {
            let (rows_next, _cols_next) = self.dims[l + 1];
            let next_dim = self.dims[l].0;
            let delta_next_ptr = cache.deltas[cache.d_offsets[l + 1]..].as_ptr();
            let delta_curr_ptr = cache.deltas[cache.d_offsets[l]..].as_mut_ptr();
            let w_next_ptr = self.weights[self.w_offsets[l + 1]..].as_ptr();

            unsafe {
                gemm(
                    bs,
                    rows_next,
                    next_dim,
                    1.0,
                    delta_next_ptr,
                    rows_next as isize,
                    1,
                    w_next_ptr,
                    next_dim as isize,
                    1,
                    0.0,
                    delta_curr_ptr,
                    next_dim as isize,
                    1,
                );
            }

            let z_prev_ptr = cache.pre_activations[cache.p_offsets[l]..].as_ptr();
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let zero = std::arch::x86_64::_mm256_setzero_ps();
                let simd_len = next_dim - (next_dim % 8);
                for s in 0..bs {
                    let offset = s * next_dim;
                    for r in (0..simd_len).step_by(8) {
                        let z = std::arch::x86_64::_mm256_loadu_ps(z_prev_ptr.add(offset + r));
                        let mask = std::arch::x86_64::_mm256_cmp_ps(z, zero, 0x1E);
                        let delta =
                            std::arch::x86_64::_mm256_loadu_ps(delta_curr_ptr.add(offset + r));
                        let result = std::arch::x86_64::_mm256_and_ps(delta, mask);
                        std::arch::x86_64::_mm256_storeu_ps(delta_curr_ptr.add(offset + r), result);
                    }
                    for r in simd_len..next_dim {
                        if *z_prev_ptr.add(offset + r) <= 0.0 {
                            *delta_curr_ptr.add(offset + r) = 0.0;
                        }
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for s in 0..bs {
                    let offset = s * next_dim;
                    for r in 0..next_dim {
                        unsafe {
                            if *z_prev_ptr.add(offset + r) <= 0.0 {
                                *delta_curr_ptr.add(offset + r) = 0.0;
                            }
                        }
                    }
                }
            }

            // Apply dropout mask to delta_curr_ptr
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::*;
                let total_len = bs * next_dim;
                let simd_len = total_len - (total_len % 8);
                let dm_ptr = cache.dropout_masks[cache.dm_offsets[l]..].as_ptr();
                for r in (0..simd_len).step_by(8) {
                    let d_vec = _mm256_loadu_ps(delta_curr_ptr.add(r));
                    let m_vec = _mm256_loadu_ps(dm_ptr.add(r));
                    let res = _mm256_mul_ps(d_vec, m_vec);
                    _mm256_storeu_ps(delta_curr_ptr.add(r), res);
                }
                for r in simd_len..total_len {
                    *delta_curr_ptr.add(r) *= *dm_ptr.add(r);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let start_dm = cache.dm_offsets[l];
                for r in 0..(bs * next_dim) {
                    unsafe {
                        *delta_curr_ptr.add(r) *= cache.dropout_masks[start_dm + r];
                    }
                }
            }

            let (r, c) = self.dims[l];
            let a_prev_ptr = cache.activations[cache.a_offsets[l]..].as_ptr();
            let dw_ptr = acc_grads.dw[acc_grads.w_offsets[l]..].as_mut_ptr();

            unsafe {
                gemm(
                    r,
                    bs,
                    c,
                    inv_bs,
                    delta_curr_ptr,
                    1,
                    r as isize,
                    a_prev_ptr,
                    c as isize,
                    1,
                    1.0,
                    dw_ptr,
                    c as isize,
                    1,
                );
            }

            let db_l_ptr = acc_grads.db[acc_grads.b_offsets[l]..].as_mut_ptr();
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let inv_bs_vec = std::arch::x86_64::_mm256_set1_ps(inv_bs);
                let simd_len = r - (r % 8);
                for rr in (0..simd_len).step_by(8) {
                    let mut acc = std::arch::x86_64::_mm256_setzero_ps();
                    for s in 0..bs {
                        let val = std::arch::x86_64::_mm256_loadu_ps(
                            delta_curr_ptr.add(s * next_dim + rr),
                        );
                        acc = std::arch::x86_64::_mm256_add_ps(acc, val);
                    }
                    let existing = std::arch::x86_64::_mm256_loadu_ps(db_l_ptr.add(rr));
                    let result = std::arch::x86_64::_mm256_fmadd_ps(acc, inv_bs_vec, existing);
                    std::arch::x86_64::_mm256_storeu_ps(db_l_ptr.add(rr), result);
                }
                for rr in simd_len..r {
                    let mut sum = 0.0;
                    for s in 0..bs {
                        sum += *delta_curr_ptr.add(s * next_dim + rr);
                    }
                    *db_l_ptr.add(rr) += sum * inv_bs;
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for rr in 0..r {
                    let mut sum = 0.0;
                    for s in 0..bs {
                        unsafe {
                            sum += *delta_curr_ptr.add(s * next_dim + rr);
                        }
                    }
                    unsafe {
                        *db_l_ptr.add(rr) += sum * inv_bs;
                    }
                }
            }
        }
    }
}
