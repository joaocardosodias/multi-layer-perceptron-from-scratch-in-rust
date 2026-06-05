use crate::linalg::gemm;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

pub struct MLP {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
    pub dims: Vec<(usize, usize)>,
}

pub struct BatchCache {
    pub pre_activations: Vec<Vec<f32>>,
    pub activations: Vec<Vec<f32>>,
    pub deltas: Vec<Vec<f32>>,
    pub batch_size: usize,
}

impl BatchCache {
    pub fn new(dims: &[(usize, usize)], batch_size: usize) -> Self {
        BatchCache {
            pre_activations: dims
                .iter()
                .map(|&(r, _)| vec![0.0; batch_size * r])
                .collect(),
            activations: {
                let mut acts = Vec::with_capacity(dims.len() + 1);
                acts.push(vec![0.0; batch_size * dims[0].1]);
                for &(r, _) in dims {
                    acts.push(vec![0.0; batch_size * r]);
                }
                acts
            },
            deltas: dims
                .iter()
                .map(|&(r, _)| vec![0.0; batch_size * r])
                .collect(),
            batch_size,
        }
    }
}

pub struct Gradients {
    pub dw: Vec<Vec<f32>>,
    pub db: Vec<Vec<f32>>,
}

impl Gradients {
    pub fn new(mlp: &MLP) -> Self {
        Gradients {
            dw: mlp.weights.iter().map(|w| vec![0.0; w.len()]).collect(),
            db: mlp.biases.iter().map(|b| vec![0.0; b.len()]).collect(),
        }
    }

    pub fn zero(&mut self) {
        for w in self.dw.iter_mut() {
            w.fill(0.0);
        }
        for b in self.db.iter_mut() {
            b.fill(0.0);
        }
    }
}

impl MLP {
    pub fn new(architecture: &[usize]) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut dims = Vec::new();

        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..(architecture.len() - 1) {
            let n_in = architecture[i];
            let n_out = architecture[i + 1];
            let std_dev = (2.0 / n_in as f32).sqrt();
            let normal = Normal::new(0.0, std_dev).unwrap();

            let mut layer_w = vec![0.0; n_out * n_in];
            for w in layer_w.iter_mut() {
                *w = normal.sample(&mut rng) as f32;
            }

            weights.push(layer_w);
            biases.push(vec![0.0; n_out]);
            dims.push((n_out, n_in));
        }
        MLP {
            weights,
            biases,
            dims,
        }
    }

    pub fn forward_batch(&self, input_batch: &[f32], cache: &mut BatchCache, bs: usize) {
        cache.activations[0].copy_from_slice(input_batch);

        for i in 0..self.weights.len() {
            let (rows, cols) = self.dims[i];

            unsafe {
                gemm(
                    bs,
                    cols,
                    rows,
                    1.0,
                    cache.activations[i].as_ptr(),
                    cols as isize,
                    1,
                    self.weights[i].as_ptr(),
                    1,
                    cols as isize,
                    0.0,
                    cache.pre_activations[i].as_mut_ptr(),
                    rows as isize,
                    1,
                );
            }

            let b = &self.biases[i];
            let z = &mut cache.pre_activations[i];
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let simd_len = rows - (rows % 8);
                for s in 0..bs {
                    let offset = s * rows;
                    for r in (0..simd_len).step_by(8) {
                        let z_vec = std::arch::x86_64::_mm256_loadu_ps(z.as_mut_ptr().add(offset + r));
                        let b_vec = std::arch::x86_64::_mm256_loadu_ps(b.as_ptr().add(r));
                        let result = std::arch::x86_64::_mm256_add_ps(z_vec, b_vec);
                        std::arch::x86_64::_mm256_storeu_ps(z.as_mut_ptr().add(offset + r), result);
                    }
                    for r in simd_len..rows {
                        z[offset + r] += b[r];
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for s in 0..bs {
                    let offset = s * rows;
                    for r in 0..rows {
                        z[offset + r] += b[r];
                    }
                }
            }

            let z = &cache.pre_activations[i];
            let a = &mut cache.activations[i + 1];

            if i == self.weights.len() - 1 {
                for s in 0..bs {
                    let offset = s * rows;

                    let mut max_val = f32::NEG_INFINITY;
                    for r in 0..rows {
                        if z[offset + r] > max_val { max_val = z[offset + r]; }
                    }

                    let mut sum_exp = 0.0;
                    for r in 0..rows {
                        let e = (z[offset + r] - max_val).exp();
                        a[offset + r] = e;
                        sum_exp += e;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for r in 0..rows {
                        a[offset + r] *= inv_sum;
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
                            let z_vec = std::arch::x86_64::_mm256_loadu_ps(z.as_ptr().add(offset + r));
                            let result = std::arch::x86_64::_mm256_max_ps(z_vec, zero);
                            std::arch::x86_64::_mm256_storeu_ps(a.as_mut_ptr().add(offset + r), result);
                        }
                        for r in simd_len..rows {
                            a[offset + r] = if z[offset + r] > 0.0 { z[offset + r] } else { 0.0 };
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for s in 0..bs {
                        let offset = s * rows;
                        for r in 0..rows {
                            a[offset + r] = if z[offset + r] > 0.0 { z[offset + r] } else { 0.0 };
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
        let num_layers = self.weights.len();
        let inv_bs = 1.0 / bs as f32;

        let out_dim = self.dims[num_layers - 1].0;
        let delta_out_ptr = cache.deltas[num_layers - 1].as_mut_ptr();
        let probs = &cache.activations[num_layers];

        for s in 0..bs {
            let offset = s * out_dim;
            let target = targets[s];
            for r in 0..out_dim {
                unsafe {
                    *delta_out_ptr.add(offset + r) = if r == target {
                        probs[offset + r] - 1.0
                    } else {
                        probs[offset + r]
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
                cache.activations[num_layers - 1].as_ptr(),
                cols as isize,
                1,
                1.0,
                acc_grads.dw[num_layers - 1].as_mut_ptr(),
                cols as isize,
                1,
            );
        }

        let db_last = &mut acc_grads.db[num_layers - 1];
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let inv_bs_vec = std::arch::x86_64::_mm256_set1_ps(inv_bs);
            let simd_len = out_dim - (out_dim % 8);
            for r in (0..simd_len).step_by(8) {
                let mut acc = std::arch::x86_64::_mm256_setzero_ps();
                for s in 0..bs {
                    let val = std::arch::x86_64::_mm256_loadu_ps(delta_out_ptr.add(s * out_dim + r));
                    acc = std::arch::x86_64::_mm256_add_ps(acc, val);
                }
                let existing = std::arch::x86_64::_mm256_loadu_ps(db_last.as_ptr().add(r));
                // result = (acc * inv_bs_vec) + existing
                let result = std::arch::x86_64::_mm256_fmadd_ps(acc, inv_bs_vec, existing);
                std::arch::x86_64::_mm256_storeu_ps(db_last.as_mut_ptr().add(r), result);
            }
            for r in simd_len..out_dim {
                let mut sum = 0.0;
                for s in 0..bs { sum += *delta_out_ptr.add(s * out_dim + r); }
                db_last[r] += sum * inv_bs;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            for r in 0..out_dim {
                let mut sum = 0.0;
                for s in 0..bs {
                    unsafe { sum += *delta_out_ptr.add(s * out_dim + r); }
                }
                db_last[r] += sum * inv_bs;
            }
        }

        for l in (0..num_layers - 1).rev() {
            let (rows_next, _cols_next) = self.dims[l + 1];
            let next_dim = self.dims[l].0;
            let delta_next_ptr = cache.deltas[l + 1].as_ptr();
            let delta_curr_ptr = cache.deltas[l].as_mut_ptr();

            unsafe {
                gemm(
                    bs,
                    rows_next,
                    next_dim,
                    1.0,
                    delta_next_ptr,
                    rows_next as isize,
                    1,
                    self.weights[l + 1].as_ptr(),
                    next_dim as isize,
                    1,
                    0.0,
                    delta_curr_ptr,
                    next_dim as isize,
                    1,
                );
            }

            let z_prev = &cache.pre_activations[l];
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let zero = std::arch::x86_64::_mm256_setzero_ps();
                let simd_len = next_dim - (next_dim % 8);
                for s in 0..bs {
                    let offset = s * next_dim;
                    for r in (0..simd_len).step_by(8) {
                        let z = std::arch::x86_64::_mm256_loadu_ps(z_prev.as_ptr().add(offset + r));
                        let mask = std::arch::x86_64::_mm256_cmp_ps(z, zero, 0x1E);
                        let delta = std::arch::x86_64::_mm256_loadu_ps(delta_curr_ptr.add(offset + r));
                        let result = std::arch::x86_64::_mm256_and_ps(delta, mask);
                        std::arch::x86_64::_mm256_storeu_ps(delta_curr_ptr.add(offset + r), result);
                    }
                    for r in simd_len..next_dim {
                        if z_prev[offset + r] <= 0.0 {
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
                        if z_prev[offset + r] <= 0.0 {
                            unsafe { *delta_curr_ptr.add(offset + r) = 0.0; }
                        }
                    }
                }
            }

            let (r, c) = self.dims[l];
            unsafe {
                gemm(
                    r,
                    bs,
                    c,
                    inv_bs,
                    delta_curr_ptr,
                    1,
                    r as isize,
                    cache.activations[l].as_ptr(),
                    c as isize,
                    1,
                    1.0,
                    acc_grads.dw[l].as_mut_ptr(),
                    c as isize,
                    1,
                );
            }

            let db_l = &mut acc_grads.db[l];
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let inv_bs_vec = std::arch::x86_64::_mm256_set1_ps(inv_bs);
                let simd_len = r - (r % 8);
                for rr in (0..simd_len).step_by(8) {
                    let mut acc = std::arch::x86_64::_mm256_setzero_ps();
                    for s in 0..bs {
                        let val = std::arch::x86_64::_mm256_loadu_ps(delta_curr_ptr.add(s * next_dim + rr));
                        acc = std::arch::x86_64::_mm256_add_ps(acc, val);
                    }
                    let existing = std::arch::x86_64::_mm256_loadu_ps(db_l.as_ptr().add(rr));
                    // result = (acc * inv_bs_vec) + existing
                    let result = std::arch::x86_64::_mm256_fmadd_ps(acc, inv_bs_vec, existing);
                    std::arch::x86_64::_mm256_storeu_ps(db_l.as_mut_ptr().add(rr), result);
                }
                for rr in simd_len..r {
                    let mut sum = 0.0;
                    for s in 0..bs { sum += *delta_curr_ptr.add(s * next_dim + rr); }
                    db_l[rr] += sum * inv_bs;
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for rr in 0..r {
                    let mut sum = 0.0;
                    for s in 0..bs {
                        unsafe { sum += *delta_curr_ptr.add(s * next_dim + rr); }
                    }
                    db_l[rr] += sum * inv_bs;
                }
            }
        }
    }

    pub fn forward<'a>(&self, input: &[f32], cache: &'a mut ForwardCache) -> &'a [f32] {
        cache.activations[0].copy_from_slice(input);

        for i in 0..self.weights.len() {
            let (rows, cols) = self.dims[i];
            let w = self.weights[i].as_ptr();
            let b = self.biases[i].as_ptr();
            let inp = cache.activations[i].as_ptr();
            let z = cache.pre_activations[i].as_mut_ptr();
            let a = cache.activations[i + 1].as_mut_ptr();

            if i == self.weights.len() - 1 {
                unsafe {
                    for r in 0..rows {
                        let mut sum = *b.add(r);
                        let w_offset = r * cols;
                        for c in 0..cols {
                            sum += *w.add(w_offset + c) * *inp.add(c);
                        }
                        *z.add(r) = sum;
                    }
                    let mut max_val = f32::NEG_INFINITY;
                    for r in 0..rows {
                        let val = *z.add(r);
                        if val > max_val {
                            max_val = val;
                        }
                    }
                    let mut sum_exp = 0.0;
                    for r in 0..rows {
                        let e = (*z.add(r) - max_val).exp();
                        *z.add(r) = e;
                        sum_exp += e;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for r in 0..rows {
                        *a.add(r) = *z.add(r) * inv_sum;
                    }
                }
            } else {
                unsafe {
                    for r in 0..rows {
                        let mut sum = *b.add(r);
                        let w_offset = r * cols;
                        for c in 0..cols {
                            sum += *w.add(w_offset + c) * *inp.add(c);
                        }
                        *z.add(r) = sum;
                        *a.add(r) = if sum > 0.0 { sum } else { 0.0 };
                    }
                }
            }
        }
        &cache.activations[self.weights.len()]
    }
}

pub struct ForwardCache {
    pub pre_activations: Vec<Vec<f32>>,
    pub activations: Vec<Vec<f32>>,
}

impl ForwardCache {
    pub fn new(dims: &[(usize, usize)]) -> Self {
        ForwardCache {
            pre_activations: dims.iter().map(|&(r, _)| vec![0.0; r]).collect(),
            activations: {
                let mut acts = Vec::with_capacity(dims.len() + 1);
                acts.push(vec![0.0; dims[0].1]);
                for &(r, _) in dims {
                    acts.push(vec![0.0; r]);
                }
                acts
            },
        }
    }
}
