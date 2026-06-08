use crate::network::{Gradients, MLP};

pub struct OneCycleLR {
    pub _total_steps: usize,
    pub warmup_steps: usize,
    pub decay_steps: usize,
    pub max_lr: f32,
    pub init_lr: f32,
    pub final_lr: f32,
    pub current_step: usize,
}

impl OneCycleLR {
    pub fn new(total_steps: usize, max_lr: f32) -> Self {
        let warmup_steps = (total_steps as f32 * 0.3) as usize;
        let decay_steps = total_steps - warmup_steps;
        OneCycleLR {
            _total_steps: total_steps,
            warmup_steps,
            decay_steps,
            max_lr,
            init_lr: max_lr / 25.0,
            final_lr: max_lr / 1000.0,
            current_step: 0,
        }
    }

    pub fn step(&mut self) -> f32 {
        let lr = if self.current_step < self.warmup_steps {
            let pct = self.current_step as f32 / self.warmup_steps as f32;
            let cos_out = (std::f32::consts::PI * (1.0 + pct)).cos() + 1.0;
            self.init_lr + (self.max_lr - self.init_lr) * 0.5 * cos_out
        } else {
            let pct = ((self.current_step - self.warmup_steps) as f32 / self.decay_steps as f32).min(1.0);
            let cos_out = (std::f32::consts::PI * pct).cos() + 1.0;
            self.final_lr + (self.max_lr - self.final_lr) * 0.5 * cos_out
        };
        self.current_step += 1;
        lr
    }
}

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
const WEIGHT_DECAY: f32 = 1e-4;

pub struct AdamState {
    pub m_w: Vec<f32>,
    pub v_w: Vec<f32>,
    pub m_b: Vec<f32>,
    pub v_b: Vec<f32>,
    pub t: usize,
}

impl AdamState {
    pub fn new(mlp: &MLP) -> Self {
        AdamState {
            m_w: vec![0.0; mlp.weights.len()],
            v_w: vec![0.0; mlp.weights.len()],
            m_b: vec![0.0; mlp.biases.len()],
            v_b: vec![0.0; mlp.biases.len()],
            t: 0,
        }
    }
}

pub fn adam_update(mlp: &mut MLP, grads: &Gradients, state: &mut AdamState, lr: f32) {
    state.t += 1;
    let bc1 = 1.0 - BETA1.powi(state.t as i32);
    let bc2 = 1.0 - BETA2.powi(state.t as i32);
    let lr_c = lr * bc2.sqrt() / bc1;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;

        let b1 = _mm256_set1_ps(BETA1);
        let b2 = _mm256_set1_ps(BETA2);
        let ob1 = _mm256_set1_ps(1.0 - BETA1);
        let ob2 = _mm256_set1_ps(1.0 - BETA2);
        let lr_c_v = _mm256_set1_ps(lr_c);
        let eps_v = _mm256_set1_ps(EPS);
        let wd_v = _mm256_set1_ps(lr * WEIGHT_DECAY);

        let w_len = mlp.weights.len();
        let sw = w_len - (w_len % 8);
        for i in (0..sw).step_by(8) {
            let g  = _mm256_loadu_ps(grads.dw.as_ptr().add(i));
            let m  = _mm256_loadu_ps(state.m_w.as_ptr().add(i));
            let v  = _mm256_loadu_ps(state.v_w.as_ptr().add(i));
            let w  = _mm256_loadu_ps(mlp.weights.as_ptr().add(i));
            let nm = _mm256_fmadd_ps(b1, m, _mm256_mul_ps(ob1, g));
            let nv = _mm256_fmadd_ps(b2, v, _mm256_mul_ps(ob2, _mm256_mul_ps(g, g)));
            let den = _mm256_add_ps(_mm256_sqrt_ps(nv), eps_v);
            let step = _mm256_div_ps(_mm256_mul_ps(lr_c_v, nm), den);
            let nw = _mm256_sub_ps(_mm256_sub_ps(w, step), _mm256_mul_ps(wd_v, w));
            _mm256_storeu_ps(state.m_w.as_mut_ptr().add(i), nm);
            _mm256_storeu_ps(state.v_w.as_mut_ptr().add(i), nv);
            _mm256_storeu_ps(mlp.weights.as_mut_ptr().add(i), nw);
        }
        for i in sw..w_len {
            let g = grads.dw[i];
            state.m_w[i] = BETA1 * state.m_w[i] + (1.0 - BETA1) * g;
            state.v_w[i] = BETA2 * state.v_w[i] + (1.0 - BETA2) * g * g;
            mlp.weights[i] -= lr_c * state.m_w[i] / (state.v_w[i].sqrt() + EPS)
                + lr * WEIGHT_DECAY * mlp.weights[i];
        }

        let b_len = mlp.biases.len();
        let sb = b_len - (b_len % 8);
        for i in (0..sb).step_by(8) {
            let g  = _mm256_loadu_ps(grads.db.as_ptr().add(i));
            let m  = _mm256_loadu_ps(state.m_b.as_ptr().add(i));
            let v  = _mm256_loadu_ps(state.v_b.as_ptr().add(i));
            let bv = _mm256_loadu_ps(mlp.biases.as_ptr().add(i));
            let nm = _mm256_fmadd_ps(b1, m, _mm256_mul_ps(ob1, g));
            let nv = _mm256_fmadd_ps(b2, v, _mm256_mul_ps(ob2, _mm256_mul_ps(g, g)));
            let den = _mm256_add_ps(_mm256_sqrt_ps(nv), eps_v);
            let nb = _mm256_sub_ps(bv, _mm256_div_ps(_mm256_mul_ps(lr_c_v, nm), den));
            _mm256_storeu_ps(state.m_b.as_mut_ptr().add(i), nm);
            _mm256_storeu_ps(state.v_b.as_mut_ptr().add(i), nv);
            _mm256_storeu_ps(mlp.biases.as_mut_ptr().add(i), nb);
        }
        for i in sb..b_len {
            let g = grads.db[i];
            state.m_b[i] = BETA1 * state.m_b[i] + (1.0 - BETA1) * g;
            state.v_b[i] = BETA2 * state.v_b[i] + (1.0 - BETA2) * g * g;
            mlp.biases[i] -= lr_c * state.m_b[i] / (state.v_b[i].sqrt() + EPS);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..mlp.weights.len() {
            let g = grads.dw[i];
            state.m_w[i] = BETA1 * state.m_w[i] + (1.0 - BETA1) * g;
            state.v_w[i] = BETA2 * state.v_w[i] + (1.0 - BETA2) * g * g;
            mlp.weights[i] -= lr_c * state.m_w[i] / (state.v_w[i].sqrt() + EPS)
                + lr * WEIGHT_DECAY * mlp.weights[i];
        }
        for i in 0..mlp.biases.len() {
            let g = grads.db[i];
            state.m_b[i] = BETA1 * state.m_b[i] + (1.0 - BETA1) * g;
            state.v_b[i] = BETA2 * state.v_b[i] + (1.0 - BETA2) * g * g;
            mlp.biases[i] -= lr_c * state.m_b[i] / (state.v_b[i].sqrt() + EPS);
        }
    }
}
