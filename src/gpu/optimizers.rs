use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use std::sync::Arc;

use crate::error::GpuError;
use crate::network::{Gradients, MLP};
use crate::kernels::Kernels;
use crate::kernels::launch_adam_update;

pub struct AdamState {
    pub m_w: CudaSlice<f32>,
    pub v_w: CudaSlice<f32>,
    pub m_b: CudaSlice<f32>,
    pub v_b: CudaSlice<f32>,
    pub t: i32,
}

impl AdamState {
    pub fn new(dev: &Arc<CudaDevice>, mlp: &MLP) -> Result<Self, GpuError> {
        let m_w = dev.alloc_zeros::<f32>(mlp.weights.len())?;
        let v_w = dev.alloc_zeros::<f32>(mlp.weights.len())?;
        let m_b = dev.alloc_zeros::<f32>(mlp.biases.len())?;
        let v_b = dev.alloc_zeros::<f32>(mlp.biases.len())?;
        Ok(AdamState {
            m_w, v_w, m_b, v_b, t: 0,
        })
    }
}

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
const WEIGHT_DECAY: f32 = 1e-5;

pub fn adam_update(
    mlp: &mut MLP,
    grads: &mut Gradients,
    state: &mut AdamState,
    lr: f32,
    kernels: &Kernels,
) -> Result<(), GpuError> {
    state.t += 1;

    let w_len = mlp.weights.len();
    let b_len = mlp.biases.len();

    launch_adam_update(
        &kernels.adam_update,
        &mut mlp.weights,
        &mut state.m_w,
        &mut state.v_w,
        &grads.dw,
        w_len,
        lr, BETA1, BETA2, EPS, WEIGHT_DECAY, state.t,
    )?;

    launch_adam_update(
        &kernels.adam_update,
        &mut mlp.biases,
        &mut state.m_b,
        &mut state.v_b,
        &grads.db,
        b_len,
        lr, BETA1, BETA2, EPS, 0.0, state.t,
    )?;

    Ok(())
}

pub struct OneCycleLR {
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
