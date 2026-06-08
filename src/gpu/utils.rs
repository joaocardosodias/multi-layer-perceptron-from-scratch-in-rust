use cudarc::driver::CudaDevice;
use std::sync::Arc;
use rand::prelude::SliceRandom;

use crate::error::GpuError;
use crate::network::{BatchCache, MLP};
use crate::kernels::Kernels;
use crate::linalg::BlasHandle;
use mlp::common::losses::cross_entropy;

pub fn evaluate_batch(
    mlp: &MLP,
    images: &[f32],
    num_images: usize,
    labels: &[usize],
    dev: &Arc<CudaDevice>,
    kernels: &Kernels,
    blas: &BlasHandle,
) -> Result<(f32, f32), GpuError> {
    let eval_bs = 256;
    let mut cache = BatchCache::new(dev, &mlp.dims, eval_bs)?;
    let mut batch_input = dev.alloc_zeros::<f32>(eval_bs * 784)?;
    let out_dim = mlp.dims.last().unwrap().0;
    let mut correct = 0usize;
    let mut total_loss = 0.0f32;

    // Alocar buffer na GPU para targets (i32)
    let mut batch_targets_host = vec![0i32; eval_bs];
    let mut batch_targets = dev.alloc_zeros::<i32>(eval_bs)?;

    for chunk_start in (0..num_images).step_by(eval_bs) {
        let bs = (chunk_start + eval_bs).min(num_images) - chunk_start;

        // Copiar imagens para GPU
        let host_slice = &images[chunk_start * 784..(chunk_start + bs) * 784];
        let mut dev_slice = batch_input.slice_mut(0..bs * 784);
        dev.htod_sync_copy_into(host_slice, &mut dev_slice)?;

        // Copiar labels para GPU (como i32)
        for i in 0..bs {
            batch_targets_host[i] = labels[chunk_start + i] as i32;
        }
        let mut dev_targets = batch_targets.slice_mut(0..bs);
        dev.htod_sync_copy_into(&batch_targets_host[..bs], &mut dev_targets)?;

        // Forward
        let dev_input = batch_input.slice(0..bs * 784);
        mlp.forward_batch(&dev_input, &mut cache, bs, false, 1.0, kernels, blas)?;

        // Copiar resultados de volta para CPU
        let a_last_off = cache.a_offsets[mlp.dims.len()];
        let probs_host = dev.dtoh_sync_copy(&cache.activations.slice(a_last_off..a_last_off + bs * out_dim))?;

        for s in 0..bs {
            let off = s * out_dim;
            let probs = &probs_host[off..off + out_dim];
            let pred = argmax(probs);
            if pred == labels[chunk_start + s] {
                correct += 1;
            }
            total_loss += cross_entropy(probs, labels[chunk_start + s]);
        }
    }

    let n = num_images as f32;
    Ok((correct as f32 / n, total_loss / n))
}

pub fn argmax(v: &[f32]) -> usize {
    let mut max_idx = 0;
    for i in 1..v.len() {
        if v[i] > v[max_idx] { max_idx = i; }
    }
    max_idx
}

pub fn shuffle(indices: &mut [usize]) {
    indices.shuffle(&mut rand::thread_rng());
}

use rand::Rng;

pub fn augment_image(src: &[f32], dst: &mut [f32], angle_deg: f32, tx: f32, ty: f32) {
    let angle_rad = angle_deg.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    let cx = 13.5f32;
    let cy = 13.5f32;

    for y in 0..28 {
        let dy = y as f32 - cy;
        for x in 0..28 {
            let dx = x as f32 - cx;
            let src_x = dx * cos_a + dy * sin_a - tx + cx;
            let src_y = -dx * sin_a + dy * cos_a - ty + cy;

            if src_x >= 0.0 && src_x < 27.0 && src_y >= 0.0 && src_y < 27.0 {
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let wx = src_x - x0 as f32;
                let wy = src_y - y0 as f32;
                let val00 = src[y0 * 28 + x0];
                let val10 = src[y0 * 28 + x1];
                let val01 = src[y1 * 28 + x0];
                let val11 = src[y1 * 28 + x1];
                let val = (1.0 - wx) * (1.0 - wy) * val00
                    + wx * (1.0 - wy) * val10
                    + (1.0 - wx) * wy * val01
                    + wx * wy * val11;
                dst[y * 28 + x] = val;
            } else {
                dst[y * 28 + x] = 0.0;
            }
        }
    }
}

fn gaussian_blur_2d(field: &mut [f32], sigma: f32) {
    let radius = (3.0 * sigma).ceil() as usize;
    let k_size = 2 * radius + 1;
    let mut kernel = vec![0.0f32; k_size];
    let mut sum = 0.0f32;
    for i in 0..k_size {
        let x = i as f32 - radius as f32;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += kernel[i];
    }
    for k in &mut kernel { *k /= sum; }

    let mut tmp = [0.0f32; 784];
    for y in 0..28usize {
        for x in 0..28usize {
            let mut val = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let xi = (x as isize + ki as isize - radius as isize).clamp(0, 27) as usize;
                val += kv * field[y * 28 + xi];
            }
            tmp[y * 28 + x] = val;
        }
    }
    for y in 0..28usize {
        for x in 0..28usize {
            let mut val = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let yi = (y as isize + ki as isize - radius as isize).clamp(0, 27) as usize;
                val += kv * tmp[yi * 28 + x];
            }
            field[y * 28 + x] = val;
        }
    }
}

pub fn elastic_distort(
    src: &[f32],
    dst: &mut [f32],
    alpha: f32,
    sigma: f32,
    rng: &mut rand::rngs::StdRng,
) {
    let mut dx: Vec<f32> = (0..784).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();
    let mut dy: Vec<f32> = (0..784).map(|_| rng.gen_range(-1.0f32..=1.0)).collect();

    gaussian_blur_2d(&mut dx, sigma);
    gaussian_blur_2d(&mut dy, sigma);

    for y in 0..28usize {
        for x in 0..28usize {
            let idx = y * 28 + x;
            let sx = x as f32 + alpha * dx[idx];
            let sy = y as f32 + alpha * dy[idx];

            if sx >= 0.0 && sx < 27.0 && sy >= 0.0 && sy < 27.0 {
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let wx = sx - x0 as f32;
                let wy = sy - y0 as f32;
                dst[idx] = (1.0 - wx) * (1.0 - wy) * src[y0 * 28 + x0]
                    + wx * (1.0 - wy) * src[y0 * 28 + x0 + 1]
                    + (1.0 - wx) * wy * src[(y0 + 1) * 28 + x0]
                    + wx * wy * src[(y0 + 1) * 28 + x0 + 1];
            } else {
                dst[idx] = 0.0;
            }
        }
    }
}
