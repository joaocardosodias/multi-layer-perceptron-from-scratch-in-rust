use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

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
    for k in &mut kernel {
        *k /= sum;
    }

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

pub fn elastic_distort(src: &[f32], dst: &mut [f32], alpha: f32, sigma: f32, rng: &mut StdRng) {
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

pub fn generate_augmented_dataset(
    train_images: &[f32],
    num_train: usize,
    p_keep: f32,
    epoch_seed: u64,
) -> Vec<f32> {
    let mut augmented = vec![0.0f32; num_train * 784];

    augmented
        .par_chunks_mut(784)
        .enumerate()
        .for_each(|(idx, dst)| {
            let mut rng = StdRng::seed_from_u64(epoch_seed.wrapping_add(idx as u64));
            let src = &train_images[idx * 784..(idx + 1) * 784];

            if rng.gen_bool(p_keep as f64) {
                let angle = rng.gen_range(-10.0..=10.0);
                let tx = rng.gen_range(-1.5..=1.5);
                let ty = rng.gen_range(-1.5..=1.5);
                let mut geo_buf = [0.0f32; 784];
                augment_image(src, &mut geo_buf, angle, tx, ty);
                elastic_distort(&geo_buf, dst, 36.0, 5.0, &mut rng);
            } else {
                dst.copy_from_slice(src);
            }
        });

    augmented
}
