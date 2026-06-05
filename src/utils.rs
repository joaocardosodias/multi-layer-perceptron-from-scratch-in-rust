use crate::network::{BatchCache, MLP};
use crate::losses::cross_entropy;
use rand::seq::SliceRandom;
use rayon::prelude::*;

pub fn shuffle(indices: &mut [usize]) {
    indices.shuffle(&mut rand::thread_rng());
}

pub fn argmax(v: &[f32]) -> usize {
    let mut max_idx = 0;
    for i in 1..v.len() {
        if v[i] > v[max_idx] { max_idx = i; }
    }
    max_idx
}

pub fn evaluate_batch(mlp: &MLP, images: &[f32], num_images: usize, labels: &[usize]) -> (f32, f32) {
    let eval_bs = 256;
    let num_threads = rayon::current_num_threads();
    let chunk_size = ((num_images + num_threads - 1) / num_threads).next_multiple_of(eval_bs).max(eval_bs);

    let (correct, total_loss): (usize, f32) = (0..num_images)
        .collect::<Vec<_>>()
        .par_chunks(chunk_size)
        .map(|indices| {
            let mut cache = BatchCache::new(&mlp.dims, eval_bs);
            let mut batch_input = vec![0.0f32; eval_bs * 784];
            let out_dim = mlp.dims.last().unwrap().0;
            let mut c = 0usize;
            let mut loss = 0.0f32;

            for chunk in indices.chunks(eval_bs) {
                let bs = chunk.len();
                for (k, &i) in chunk.iter().enumerate() {
                    #[cfg(target_arch = "x86_64")]
                    if k + 6 < bs {
                        let pf_i = chunk[k + 6];
                        unsafe {
                            std::arch::x86_64::_mm_prefetch(
                                images.as_ptr().add(pf_i * 784) as *const i8,
                                std::arch::x86_64::_MM_HINT_T0,
                            );
                        }
                    }
                    batch_input[k * 784..(k + 1) * 784]
                        .copy_from_slice(&images[i * 784..(i + 1) * 784]);
                }
                mlp.forward_batch(&batch_input, &mut cache, bs);
                for (k, &i) in chunk.iter().enumerate() {
                    let off = k * out_dim;
                    let a_off = cache.a_offsets[mlp.dims.len()];
                    let probs = &cache.activations[a_off + off .. a_off + off + out_dim];
                    if argmax(probs) == labels[i] { c += 1; }
                    loss += cross_entropy(probs, labels[i]);
                }
            }
            (c, loss)
        })
        .reduce(|| (0usize, 0.0f32), |a, b| (a.0 + b.0, a.1 + b.1));

    let n = num_images as f32;
    (correct as f32 / n, total_loss / n)
}
