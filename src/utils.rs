use crate::network::{MLP, ForwardCache, Gradients};
use crate::losses::cross_entropy;
use rayon::prelude::*;

pub fn shuffle(indices: &mut [usize]) {
    use std::time::SystemTime;
    let mut seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    for i in (1..indices.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((seed >> 33) as usize) % (i + 1);
        indices.swap(i, j);
    }
}

pub fn argmax(v: &[f32]) -> usize {
    let mut max_idx = 0;
    for i in 1..v.len() {
        if v[i] > v[max_idx] { max_idx = i; }
    }
    max_idx
}

pub fn zero_gradients(mlp: &MLP) -> Gradients {
    Gradients {
        dw: mlp.weights.iter().map(|w| vec![0.0; w.len()]).collect(),
        db: mlp.biases.iter().map(|b| vec![0.0; b.len()]).collect(),
    }
}

pub fn scale_gradients(grads: &mut Gradients, scale: f32) {
    for w in grads.dw.iter_mut() { for x in w.iter_mut() { *x *= scale; } }
    for b in grads.db.iter_mut() { for x in b.iter_mut() { *x *= scale; } }
}

pub fn accumulate_gradients(acc: &mut Gradients, grads: &Gradients) {
    for (a_w, g_w) in acc.dw.iter_mut().zip(grads.dw.iter()) {
        for (a, g) in a_w.iter_mut().zip(g_w.iter()) { *a += g; }
    }
    for (a_b, g_b) in acc.db.iter_mut().zip(grads.db.iter()) {
        for (a, g) in a_b.iter_mut().zip(g_b.iter()) { *a += g; }
    }
}

pub fn evaluate_single(mlp: &MLP, images: &[Vec<f32>], labels: &[usize]) -> (f32, f32) {
    let chunk_size = (images.len() + rayon::current_num_threads() - 1) / rayon::current_num_threads();

    let (correct, total_loss): (usize, f32) = images
        .par_chunks(chunk_size)
        .zip(labels.par_chunks(chunk_size))
        .map(|(imgs, lbls)| {
            let mut cache = ForwardCache::new(&mlp.dims);
            let mut c = 0usize;
            let mut loss = 0.0f32;
            for (img, &label) in imgs.iter().zip(lbls.iter()) {
                let probs = mlp.forward(img, &mut cache);
                if argmax(probs) == label { c += 1; }
                loss += cross_entropy(probs, label);
            }
            (c, loss)
        })
        .reduce(|| (0usize, 0.0f32), |a, b| (a.0 + b.0, a.1 + b.1));

    let n = images.len() as f32;
    (correct as f32 / n, total_loss / n)
}
