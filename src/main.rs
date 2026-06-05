mod activations;
mod data;
mod linalg;
mod losses;
mod network;
mod optimizers;
mod utils;

use data::{load_images, load_labels};
use losses::cross_entropy;
use network::{BatchCache, Gradients, MLP};
use rayon::prelude::*;
use std::time::Instant;
use utils::*;

fn main() {
    let (train_images, num_train) = load_images("src/data/train-images-idx3-ubyte/train-images.idx3-ubyte");
    let train_labels = load_labels("src/data/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    let (test_images, num_test) = load_images("src/data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    let test_labels = load_labels("src/data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    let num_threads = rayon::current_num_threads();
    println!("Treino: {} | Teste: {} | Threads: {}", num_train, num_test, num_threads);

    let mut mlp = MLP::new(&[784, 256, 128, 10]);

    let learning_rate = 0.01;
    let batch_size = 64;
    let epochs = 25;

    let total_start = Instant::now();

    let mut thread_data: Vec<(Vec<f32>, Vec<usize>, BatchCache, Gradients)> = (0..num_threads)
        .map(|_| (
            vec![0.0f32; batch_size * 784],
            vec![0usize; batch_size],
            BatchCache::new(&mlp.dims, batch_size),
            Gradients::new(&mlp),
        ))
        .collect();

    let mut indices: Vec<usize> = (0..num_train).collect();
    let batch_ranges: Vec<(usize, usize)> = (0..num_train)
        .step_by(batch_size)
        .map(|s| (s, (s + batch_size).min(num_train)))
        .collect();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        shuffle(&mut indices);

        let mut epoch_loss = 0.0f32;
        let mut correct = 0usize;
        let mut total = 0usize;

        for super_chunk in batch_ranges.chunks(num_threads) {
            let n = super_chunk.len();

            let metrics: Vec<(f32, usize, usize)> = thread_data[..n]
                .par_iter_mut()
                .zip(super_chunk.par_iter())
                .map(|(res, &(b_start, b_end))| {
                    let (bi, bt, cache, grads) = res;
                    let bs = b_end - b_start;

                    for (i, &idx) in indices[b_start..b_end].iter().enumerate() {
                        let src = idx * 784;
                        let dst = i * 784;
                        bi[dst..dst + 784].copy_from_slice(&train_images[src..src + 784]);
                        bt[i] = train_labels[idx];
                    }

                    mlp.forward_batch(bi, cache, bs);

                    let out_dim = mlp.dims.last().unwrap().0;
                    let mut loss = 0.0f32;
                    let mut corr = 0usize;
                    for s in 0..bs {
                        let off = s * out_dim;
                        let probs = &cache.activations[mlp.weights.len()][off..off + out_dim];
                        loss += cross_entropy(probs, bt[s]);
                        if argmax(probs) == bt[s] { corr += 1; }
                    }

                    grads.zero();
                    mlp.backward_batch(cache, &bt[..bs], grads, bs);

                    (loss, corr, bs)
                })
                .collect();

            let lr = learning_rate / (1.0 + 0.01 * epoch as f32);
            for (i, &(loss, corr, bs)) in metrics.iter().enumerate() {
                epoch_loss += loss;
                correct += corr;
                total += bs;
                optimizers::sgd_update(&mut mlp.weights, &mut mlp.biases, &thread_data[i].3, lr);
            }
        }

        let train_time = epoch_start.elapsed();
        let eval_start = Instant::now();
        let (test_acc, test_loss) = evaluate_batch(&mlp, &test_images, num_test, &test_labels);
        let eval_time = eval_start.elapsed();

        println!(
            "Epoca {}/{} | Loss: {:.4} | Acc: {:.2}% | Test Acc: {:.2}% | Test Loss: {:.4}",
            epoch + 1, epochs,
            epoch_loss / total as f32,
            100.0 * correct as f32 / total as f32,
            100.0 * test_acc, test_loss
        );
        println!(
            "  Treino: {:.2}s | Avaliacao: {:.2}s",
            train_time.as_secs_f64(), eval_time.as_secs_f64()
        );
    }

    println!("Tempo total: {:.2}s", total_start.elapsed().as_secs_f64());
}
