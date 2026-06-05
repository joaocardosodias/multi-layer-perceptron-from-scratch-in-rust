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
use std::time::Instant;
use utils::*;

fn main() {
    let train_images = load_images("src/data/train-images-idx3-ubyte/train-images.idx3-ubyte");
    let train_labels = load_labels("src/data/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    let test_images = load_images("src/data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    let test_labels = load_labels("src/data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    println!(
        "Treino: {} | Teste: {}",
        train_images.len(),
        test_images.len()
    );

    let mut mlp = MLP::new(&[784, 256, 128, 10]);
    let mut acc_grads = Gradients::new(&mlp);

    let learning_rate = 0.01;
    let batch_size = 64;
    let epochs = 25;

    let total_start = Instant::now();

    let mut batch_input = vec![0.0f32; batch_size * 784];
    let mut batch_targets = vec![0usize; batch_size];
    let mut cache = BatchCache::new(&mlp.dims, batch_size);

    for epoch in 0..epochs {
        let epoch_start = Instant::now();
        let mut indices: Vec<usize> = (0..train_images.len()).collect();
        shuffle(&mut indices);

        let mut epoch_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        for batch_start in (0..train_images.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_images.len());
            let current_batch_size = batch_end - batch_start;

            for (i, &idx) in indices[batch_start..batch_end].iter().enumerate() {
                let offset = i * 784;
                batch_input[offset..offset + 784].copy_from_slice(&train_images[idx]);
                batch_targets[i] = train_labels[idx];
            }

            let out_dim = mlp.dims.last().unwrap().0;
            for s in 0..current_batch_size {
                let offset = s * out_dim;
                let probs = &cache.activations[mlp.weights.len()][offset..offset + out_dim];
                let target = batch_targets[s];
                epoch_loss += cross_entropy(probs, target);
                if argmax(probs) == target { correct += 1; }
                total += 1;
            }

            acc_grads.zero();
            mlp.backward_batch(&mut cache, &batch_targets[..current_batch_size], &mut acc_grads, current_batch_size);

            let lr = learning_rate / (1.0 + 0.01 * epoch as f32);
            optimizers::sgd_update(&mut mlp.weights, &mut mlp.biases, &acc_grads, lr);
        }

        let train_time = epoch_start.elapsed();
        let eval_start = Instant::now();
        let (test_acc, test_loss) = evaluate_single(&mlp, &test_images, &test_labels);
        let eval_time = eval_start.elapsed();

        println!(
            "Epoca {}/{} | Loss: {:.4} | Acc: {:.2}% | Test Acc: {:.2}% | Test Loss: {:.4}",
            epoch + 1,
            epochs,
            epoch_loss / total as f32,
            100.0 * correct as f32 / total as f32,
            100.0 * test_acc,
            test_loss
        );
        println!(
            "  Treino: {:.2}s | Avaliacao: {:.2}s",
            train_time.as_secs_f64(),
            eval_time.as_secs_f64()
        );
    }

    println!("Tempo total: {:.2}s", total_start.elapsed().as_secs_f64());
}
