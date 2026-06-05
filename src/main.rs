mod data;
mod linalg;
mod activations;
mod losses;
mod network;
mod optimizers;
mod utils;

use data::{load_images, load_labels};
use network::{MLP, Gradients};
use losses::cross_entropy;
use optimizers::sgd_update;
use utils::*;

fn main() {
    let train_images = load_images("src/data/train-images-idx3-ubyte/train-images.idx3-ubyte");
    let train_labels = load_labels("src/data/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    let test_images = load_images("src/data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    let test_labels = load_labels("src/data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    println!("Treino: {} imagens | Teste: {} imagens", train_images.len(), test_images.len());

    let mut mlp = MLP::new(&[784, 256, 128, 10]);

    let learning_rate = 0.01;
    let batch_size = 64;
    let epochs = 25;

    for epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..train_images.len()).collect();
        shuffle(&mut indices);

        let mut epoch_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        for batch_start in (0..train_images.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_images.len());
            let batch_indices = &indices[batch_start..batch_end];
            let current_batch_size = batch_indices.len();

            let mut acc_dw = zero_gradients_dw(&mlp.weights);
            let mut acc_db = zero_gradients_db(&mlp.biases);

            for &idx in batch_indices {
                let (probs, cache) = mlp.forward(&train_images[idx]);
                let loss = cross_entropy(&probs, train_labels[idx]);
                epoch_loss += loss;

                let pred = argmax(&probs);
                if pred == train_labels[idx] {
                    correct += 1;
                }
                total += 1;

                let grads = mlp.backward(&cache, &probs, train_labels[idx]);
                accumulate_gradients(&mut acc_dw, &grads.dw);
                accumulate_gradients_db(&mut acc_db, &grads.db);
            }

            scale_gradients(&mut acc_dw, 1.0 / current_batch_size as f64);
            scale_gradients_db(&mut acc_db, 1.0 / current_batch_size as f64);

            let temp_grads = Gradients { dw: acc_dw, db: acc_db };
            let lr = learning_rate / (1.0 + 0.01 * epoch as f64);
            sgd_update(&mut mlp.weights, &mut mlp.biases, &temp_grads, lr);
        }

        let (test_acc, test_loss) = evaluate(&mlp, &test_images, &test_labels);

        println!(
            "Epoca {}/{} | Loss: {:.4} | Acc: {:.2}% | Test Acc: {:.2}% | Test Loss: {:.4}",
            epoch + 1, epochs, epoch_loss / total as f64,
            100.0 * correct as f64 / total as f64,
            100.0 * test_acc, test_loss
        );
    }
}
