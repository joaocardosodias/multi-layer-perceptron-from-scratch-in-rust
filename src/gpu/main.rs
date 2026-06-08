mod error;
mod linalg;
mod network;
mod optimizers;
mod utils;
mod kernels;

use cudarc::driver::CudaDevice;
use std::time::Instant;
use rand::SeedableRng;

use mlp::common::data::{load_images, load_labels};
use mlp::common::losses::cross_entropy;
use utils::*;
use network::{BatchCache, Gradients, MLP};
use optimizers::{AdamState, OneCycleLR, adam_update};
use linalg::BlasHandle;
use kernels::Kernels;

fn main() {
    let dev = CudaDevice::new(0).expect("Falha ao inicializar CUDA device 0");
    println!("GPU: {:?}", dev);

    let (train_images, num_train) = load_images("src/data/train-images-idx3-ubyte/train-images.idx3-ubyte");
    let train_labels = load_labels("src/data/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    let (test_images, num_test) = load_images("src/data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    let test_labels = load_labels("src/data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    println!("Treino: {} | Teste: {}", num_train, num_test);

    let mut mlp = MLP::new(&dev, &[784, 2048, 1024, 10]).expect("Falha ao criar MLP na GPU");
    let blas    = BlasHandle::new(dev.clone()).expect("Falha ao criar cuBLAS handle");
    let kernels = Kernels::new(&dev).expect("Falha ao compilar kernels CUDA");

    let batch_size = 512;
    let epochs     = 300;

    let mut adam      = AdamState::new(&dev, &mlp).expect("Falha ao criar AdamState");
    let mut acc_grads = Gradients::new(&dev, &mlp).expect("Falha ao criar Gradients");
    let mut cache     = BatchCache::new(&dev, &mlp.dims, batch_size).expect("Falha ao criar BatchCache");

    let total_start = Instant::now();
    let mut best_test_acc = 0.0f32;
    let mut best_epoch    = 0;

    let mut batch_input      = dev.alloc_zeros::<f32>(batch_size * 784).expect("Falha ao alocar batch_input");
    let mut batch_targets     = dev.alloc_zeros::<i32>(batch_size).expect("Falha ao alocar batch_targets");
    let mut batch_targets_host = vec![0i32; batch_size];

    let mut indices: Vec<usize> = (0..num_train).collect();

    let num_batches = (num_train + batch_size - 1) / batch_size;
    let total_steps = epochs * num_batches;
    let max_lr      = 3e-3;
    let mut scheduler = OneCycleLR::new(total_steps, max_lr);

    let mut batch_input_host = vec![0.0f32; batch_size * 784];

    for epoch in 0..epochs {
        let epoch_start = Instant::now();

        let mut rng = rand::rngs::StdRng::seed_from_u64(epoch as u64 * 1_000_003 + 7);

        shuffle(&mut indices);

        let mut epoch_loss = 0.0f32;
        let mut correct    = 0usize;
        let mut total      = 0usize;

        for batch_start in (0..num_train).step_by(batch_size) {
            let bs = (batch_start + batch_size).min(num_train) - batch_start;

            for i in 0..bs {
                let idx = indices[batch_start + i];
                let src = idx * 784;
                let dst = i * 784;
                let mut host_buf   = [0.0f32; 784];

                use rand::Rng;
                if rng.gen_bool(0.85) {
                    let angle = rng.gen_range(-10.0f32..=10.0);
                    let tx    = rng.gen_range(-1.5f32..=1.5);
                    let ty    = rng.gen_range(-1.5f32..=1.5);
                    augment_image(&train_images[src..src + 784], &mut host_buf, angle, tx, ty);
                    elastic_distort(&host_buf, &mut batch_input_host[dst..dst + 784], 36.0, 5.0, &mut rng);
                } else {
                    batch_input_host[dst..dst + 784].copy_from_slice(&train_images[src..src + 784]);
                }
                batch_targets_host[i] = train_labels[idx] as i32;
            }

            {
                let mut dev_slice = batch_input.slice_mut(0..bs * 784);
                dev.htod_sync_copy_into(&batch_input_host[0..bs * 784], &mut dev_slice)
                    .expect("Falha ao copiar batch input");
                
                let mut dev_targets = batch_targets.slice_mut(0..bs);
                dev.htod_sync_copy_into(&batch_targets_host[..bs], &mut dev_targets)
                    .expect("Falha ao copiar targets");
            }

            let dev_input = batch_input.slice(0..bs * 784);
            mlp.forward_batch(&dev_input, &mut cache, bs, true, &kernels, &blas)
                .expect("Falha no forward");

            let a_last_off  = cache.a_offsets[mlp.dims.len()];
            let probs_host  = dev.dtoh_sync_copy(
                &cache.activations.slice(a_last_off..a_last_off + bs * 10)
            ).expect("Falha ao copiar ativações");

            let mut loss = 0.0f32;
            let mut corr = 0usize;
            for s in 0..bs {
                let off   = s * 10;
                let probs = &probs_host[off..off + 10];
                loss += cross_entropy(probs, batch_targets_host[s] as usize);
                if argmax(probs) == batch_targets_host[s] as usize { corr += 1; }
            }
            epoch_loss += loss;
            correct    += corr;
            total      += bs;

            acc_grads.zero().expect("Falha ao zerar gradientes");
            let targets_view = batch_targets.slice(0..bs);
            mlp.backward_batch(&mut cache, &targets_view, &mut acc_grads, bs, &kernels, &blas)
                .expect("Falha no backward");

            let step_lr = scheduler.step();
            adam_update(&mut mlp, &mut acc_grads, &mut adam, step_lr, &kernels)
                .expect("Falha no Adam update");
        }

        let train_time = epoch_start.elapsed();
        let eval_start = Instant::now();
        let (test_acc, test_loss) = evaluate_batch(
            &mlp, &test_images, num_test, &test_labels,
            &dev, &kernels, &blas,
        ).expect("Falha na avaliação");
        let eval_time = eval_start.elapsed();

        println!(
            "Epoca {}/{} | Loss: {:.4} | Acc: {:.2}% | Test Acc: {:.2}% | Test Loss: {:.4}",
            epoch + 1, epochs,
            epoch_loss / total as f32,
            100.0 * correct as f32 / total as f32,
            100.0 * test_acc,
            test_loss,
        );
        println!(
            "  Treino: {:.2}s | Avaliacao: {:.2}s",
            train_time.as_secs_f64(),
            eval_time.as_secs_f64(),
        );

        if test_acc > best_test_acc {
            best_test_acc = test_acc;
            best_epoch    = epoch + 1;
        }
    }

    println!("Tempo total de treino: {:.2}s", total_start.elapsed().as_secs_f64());
    println!("Melhor acuracia de teste: {:.2}% na Epoca {}", best_test_acc * 100.0, best_epoch);
}
