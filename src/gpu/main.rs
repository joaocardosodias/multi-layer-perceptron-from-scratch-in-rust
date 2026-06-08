mod error;
mod linalg;
mod network;
mod optimizers;
mod utils;
mod kernels;

use cudarc::driver::{CudaDevice, DeviceSlice};
use std::time::Instant;

use mlp::common::data::{load_images, load_labels};
use mlp::common::augment::generate_augmented_dataset;
use network::{BatchCache, Gradients, MLP};
use optimizers::{AdamState, OneCycleLR, adam_update};
use linalg::BlasHandle;
use kernels::{Kernels, launch_gather_and_augment, launch_gather_labels, launch_count_correct, launch_compute_loss};

fn main() {
    const BATCH_SIZE: usize = 256;
    const EPOCHS: usize = 300;
    const LABEL_SMOOTHING: f32 = 0.0;
    const MAX_LR: f32 = 6e-3;
    const AUGMENT_P_KEEP: f32 = 0.90;

    let dev = CudaDevice::new(0).expect("Falha ao inicializar CUDA");
    println!("GPU: {:?}", dev);

    // 1. Carrega dados na CPU
    let (train_images, num_train) = load_images("src/data/train-images-idx3-ubyte/train-images.idx3-ubyte");
    let train_labels = load_labels("src/data/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    let (test_images, num_test) = load_images("src/data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    let test_labels = load_labels("src/data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    // 2. Envia dados de teste para GPU (uma vez só)
    let test_images_gpu = dev.htod_sync_copy(&test_images).expect("Falha ao copiar imagens de teste");
    let test_labels_gpu = dev.htod_sync_copy(&test_labels.iter().map(|&x| x as i32).collect::<Vec<_>>()).expect("Falha ao copiar labels de teste");

    // 3. Cria MLP e kernels na GPU
    let mut mlp = MLP::new(&dev, &[784, 2048, 1024, 10]).expect("Falha ao criar MLP");
    let blas = BlasHandle::new(dev.clone()).expect("Falha ao criar cuBLAS");
    let kernels = Kernels::new(&dev).expect("Falha ao compilar kernels");

    // 4. Buffers de batch e índices
    let mut batch_input = dev.alloc_zeros::<f32>(BATCH_SIZE * 784).expect("Falha alloc batch");
    let mut batch_labels = dev.alloc_zeros::<i32>(BATCH_SIZE).expect("Falha alloc labels");
    let mut batch_indices = dev.alloc_zeros::<i32>(BATCH_SIZE).expect("Falha alloc indices");
    let mut batch_indices_cpu = vec![0i32; BATCH_SIZE];
    let mut indices: Vec<usize> = (0..num_train).collect();

    // 5. Buffers para métricas na GPU
    let mut correct_gpu = dev.alloc_zeros::<i32>(1).expect("Falha alloc correct");
    let mut loss_gpu = dev.alloc_zeros::<f32>(1).expect("Falha alloc loss");

    // 6. Cache para treino e eval
    let mut cache = BatchCache::new(&dev, &mlp.dims, BATCH_SIZE).expect("Falha cache");
    let mut eval_cache = BatchCache::new(&dev, &mlp.dims, BATCH_SIZE).expect("Falha eval cache");

    let mut adam = AdamState::new(&dev, &mlp).expect("Falha Adam");
    let mut acc_grads = Gradients::new(&dev, &mlp).expect("Falha Grads");

    // 7. Buffer para dataset augmentado na CPU
    let mut train_images_augmented = vec![0.0f32; num_train * 784];

    let total_start = Instant::now();
    let mut best_test_acc = 0.0f32;
    let mut best_epoch = 0;

    let num_batches = (num_train + BATCH_SIZE - 1) / BATCH_SIZE;
    let total_steps = EPOCHS * num_batches;
    let mut scheduler = OneCycleLR::new(total_steps, MAX_LR);

    for epoch in 0..EPOCHS {
        let epoch_start = Instant::now();
        shuffle_indices(&mut indices);

        // Gera dataset augmentado na CPU (exatamente como CPU faz)
        let aug_start = Instant::now();
        train_images_augmented = generate_augmented_dataset(
            &train_images,
            num_train,
            AUGMENT_P_KEEP,
            epoch as u64,
        );
        let aug_time = aug_start.elapsed();

        // Copia dataset augmentado para GPU
        let train_images_gpu = dev.htod_sync_copy(&train_images_augmented).expect("Falha ao copiar augmented");
        let train_labels_gpu = dev.htod_sync_copy(&train_labels.iter().map(|&x| x as i32).collect::<Vec<_>>()).expect("Falha ao copiar labels");

        let mut epoch_loss = 0.0f32;
        let mut correct = 0usize;
        let mut total = 0usize;

        for batch_start in (0..num_train).step_by(BATCH_SIZE) {
            let bs = (batch_start + BATCH_SIZE).min(num_train) - batch_start;

            // Copia índices CPU -> GPU
            for i in 0..bs {
                batch_indices_cpu[i] = indices[batch_start + i] as i32;
            }
            dev.htod_sync_copy_into(&batch_indices_cpu[..bs], &mut batch_indices.slice_mut(0..bs))
                .expect("Falha copiar indices");

            // Copia batch augmentado da CPU para GPU
            let mut batch_cpu = vec![0.0f32; bs * 784];
            for i in 0..bs {
                let idx = indices[batch_start + i];
                batch_cpu[i * 784..(i + 1) * 784].copy_from_slice(
                    &train_images_augmented[idx * 784..(idx + 1) * 784]
                );
            }
            dev.htod_sync_copy_into(&batch_cpu, &mut batch_input.slice_mut(0..bs * 784))
                .expect("Falha copiar batch");

            // Gather labels na GPU
            launch_gather_labels(
                &kernels.gather_labels,
                &train_labels_gpu.slice(0..train_labels_gpu.len()),
                &batch_indices.slice(0..bs),
                &mut batch_labels.slice_mut(0..bs),
                bs,
            ).expect("Falha gather labels");

            // Forward
            let dev_input = batch_input.slice(0..bs * 784);
            mlp.forward_batch(&dev_input, &mut cache, bs, true, &kernels, &blas)
                .expect("Falha forward");

            // Métricas na GPU
            let a_last = cache.a_offsets[mlp.dims.len()];
            let probs = cache.activations.slice(a_last..a_last + bs * 10);

            // Count correct
            dev.memset_zeros(&mut correct_gpu).expect("memset");
            launch_count_correct(
                &kernels.count_correct,
                &probs,
                &batch_labels.slice(0..bs),
                &mut correct_gpu.slice_mut(0..1),
                bs,
            ).expect("Falha count correct");
            let correct_host = dev.dtoh_sync_copy(&correct_gpu.slice(0..1)).expect("dtoh");
            correct += correct_host[0] as usize;

            // Compute loss
            dev.memset_zeros(&mut loss_gpu).expect("memset");
            launch_compute_loss(
                &kernels.compute_loss,
                &probs,
                &batch_labels.slice(0..bs),
                &mut loss_gpu.slice_mut(0..1),
                bs, 10,
            ).expect("Falha loss");
            let loss_host = dev.dtoh_sync_copy(&loss_gpu.slice(0..1)).expect("dtoh");
            epoch_loss += loss_host[0];
            total += bs;

            // Backward
            acc_grads.zero().expect("Falha zero grads");
            mlp.backward_batch(&mut cache, &batch_labels.slice(0..bs), &mut acc_grads, bs, &kernels, &blas, LABEL_SMOOTHING)
                .expect("Falha backward");

            // Adam
            let step_lr = scheduler.step();
            adam_update(&mut mlp, &mut acc_grads, &mut adam, step_lr, &kernels)
                .expect("Falha adam");
        }

        let train_time = epoch_start.elapsed();

        // Eval no teste
        let eval_start = Instant::now();
        let mut test_correct = 0usize;
        let mut test_loss = 0.0f32;
        let mut batch_input_eval = dev.alloc_zeros::<f32>(BATCH_SIZE * 784).expect("Falha alloc eval");
        let mut batch_labels_eval = dev.alloc_zeros::<i32>(BATCH_SIZE).expect("Falha alloc eval labels");

        for chunk_start in (0..num_test).step_by(BATCH_SIZE) {
            let bs = (chunk_start + BATCH_SIZE).min(num_test) - chunk_start;

            // Índices sequenciais para teste
            for i in 0..bs {
                batch_indices_cpu[i] = (chunk_start + i) as i32;
            }
            dev.htod_sync_copy_into(&batch_indices_cpu[..bs], &mut batch_indices.slice_mut(0..bs))
                .expect("Falha copiar indices eval");

            // Gather imagens de teste (sem augmentation)
            launch_gather_and_augment(
                &kernels.gather_and_augment,
                &test_images_gpu.slice(0..test_images_gpu.len()),
                &batch_indices.slice(0..bs),
                &mut batch_input_eval.slice_mut(0..bs * 784),
                bs,
                0,
                0.0,
            ).expect("Falha gather eval");

            // Gather labels
            launch_gather_labels(
                &kernels.gather_labels,
                &test_labels_gpu.slice(0..test_labels_gpu.len()),
                &batch_indices.slice(0..bs),
                &mut batch_labels_eval.slice_mut(0..bs),
                bs,
            ).expect("Falha gather labels eval");

            let dev_input = batch_input_eval.slice(0..bs * 784);
            mlp.forward_batch(&dev_input, &mut eval_cache, bs, false, &kernels, &blas)
                .expect("Falha forward eval");

            let a_last = eval_cache.a_offsets[mlp.dims.len()];
            let probs = eval_cache.activations.slice(a_last..a_last + bs * 10);

            // Count correct
            dev.memset_zeros(&mut correct_gpu).expect("memset");
            launch_count_correct(
                &kernels.count_correct,
                &probs,
                &batch_labels_eval.slice(0..bs),
                &mut correct_gpu.slice_mut(0..1),
                bs,
            ).expect("Falha count eval");
            let correct_host = dev.dtoh_sync_copy(&correct_gpu.slice(0..1)).expect("dtoh");
            test_correct += correct_host[0] as usize;

            // Loss
            dev.memset_zeros(&mut loss_gpu).expect("memset");
            launch_compute_loss(
                &kernels.compute_loss,
                &probs,
                &batch_labels_eval.slice(0..bs),
                &mut loss_gpu.slice_mut(0..1),
                bs, 10,
            ).expect("Falha loss eval");
            let loss_host = dev.dtoh_sync_copy(&loss_gpu.slice(0..1)).expect("dtoh");
            test_loss += loss_host[0];
        }

        let eval_time = eval_start.elapsed();
        let test_acc = test_correct as f32 / num_test as f32;
        test_loss /= num_test as f32;

        println!(
            "Epoca {}/{} | Loss: {:.4} | Acc: {:.2}% | Test Acc: {:.2}% | Test Loss: {:.4} | Aug: {:.2}s",
            epoch + 1, EPOCHS,
            epoch_loss / total as f32,
            100.0 * correct as f32 / total as f32,
            100.0 * test_acc, test_loss,
            aug_time.as_secs_f64()
        );
        println!(
            "  Treino: {:.2}s | Avaliacao: {:.2}s",
            train_time.as_secs_f64(), eval_time.as_secs_f64()
        );

        if test_acc > best_test_acc {
            best_test_acc = test_acc;
            best_epoch = epoch + 1;
        }
    }

    println!("Tempo total: {:.2}s", total_start.elapsed().as_secs_f64());
    println!("Melhor: {:.2}% na Epoca {}", best_test_acc * 100.0, best_epoch);
}

fn shuffle_indices(indices: &mut [usize]) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for i in (1..indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
}
