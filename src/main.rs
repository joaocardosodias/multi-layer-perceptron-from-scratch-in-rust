mod data;
mod linalg;
mod losses;
mod network;
mod optimizers;
mod utils;

use data::{load_images, load_labels};
use losses::cross_entropy;
use network::{BatchCache, Gradients, MLP};
use optimizers::AdamState;
use rayon::prelude::*;
use std::time::Instant;
use utils::*;
use rand::Rng;


fn main() {
    unsafe {
        std::env::set_var("MKL_NUM_THREADS", "1");
        std::env::set_var("OMP_NUM_THREADS", "1");
    }

    let (train_images, num_train) = load_images("src/data/train-images-idx3-ubyte/train-images.idx3-ubyte");
    let train_labels = load_labels("src/data/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    let (test_images, num_test) = load_images("src/data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    let test_labels = load_labels("src/data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    let num_threads = rayon::current_num_threads();
    println!("Treino: {} | Teste: {} | Threads: {}", num_train, num_test, num_threads);

    let mut mlp = MLP::new(&[784, 1024, 512, 10]);

    let batch_size = 256;
    let epochs = 150;

    let mut adam = AdamState::new(&mlp);
    let mut acc_grads = Gradients::new(&mlp);
    let total_start = Instant::now();

    let mut thread_data: Vec<(Vec<f32>, Vec<usize>, BatchCache, Gradients, rand::rngs::StdRng)> = (0..num_threads)
        .map(|t| (
            vec![0.0f32; batch_size * 784],
            vec![0usize; batch_size],
            BatchCache::new(&mlp.dims, batch_size),
            Gradients::new(&mlp),
            rand::SeedableRng::seed_from_u64(42 + t as u64),
        ))
        .collect();

    let mut indices: Vec<usize> = (0..num_train).collect();
    let batch_ranges: Vec<(usize, usize)> = (0..num_train)
        .step_by(batch_size)
        .map(|s| (s, (s + batch_size).min(num_train)))
        .collect();

    let num_super_chunks = (batch_ranges.len() + num_threads - 1) / num_threads;
    let total_steps = epochs * num_super_chunks;
    let max_lr = 9e-3;
    let mut scheduler = optimizers::OneCycleLR::new(total_steps, max_lr);

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
                    let (bi, bt, cache, grads, rng) = res;
                    let bs = b_end - b_start;

                    for (i, &idx) in indices[b_start..b_end].iter().enumerate() {
                        #[cfg(target_arch = "x86_64")]
                        if i + 6 < bs {
                            let pf_idx = indices[b_start + i + 6];
                            unsafe {
                                std::arch::x86_64::_mm_prefetch(
                                    train_images.as_ptr().add(pf_idx * 784) as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                        }
                        let src = idx * 784;
                        let dst = i * 784;
                        if rng.gen_bool(0.85) {
                            let angle = rng.gen_range(-10.0..=10.0);
                            let tx = rng.gen_range(-1.5..=1.5);
                            let ty = rng.gen_range(-1.5..=1.5);
                            let mut geo_buf = [0.0f32; 784];
                            augment_image(
                                &train_images[src..src + 784],
                                &mut geo_buf,
                                angle, tx, ty,
                            );
                            elastic_distort(
                                &geo_buf,
                                &mut bi[dst..dst + 784],
                                38.0, 6.0, rng,
                            );
                        } else {
                            bi[dst..dst + 784].copy_from_slice(&train_images[src..src + 784]);
                        }
                        bt[i] = train_labels[idx];
                    }

                    mlp.forward_batch(bi, cache, bs, true, rng);

                    let out_dim = mlp.dims.last().unwrap().0;
                    let mut loss = 0.0f32;
                    let mut corr = 0usize;
                    for s in 0..bs {
                        let off = s * out_dim;
                        let a_off = cache.a_offsets[mlp.dims.len()];
                        let probs = &cache.activations[a_off + off .. a_off + off + out_dim];
                        loss += cross_entropy(probs, bt[s]);
                        if argmax(probs) == bt[s] { corr += 1; }
                    }

                    grads.zero();
                    mlp.backward_batch(cache, &bt[..bs], grads, bs);

                    (loss, corr, bs)
                })
                .collect();

            acc_grads.zero();
            for (i, &(loss, corr, bs)) in metrics.iter().enumerate() {
                epoch_loss += loss;
                correct += corr;
                total += bs;
                acc_grads.accumulate(&thread_data[i].3);
            }
            let step_lr = scheduler.step();
            optimizers::adam_update(&mut mlp, &acc_grads, &mut adam, step_lr);
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

    println!("Tempo total de treino: {:.2}s", total_start.elapsed().as_secs_f64());
    
    println!("Avaliando com APAC (M=64) ...");
    let apac_start = Instant::now();
    let (apac_acc, apac_loss) = evaluate_apac(&mlp, &test_images, num_test, &test_labels, 64);
    println!("APAC Test Acc: {:.2}% | Test Loss: {:.4} | Tempo: {:.2}s", 
             100.0 * apac_acc, apac_loss, apac_start.elapsed().as_secs_f64());
}

