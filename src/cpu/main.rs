mod linalg;
mod network;
mod optimizers;
mod utils;

use clap::Parser;
use mlp::common::data::{load_images, load_labels};
use mlp::common::losses::cross_entropy;
use network::{BatchCache, Gradients, MLP};
use optimizers::AdamState;
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;
use utils::*;

#[derive(Parser)]
#[command(name = "mlp-cpu", about = "MLP Trainer using CPU with BLAS")]
struct Cli {
    #[arg(long, default_value_t = 300)]
    epochs: usize,

    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    #[arg(long, default_value_t = 3e-3)]
    learning_rate: f32,

    #[arg(long, default_value_t = 0.85)]
    augment_p_keep: f32,

    #[arg(long, default_value_t = 0.9)]
    dropout_keep: f32,

    #[arg(long, default_value_t = 1e-4)]
    weight_decay: f32,

    #[arg(long, default_value_t = 0.0)]
    label_smoothing: f32,

    #[arg(long)]
    arch: Option<String>,
}

fn main() {
    let args = Cli::parse();

    unsafe {
        std::env::set_var("MKL_NUM_THREADS", "1");
        std::env::set_var("OMP_NUM_THREADS", "1");
    }

    let (train_images, num_train) =
        load_images("src/data/train-images-idx3-ubyte/train-images.idx3-ubyte");
    let train_labels = load_labels("src/data/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    let (test_images, num_test) =
        load_images("src/data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    let test_labels = load_labels("src/data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");

    let num_threads = rayon::current_num_threads();
    println!(
        "Treino: {} | Teste: {} | Threads: {}",
        num_train, num_test, num_threads
    );

    let arch_str = args.arch.clone().unwrap_or("784,2048,1024,10".to_string());
    let architecture: Vec<usize> = arch_str
        .split(',')
        .map(|s| s.parse().expect("Arquitetura inválida"))
        .collect();
    println!("Arquitetura: {:?}", architecture);
    println!("Épocas: {} | Batch: {} | LR: {:.1e}", args.epochs, args.batch_size, args.learning_rate);

    let mut mlp = MLP::new(&architecture);

    let batch_size = args.batch_size;
    let epochs = args.epochs;

    let mut adam = AdamState::new(&mlp);
    let mut acc_grads = Gradients::new(&mlp);
    let total_start = Instant::now();
    let mut best_test_acc = 0.0f32;
    let mut best_epoch = 0;

    let mut thread_data: Vec<(
        Vec<f32>,
        Vec<usize>,
        BatchCache,
        Gradients,
        rand::rngs::StdRng,
    )> = (0..num_threads)
        .map(|t| {
            (
                vec![0.0f32; batch_size * 784],
                vec![0usize; batch_size],
                BatchCache::new(&mlp.dims, batch_size),
                Gradients::new(&mlp),
                rand::SeedableRng::seed_from_u64(42 + t as u64),
            )
        })
        .collect();

    let mut indices: Vec<usize> = (0..num_train).collect();
    let batch_ranges: Vec<(usize, usize)> = (0..num_train)
        .step_by(batch_size)
        .map(|s| (s, (s + batch_size).min(num_train)))
        .collect();

    let num_super_chunks = (batch_ranges.len() + num_threads - 1) / num_threads;
    let total_steps = epochs * num_super_chunks;
    let max_lr = 3e-3;
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
                                angle,
                                tx,
                                ty,
                            );
                            elastic_distort(&geo_buf, &mut bi[dst..dst + 784], 36.0, 5.0, rng);
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
                        let probs = &cache.activations[a_off + off..a_off + off + out_dim];
                        loss += cross_entropy(probs, bt[s]);
                        if argmax(probs) == bt[s] {
                            corr += 1;
                        }
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

        let train_loss = epoch_loss / total as f32;
        let train_acc = 100.0 * correct as f32 / total as f32;
        
        use std::fs::File;
        use std::io::Write;
        
        if epoch == 0 {
            if let Ok(mut file) = File::create("training_log.csv") {
                let _ = writeln!(file, "epoch,train_loss,train_acc,test_acc,test_loss");
            }
        }
        
        if let Ok(mut file) = File::options().append(true).open("training_log.csv") {
            let _ = writeln!(
                file, 
                "{},{:.6},{:.2},{:.2},{:.6}", 
                epoch + 1, 
                train_loss, 
                train_acc, 
                test_acc * 100.0, 
                test_loss
            );
        }

        if test_acc > best_test_acc {
            best_test_acc = test_acc;
            best_epoch = epoch + 1;
        }
    }

    println!(
        "Tempo total de treino: {:.2}s",
        total_start.elapsed().as_secs_f64()
    );
    println!(
        "Melhor acuracia de teste: {:.2}% na Epoca {}",
        best_test_acc * 100.0,
        best_epoch
    );

    #[cfg(feature = "auto-plot")]
    {
        use plotters::prelude::*;
        use std::fs::File;
        use std::io::BufReader;

        let mut epochs = vec![];
        let mut train_losses = vec![];
        let mut train_accs = vec![];
        let mut test_accs = vec![];
        let mut test_losses = vec![];

        if let Ok(file) = File::open("training_log.csv") {
            let mut reader = csv::Reader::from_reader(BufReader::new(file));
            for result in reader.records() {
                if let Ok(record) = result {
                    if let (Ok(e), Ok(tl), Ok(ta), Ok(te), Ok(tel)) = (
                        record[0].parse::<u32>(),
                        record[1].parse::<f32>(),
                        record[2].parse::<f32>(),
                        record[3].parse::<f32>(),
                        record[4].parse::<f32>(),
                    ) {
                        epochs.push(e);
                        train_losses.push(tl);
                        train_accs.push(ta);
                        test_accs.push(te);
                        test_losses.push(tel);
                    }
                }
            }
        }

        if !epochs.is_empty() {
            let output_path = "training_plot.png";
            let root = BitMapBackend::new(output_path, (1200, 500)).into_drawing_area();
            let white = WHITE;
            root.fill(&white).unwrap();

            let (left, right) = root.split_horizontally(600);

            let max_epoch = *epochs.iter().max().unwrap_or(&300) + 10;
            let mut acc_chart = ChartBuilder::on(&left)
                .caption("Acurácia ao longo do Treinamento", ("sans-serif", 30))
                .margin(10)
                .x_label_area_size(50)
                .y_label_area_size(60)
                .build_cartesian_2d(0u32..max_epoch, 90.0f32..100.0f32)
                .unwrap();

            acc_chart
                .configure_mesh()
                .x_label_style(("sans-serif", 20))
                .y_label_style(("sans-serif", 20))
                .draw()
                .unwrap();

            acc_chart
                .draw_series(LineSeries::new(
                    epochs.iter().zip(train_accs.iter()).map(|(&x, &y)| (x, y)),
                    BLUE,
                ))
                .unwrap()
                .label("Treino")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

            acc_chart
                .draw_series(LineSeries::new(
                    epochs.iter().zip(test_accs.iter()).map(|(&x, &y)| (x, y)),
                    RED,
                ))
                .unwrap()
                .label("Teste")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

            acc_chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperLeft)
                .label_font(("sans-serif", 20))
                .draw()
                .unwrap();

            let max_loss = train_losses.iter().chain(test_losses.iter()).cloned().fold(0.0f32, f32::max) * 1.2;
            let mut loss_chart = ChartBuilder::on(&right)
                .caption("Loss ao longo do Treinamento", ("sans-serif", 30))
                .margin(10)
                .x_label_area_size(50)
                .y_label_area_size(60)
                .build_cartesian_2d(0u32..max_epoch, 0.0f32..max_loss)
                .unwrap();

            loss_chart
                .configure_mesh()
                .x_label_style(("sans-serif", 20))
                .y_label_style(("sans-serif", 20))
                .draw()
                .unwrap();

            loss_chart
                .draw_series(LineSeries::new(
                    epochs.iter().zip(train_losses.iter()).map(|(&x, &y)| (x, y)),
                    BLUE,
                ))
                .unwrap()
                .label("Treino")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

            loss_chart
                .draw_series(LineSeries::new(
                    epochs.iter().zip(test_losses.iter()).map(|(&x, &y)| (x, y)),
                    RED,
                ))
                .unwrap()
                .label("Teste")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

            loss_chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperRight)
                .label_font(("sans-serif", 20))
                .draw()
                .unwrap();

            root.present().unwrap();
            println!("✅ Gráfico salvo em '{}'", output_path);
        }
    }
}
