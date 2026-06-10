use crate::network::{BatchCache, MLP};
use mlp::common::losses::cross_entropy;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

pub fn shuffle(indices: &mut [usize]) {
    indices.shuffle(&mut rand::thread_rng());
}

pub fn argmax(v: &[f32]) -> usize {
    let mut max_idx = 0;
    for i in 1..v.len() {
        if v[i] > v[max_idx] {
            max_idx = i;
        }
    }
    max_idx
}

pub fn evaluate_batch(
    mlp: &MLP,
    images: &[f32],
    num_images: usize,
    labels: &[usize],
) -> (f32, f32) {
    let eval_bs = 256;
    let num_threads = rayon::current_num_threads();
    let chunk_size = ((num_images + num_threads - 1) / num_threads)
        .next_multiple_of(eval_bs)
        .max(eval_bs);

    let (correct, total_loss): (usize, f32) = (0..num_images)
        .collect::<Vec<_>>()
        .par_chunks(chunk_size)
        .map(|indices| {
            let mut cache = BatchCache::new(&mlp.dims, eval_bs);
            let mut batch_input = vec![0.0f32; eval_bs * 784];
            let out_dim = mlp.dims.last().unwrap().0;
            let mut c = 0usize;
            let mut loss = 0.0f32;
            let mut rng = StdRng::seed_from_u64(42);

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
                mlp.forward_batch(&batch_input, &mut cache, bs, false, &mut rng);
                for (k, &i) in chunk.iter().enumerate() {
                    let off = k * out_dim;
                    let a_off = cache.a_offsets[mlp.dims.len()];
                    let probs = &cache.activations[a_off + off..a_off + off + out_dim];
                    if argmax(probs) == labels[i] {
                        c += 1;
                    }
                    loss += cross_entropy(probs, labels[i]);
                }
            }
            (c, loss)
        })
        .reduce(|| (0usize, 0.0f32), |a, b| (a.0 + b.0, a.1 + b.1));

    let n = num_images as f32;
    (correct as f32 / n, total_loss / n)
}

/// Gera e imprime a Matriz de Confusão para o conjunto de teste.
///
/// A Matriz de Confusão é uma tabela 10×10 onde:
/// - Cada **linha** representa a classe REAL (o dígito verdadeiro)
/// - Cada **coluna** representa a classe PREDITA (o que a rede acha que é)
///
/// A diagonal principal (linha == coluna) são os ACERTOS.
/// Qualquer célula fora da diagonal é um ERRO.
///
/// Exemplo: conf[3][5] = 7 significa que a rede confundiu o dígito "3"
/// com o dígito "5" em 7 exemplos do conjunto de teste.
///
/// Retorna a matriz raw para que possa ser usada para gerar um gráfico.
pub fn confusion_matrix(
    mlp: &MLP,
    images: &[f32],
    num_images: usize,
    labels: &[usize],
) -> Vec<Vec<usize>> {
    // Matriz 10×10 inicializada com zeros.
    // conf[real][predito] = número de vezes que ocorreu essa combinação.
    let num_classes = 10;
    let mut conf = vec![vec![0usize; num_classes]; num_classes];

    // Roda o modelo em batches para não estourar memória
    let eval_bs = 256;
    let mut cache = BatchCache::new(&mlp.dims, eval_bs);
    let mut batch_input = vec![0.0f32; eval_bs * 784];
    let out_dim = mlp.dims.last().unwrap().0;
    let mut rng = StdRng::seed_from_u64(42);

    let indices: Vec<usize> = (0..num_images).collect();
    for chunk in indices.chunks(eval_bs) {
        let bs = chunk.len();
        // Copia as imagens do chunk atual para o buffer de entrada
        for (k, &i) in chunk.iter().enumerate() {
            batch_input[k * 784..(k + 1) * 784]
                .copy_from_slice(&images[i * 784..(i + 1) * 784]);
        }
        // Forward pass sem dropout (modo inferência)
        mlp.forward_batch(&batch_input, &mut cache, bs, false, &mut rng);

        // Para cada amostra do batch, pega a classe com maior probabilidade
        // e registra na posição [classe_real][classe_predita]
        for (k, &i) in chunk.iter().enumerate() {
            let off = k * out_dim;
            let a_off = cache.a_offsets[mlp.dims.len()];
            let probs = &cache.activations[a_off + off..a_off + off + out_dim];
            let predicted = argmax(probs); // classe que a rede escolheu
            let real = labels[i];         // classe verdadeira
            conf[real][predicted] += 1;
        }
    }

    // --- Impressão formatada da matriz no terminal ---
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║              MATRIZ DE CONFUSÃO (Conjunto de Teste)     ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Linhas = Classe Real  |  Colunas = Classe Predita      ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    print!("       ");
    for j in 0..num_classes {
        print!(" {:>5}", j); // cabeçalho das colunas (predições)
    }
    println!("\n       {}", "──────".repeat(num_classes));

    for i in 0..num_classes {
        print!("  {:>2}  │", i); // cabeçalho da linha (classe real)
        for j in 0..num_classes {
            if i == j {
                // Diagonal principal = acertos
                print!(" {:>5}", conf[i][j]);
            } else {
                // Fora da diagonal = erros
                print!(" {:>5}", conf[i][j]);
            }
        }
        // Calcula e exibe o Recall por classe (acurácia só daquele dígito)
        let total_real: usize = conf[i].iter().sum();
        let recall = if total_real > 0 {
            conf[i][i] as f32 / total_real as f32 * 100.0
        } else {
            0.0
        };
        println!("  │ Acc: {:.1}%", recall);
    }
    println!("       {}", "──────".repeat(num_classes));

    // Resumo dos principais erros (as maiores confusões fora da diagonal)
    println!("\n── Top-5 Confusões ──────────────────────────────────────────");
    let mut erros: Vec<(usize, usize, usize)> = vec![]; // (contagem, real, predito)
    for i in 0..num_classes {
        for j in 0..num_classes {
            if i != j && conf[i][j] > 0 {
                erros.push((conf[i][j], i, j));
            }
        }
    }
    erros.sort_by(|a, b| b.0.cmp(&a.0)); // ordena do maior erro para o menor
    for (count, real, pred) in erros.iter().take(5) {
        println!(
            "  Dígito {:} confundido com {:} → {:} vezes",
            real, pred, count
        );
    }
    println!("─────────────────────────────────────────────────────────────\n");

    conf
}

/// Gera um heatmap PNG da Matriz de Confusão usando `plotters`.
///
/// Cada célula é colorida por intensidade:
/// - **Diagonal (acertos):** escala de verde — quanto mais acertos, mais escuro
/// - **Fora da diagonal (erros):** escala de vermelho — quanto mais erros, mais intenso
///
/// O número de ocorrências é escrito dentro de cada célula.
/// Disponível apenas quando compilado com `--features auto-plot`.
#[cfg(feature = "auto-plot")]
pub fn plot_confusion_matrix(conf: &[Vec<usize>], output_path: &str) {
    use plotters::prelude::*;

    let num_classes = conf.len();

    // Valor máximo fora da diagonal (usado para normalizar a escala de cores dos erros)
    let max_off_diag = conf.iter().enumerate()
        .flat_map(|(i, row)| row.iter().enumerate().filter(move |&(j, _)| i != j).map(|(_, &v)| v))
        .max()
        .unwrap_or(1)
        .max(1);

    // Valor mínimo e máximo na diagonal (para normalizar a escala de acertos)
    let max_diag = (0..num_classes).map(|i| conf[i][i]).max().unwrap_or(1).max(1);

    // Tamanho da imagem: margem para labels + células de 60px cada
    let cell = 60i32;
    let margin = 80i32;
    let img_w = (margin + cell * num_classes as i32 + margin) as u32;
    let img_h = (margin + cell * num_classes as i32 + margin) as u32;

    let root = BitMapBackend::new(output_path, (img_w, img_h)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Título
    root.draw_text(
        "Matriz de Confusão",
        &TextStyle::from(("sans-serif", 22).into_font()).color(&BLACK),
        ((img_w / 2 - 100) as i32, 8),
    ).unwrap();

    // Subtítulo
    root.draw_text(
        "Linhas = Real  |  Colunas = Predito",
        &TextStyle::from(("sans-serif", 13).into_font()).color(&RGBColor(80, 80, 80)),
        ((img_w / 2 - 115) as i32, 34),
    ).unwrap();

    for i in 0..num_classes {
        for j in 0..num_classes {
            let val = conf[i][j];

            // Calcula a cor de fundo da célula por intensidade normalizada
            let bg_color = if i == j {
                // Diagonal: verde mais escuro conforme mais acertos
                let intensity = val as f32 / max_diag as f32;
                let g = (80.0 + 120.0 * (1.0 - intensity)) as u8;  // 80–200
                RGBColor(
                    (230.0 * (1.0 - intensity * 0.6)) as u8,
                    g,
                    (230.0 * (1.0 - intensity * 0.6)) as u8,
                )
            } else if val == 0 {
                // Sem erros: branco puro
                RGBColor(255, 255, 255)
            } else {
                // Erros: vermelho com intensidade proporcional
                let intensity = (val as f32 / max_off_diag as f32).min(1.0);
                RGBColor(
                    255,
                    (255.0 * (1.0 - intensity * 0.85)) as u8,
                    (255.0 * (1.0 - intensity * 0.85)) as u8,
                )
            };

            // Coordenadas da célula em pixels
            let x0 = margin + j as i32 * cell;
            let y0 = margin + i as i32 * cell;
            let x1 = x0 + cell;
            let y1 = y0 + cell;

            // Pinta o fundo da célula
            root.draw(&Rectangle::new(
                [(x0, y0), (x1, y1)],
                bg_color.filled(),
            )).unwrap();

            // Borda da célula
            root.draw(&Rectangle::new(
                [(x0, y0), (x1, y1)],
                RGBColor(180, 180, 180).stroke_width(1),
            )).unwrap();

            // Texto com o valor — preto se claro, branco se célula muito escura
            let text_color = if i == j && val as f32 / max_diag as f32 > 0.6 {
                WHITE
            } else {
                BLACK
            };
            let label = format!("{}", val);
            // Centraliza o texto na célula
            let text_x = x0 + cell / 2 - (label.len() as i32 * 5);
            let text_y = y0 + cell / 2 - 8;
            root.draw_text(
                &label,
                &TextStyle::from(("sans-serif", 14).into_font()).color(&text_color),
                (text_x, text_y),
            ).unwrap();
        }

        // Labels das linhas (Real) — à esquerda
        root.draw_text(
            &format!("{}", i),
            &TextStyle::from(("sans-serif", 15).into_font()).color(&BLACK),
            (margin - 25, margin + i as i32 * cell + cell / 2 - 8),
        ).unwrap();

        // Labels das colunas (Predito) — acima
        root.draw_text(
            &format!("{}", i),
            &TextStyle::from(("sans-serif", 15).into_font()).color(&BLACK),
            (margin + i as i32 * cell + cell / 2 - 5, margin - 25),
        ).unwrap();
    }

    // Legenda dos eixos
    root.draw_text(
        "Real →",
        &TextStyle::from(("sans-serif", 13).into_font()).color(&RGBColor(60, 60, 60)),
        (4, margin + (num_classes as i32 * cell) / 2 - 30),
    ).unwrap();
    root.draw_text(
        "← Predito",
        &TextStyle::from(("sans-serif", 13).into_font()).color(&RGBColor(60, 60, 60)),
        (margin + (num_classes as i32 * cell) / 2 - 35, img_h as i32 - 25),
    ).unwrap();

    root.present().unwrap();
    println!("✅ Matriz de Confusão salva em '{}'", output_path);
}

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
