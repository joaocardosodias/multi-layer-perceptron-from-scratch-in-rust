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

// ─── Serialização dos pesos ───────────────────────────────────────────────────

/// Salva os pesos e biases da rede em um arquivo binário compacto.
///
/// ## Formato do arquivo (`best_model.bin`)
///
/// ```text
/// [magic : u32  = 0x4D4C5000]   "MLP\0" — identifica o tipo do arquivo
/// [nlayers: u32]                 número de camadas
/// [dims   : (u32, u32) × nlayers]  (n_out, n_in) de cada camada
/// [weights: f32 × total_weights] pesos em ordem flat (mesmo layout do treino)
/// [biases : f32 × total_biases]  biases em ordem flat
/// ```
///
/// Usa `std::io::Write` diretamente em bytes para evitar dependências externas.
pub fn save_weights(mlp: &MLP, path: &str) {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let file = match File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("⚠️  Não foi possível criar '{}': {}", path, e);
            return;
        }
    };
    let mut w = BufWriter::new(file);

    // Magic number
    let _ = w.write_all(&0x4D4C5000u32.to_le_bytes());

    // Número de camadas
    let nlayers = mlp.dims.len() as u32;
    let _ = w.write_all(&nlayers.to_le_bytes());

    // Dimensões de cada camada
    for &(r, c) in &mlp.dims {
        let _ = w.write_all(&(r as u32).to_le_bytes());
        let _ = w.write_all(&(c as u32).to_le_bytes());
    }

    // Pesos (flat)
    for &wv in &mlp.weights {
        let _ = w.write_all(&wv.to_le_bytes());
    }

    // Biases (flat)
    for &bv in &mlp.biases {
        let _ = w.write_all(&bv.to_le_bytes());
    }

    let _ = w.flush();
    println!("💾 Pesos salvos em '{}'", path);
}

/// Carrega os pesos e biases de um arquivo gerado por [`save_weights`] para um `MLP` existente.
///
/// Verifica o magic number e a compatibilidade das dimensões antes de sobrescrever qualquer dado.
/// Se o arquivo não existir ou estiver incompatível, nada é alterado e um aviso é impresso.
pub fn load_weights(mlp: &mut MLP, path: &str) {
    use std::fs;
    use std::io::{Cursor, Read};

    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("⚠️  Não foi possível ler '{}': {}", path, e);
            return;
        }
    };

    let mut cur = Cursor::new(&bytes);
    let mut buf4 = [0u8; 4];

    macro_rules! read_u32 {
        () => {{
            if cur.read_exact(&mut buf4).is_err() { return; }
            u32::from_le_bytes(buf4)
        }};
    }
    macro_rules! read_f32 {
        () => {{
            if cur.read_exact(&mut buf4).is_err() { return; }
            f32::from_le_bytes(buf4)
        }};
    }

    // Verifica magic
    let magic = read_u32!();
    if magic != 0x4D4C5000 {
        eprintln!("⚠️  '{}' não é um arquivo de pesos válido (magic incorreto).", path);
        return;
    }

    // Verifica número de camadas
    let nlayers = read_u32!() as usize;
    if nlayers != mlp.dims.len() {
        eprintln!(
            "⚠️  Incompatibilidade de arquitetura: arquivo tem {} camadas, rede tem {}.",
            nlayers,
            mlp.dims.len()
        );
        return;
    }

    // Verifica dimensões
    for (i, &(r, c)) in mlp.dims.iter().enumerate() {
        let fr = read_u32!() as usize;
        let fc = read_u32!() as usize;
        if fr != r || fc != c {
            eprintln!(
                "⚠️  Dimensão incompatível na camada {}: arquivo=({},{}) rede=({},{}).",
                i, fr, fc, r, c
            );
            return;
        }
    }

    // Carrega pesos
    for wv in &mut mlp.weights {
        *wv = read_f32!();
    }

    // Carrega biases
    for bv in &mut mlp.biases {
        *bv = read_f32!();
    }

    println!("✅ Melhor modelo carregado de '{}'", path);
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

/// Visualiza os pesos da primeira camada como uma grade de imagens 28×28.
///
/// Cada neurônio da primeira camada tem um vetor de pesos de 784 dimensões
/// (mesmo shape da imagem MNIST 28×28). Visualizar esses pesos revela quais
/// padrões cada neurônio aprendeu a detectar na imagem de entrada.
///
/// **Colormap divergente:** azul = peso muito negativo, branco = zero, vermelho = peso positivo.
///
/// Disponível apenas com `--features auto-plot`.
#[cfg(feature = "auto-plot")]
pub fn plot_weight_grid(mlp: &MLP, output_path: &str) {
    use plotters::prelude::*;

    let n_in = mlp.dims[0].1;
    let n_out = mlp.dims[0].0;

    if n_in != 784 {
        println!("⚠️  plot_weight_grid: esperava n_in=784, encontrou {n_in}. Pulando.");
        return;
    }

    // Mostra até 64 neurônios em uma grade 8×8
    let max_neurons = 64.min(n_out);
    let grid_cols = 8usize;
    let grid_rows = (max_neurons + grid_cols - 1) / grid_cols;

    let tile = 28i32;     // cada neurônio = 28×28 px
    let gap = 4i32;       // espaço entre tiles
    let margin_h = 50i32; // margem horizontal
    let title_h = 44i32;  // altura do bloco de título
    let margin_b = 20i32;

    let img_w = (margin_h + (tile + gap) * grid_cols as i32 + margin_h) as u32;
    let img_h = (title_h + (tile + gap) * grid_rows as i32 + margin_b) as u32;

    let root = BitMapBackend::new(output_path, (img_w, img_h)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    root.draw_text(
        "Pesos da Primeira Camada",
        &TextStyle::from(("sans-serif", 19).into_font()).color(&BLACK),
        (margin_h, 5),
    ).unwrap();
    root.draw_text(
        &format!(
            "{} de {} neurônios  ·  28×28 px por tile  ·  azul=neg  branco=0  vermelho=pos",
            max_neurons, n_out
        ),
        &TextStyle::from(("sans-serif", 11).into_font()).color(&RGBColor(100, 100, 100)),
        (margin_h, 27),
    ).unwrap();

    let w_all = &mlp.weights[mlp.w_offsets[0]..mlp.w_offsets[1]];

    for idx in 0..max_neurons {
        let row = idx / grid_cols;
        let col = idx % grid_cols;
        let x0 = margin_h + col as i32 * (tile + gap);
        let y0 = title_h + row as i32 * (tile + gap);

        let w = &w_all[idx * n_in..(idx + 1) * n_in];

        // Normaliza pelo valor absoluto máximo → [-1, 1]
        let abs_max = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-8);

        for py in 0..28usize {
            for px in 0..28usize {
                let v = w[py * 28 + px] / abs_max; // em [-1, 1]
                let t = (v + 1.0) * 0.5;            // em [0, 1]

                // Divergente: azul → branco → vermelho
                let (r, g, b) = if t < 0.5 {
                    let s = 1.0 - t * 2.0;
                    ((255.0 * (1.0 - s * 0.85)) as u8, (255.0 * (1.0 - s * 0.85)) as u8, 255u8)
                } else {
                    let s = (t - 0.5) * 2.0;
                    (255u8, (255.0 * (1.0 - s * 0.85)) as u8, (255.0 * (1.0 - s * 0.85)) as u8)
                };

                root.draw(&Rectangle::new(
                    [(x0 + px as i32, y0 + py as i32), (x0 + px as i32 + 1, y0 + py as i32 + 1)],
                    RGBColor(r, g, b).filled(),
                )).unwrap();
            }
        }
    }

    root.present().unwrap();
    println!("✅ Grade de pesos salva em '{}'", output_path);
}

/// Gera um heatmap da ativação média de cada neurônio da última camada oculta,
/// separado por classe de dígito (0–9).
///
/// Para cada classe, percorre todas as imagens de teste daquela classe e calcula
/// a ativação média do último bloco ReLU (camada antes do softmax).
/// Em seguida, seleciona os **top-K neurônios com maior variância entre classes**
/// — esses são os mais discriminativos — e os exibe como heatmap.
///
/// **Leitura do heatmap:**
/// - Linhas = dígito real (0 a 9)
/// - Colunas = neurônio selecionado (ordenado por variância decrescente)
/// - Cor mais escura (roxo/azul) = neurônio mais ativo para aquela classe
/// - Uma coluna com padrão claro/escuro forte = neurônio que distingue classes
///
/// Disponível apenas com `--features auto-plot`.
#[cfg(feature = "auto-plot")]
pub fn plot_class_activations(
    mlp: &MLP,
    images: &[f32],
    num_images: usize,
    labels: &[usize],
    output_path: &str,
) {
    use plotters::prelude::*;

    let num_classes = 10;
    let num_layers = mlp.dims.len();

    // Última camada oculta = penúltima entrada do cache (antes do softmax)
    let hidden_dim = mlp.dims[num_layers - 2].0;

    // Acumuladores: soma das ativações e contagem por classe
    let mut acc = vec![vec![0.0f64; hidden_dim]; num_classes];
    let mut count = vec![0usize; num_classes];

    let eval_bs = 256;
    let mut cache = BatchCache::new(&mlp.dims, eval_bs);
    let mut batch_input = vec![0.0f32; eval_bs * 784];
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    for chunk_start in (0..num_images).step_by(eval_bs) {
        let chunk_end = (chunk_start + eval_bs).min(num_images);
        let bs = chunk_end - chunk_start;

        for k in 0..bs {
            let i = chunk_start + k;
            batch_input[k * 784..(k + 1) * 784]
                .copy_from_slice(&images[i * 784..(i + 1) * 784]);
        }

        mlp.forward_batch(&batch_input, &mut cache, bs, false, &mut rng);

        // cache.a_offsets[num_layers - 1] = início das ativações da última camada oculta
        let a_off = cache.a_offsets[num_layers - 1];

        for k in 0..bs {
            let label = labels[chunk_start + k];
            let off = k * hidden_dim;
            for j in 0..hidden_dim {
                acc[label][j] += cache.activations[a_off + off + j] as f64;
            }
            count[label] += 1;
        }
    }

    // Média por classe
    let means: Vec<Vec<f32>> = (0..num_classes)
        .map(|c| {
            let n = count[c].max(1) as f64;
            acc[c].iter().map(|&s| (s / n) as f32).collect()
        })
        .collect();

    // Seleciona top-K neurônios por variância entre classes (os mais discriminativos)
    let top_k = 80.min(hidden_dim);
    let mut neuron_vars: Vec<(f32, usize)> = (0..hidden_dim)
        .map(|j| {
            let mu = means.iter().map(|m| m[j]).sum::<f32>() / num_classes as f32;
            let var = means.iter().map(|m| (m[j] - mu).powi(2)).sum::<f32>() / num_classes as f32;
            (var, j)
        })
        .collect();
    neuron_vars.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top_neurons: Vec<usize> = neuron_vars.iter().take(top_k).map(|&(_, j)| j).collect();

    // Monta a matriz de exibição e normaliza cada coluna para [0, 1]
    let mut mat: Vec<Vec<f32>> = (0..num_classes)
        .map(|c| top_neurons.iter().map(|&j| means[c][j]).collect())
        .collect();

    for ki in 0..top_k {
        let col_max = mat.iter().map(|r| r[ki]).fold(0.0f32, f32::max).max(1e-8);
        for c in 0..num_classes {
            mat[c][ki] /= col_max;
        }
    }

    // ── Renderização ─────────────────────────────────────────────────────────
    let cell_w = 9i32;
    let cell_h = 32i32;
    let margin_l = 55i32;
    let margin_t = 58i32;
    let margin_r = 20i32;
    let margin_b = 30i32;

    let img_w = (margin_l + cell_w * top_k as i32 + margin_r) as u32;
    let img_h = (margin_t + cell_h * num_classes as i32 + margin_b) as u32;

    let root = BitMapBackend::new(output_path, (img_w, img_h)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    root.draw_text(
        "Ativações Médias por Classe — Última Camada Oculta",
        &TextStyle::from(("sans-serif", 17).into_font()).color(&BLACK),
        (margin_l, 5),
    ).unwrap();
    root.draw_text(
        &format!(
            "Top-{top_k} neurônios mais discriminativos  ·  linhas=dígito  ·  mais escuro=mais ativo"
        ),
        &TextStyle::from(("sans-serif", 11).into_font()).color(&RGBColor(100, 100, 100)),
        (margin_l, 27),
    ).unwrap();
    root.draw_text(
        "Neurônios →",
        &TextStyle::from(("sans-serif", 11).into_font()).color(&RGBColor(80, 80, 80)),
        (margin_l, margin_t - 14),
    ).unwrap();

    for c in 0..num_classes {
        let y0 = margin_t + c as i32 * cell_h;

        // Label da linha
        root.draw_text(
            &format!("  {c}"),
            &TextStyle::from(("sans-serif", 15).into_font()).color(&BLACK),
            (2, y0 + cell_h / 2 - 9),
        ).unwrap();

        for ki in 0..top_k {
            let x0 = margin_l + ki as i32 * cell_w;
            let v = mat[c][ki]; // [0, 1]

            // Colormap: branco → roxo escuro (inspirado em "plasma")
            let r = (255.0 * (1.0 - v * 0.72)) as u8;
            let g = (255.0 * (1.0 - v * 0.82)) as u8;
            let b = (255.0 * (1.0 - v * 0.30)) as u8;

            root.draw(&Rectangle::new(
                [(x0, y0), (x0 + cell_w, y0 + cell_h)],
                RGBColor(r, g, b).filled(),
            )).unwrap();

            // Borda sutil entre células
            root.draw(&Rectangle::new(
                [(x0, y0), (x0 + cell_w, y0 + cell_h)],
                RGBColor(220, 220, 220).stroke_width(1),
            )).unwrap();
        }
    }

    root.present().unwrap();
    println!("✅ Mapa de ativações por classe salvo em '{}'", output_path);
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
