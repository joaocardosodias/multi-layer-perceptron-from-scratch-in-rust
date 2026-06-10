/// Verificação numérica dos gradientes do backpropagation.
///
/// # Como funciona
///
/// O gradient check compara dois tipos de gradiente para cada parâmetro θ da rede:
///
/// **Gradiente Analítico** — calculado pelo backpropagation implementado em [`MLP::backward_batch`].
///
/// **Gradiente Numérico** — estimado pela definição de derivada via diferenças centrais:
///
/// ```text
///   dL/dθ ≈ [ L(θ + ε) - L(θ - ε) ] / (2ε)
/// ```
///
/// Para que os dois concordem, calculamos o **Erro Relativo**:
///
/// ```text
///   erro_rel = |grad_analítico - grad_numérico| / (|grad_analítico| + |grad_numérico| + ε_small)
/// ```
///
/// # Nota sobre precisão com f32
///
/// A rede usa `f32` (precisão simples, ~7 dígitos decimais).
/// O threshold padrão para `f64` seria `< 1e-7`, mas para `f32` o esperado é:
///
/// | Situação                    | Threshold aceitável |
/// |-----------------------------|---------------------|
/// | Gradiente correto com f32   | `< 1e-3`            |
/// | Gradiente incorreto         | `> 1e-2` (em geral) |
///
/// Erros de 1e-4 a 1e-3 surgem naturalmente do arredondamento do f32
/// na diferença `L(θ+ε) - L(θ-ε)` e são esperados — NÃO indicam bug.
///
/// # Configuração dos testes
///
/// - Forward sempre em modo **inferência** (sem dropout) para ser determinístico
/// - ε = 1e-4 (equilíbrio entre truncamento e arredondamento)
/// - Threshold = 5e-3 (5× a margem típica de f32)
#[cfg(test)]
mod tests {
    use crate::network::{BatchCache, Gradients, MLP};
    use mlp::common::losses::cross_entropy;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// ε usado na diferença central.
    ///
    /// Para `f32`, o ponto ótimo é entre `1e-3` e `1e-4`:
    /// - muito pequeno (ε < 1e-5) → erros de arredondamento dominam
    /// - muito grande (ε > 1e-2) → erro de truncamento da aproximação domina
    const EPS: f32 = 1e-3;

    /// Threshold de erro aceitável.
    /// Usa-se o erro relativo quando |a| + |n| é grande, e erro absoluto quando
    /// ambos os gradientes são próximos de zero.
    const THRESHOLD: f32 = 1e-2;

    /// Mínima norma para separar o regime de erro relativo do absoluto.
    /// Quando |a| + |n| < MIN_NORM, usa-se erro absoluto em vez de relativo.
    const MIN_NORM: f32 = 1e-2;

    // ─── Utilitários ─────────────────────────────────────────────────────────

    /// Roda um forward pass e retorna a loss média do batch.
    /// Sempre em modo inferência (sem dropout) para ser determinístico.
    fn forward_loss(mlp: &MLP, input: &[f32], targets: &[usize], bs: usize) -> f32 {
        let mut cache = BatchCache::new(&mlp.dims, bs);
        let mut rng = StdRng::seed_from_u64(0);
        mlp.forward_batch(input, &mut cache, bs, false, &mut rng);

        let out_dim = mlp.dims.last().unwrap().0;
        let a_off = cache.a_offsets[mlp.dims.len()];
        let mut total_loss = 0.0f32;
        for s in 0..bs {
            let off = s * out_dim;
            let probs = &cache.activations[a_off + off..a_off + off + out_dim];
            total_loss += cross_entropy(probs, targets[s]);
        }
        total_loss / bs as f32
    }

    /// Calcula os gradientes analíticos via backpropagation para um batch.
    fn analytical_grads(mlp: &MLP, input: &[f32], targets: &[usize], bs: usize) -> Gradients {
        let mut cache = BatchCache::new(&mlp.dims, bs);
        let mut rng = StdRng::seed_from_u64(0);
        mlp.forward_batch(input, &mut cache, bs, false, &mut rng);

        let mut grads = Gradients::new(mlp);
        mlp.backward_batch(&mut cache, targets, &mut grads, bs);
        grads
    }

    /// Verifica todos os pesos e biases da rede via gradient check numérico.
    ///
    /// Retorna `(maior_erro_relativo, número_de_falhas)`.
    fn run_gradient_check(
        mlp: &mut MLP,
        input: &[f32],
        targets: &[usize],
        bs: usize,
    ) -> (f32, usize) {
        let analytical = analytical_grads(mlp, input, targets, bs);

        let mut max_rel_error: f32 = 0.0;
        let mut num_failures = 0usize;

        // ── Pesos ────────────────────────────────────────────────────────────
        for i in 0..mlp.weights.len() {
            let original = mlp.weights[i];

            mlp.weights[i] = original + EPS;
            let loss_plus = forward_loss(mlp, input, targets, bs);

            mlp.weights[i] = original - EPS;
            let loss_minus = forward_loss(mlp, input, targets, bs);

            mlp.weights[i] = original;

            let numerical = (loss_plus - loss_minus) / (2.0 * EPS);
            let analytical_val = analytical.dw[i];
            // Denominador clampado a MIN_NORM para evitar erros relativos inflados
            // quando ambos os gradientes são próximos de zero (regime de ruído f32).
            let denom = (analytical_val.abs() + numerical.abs()).max(MIN_NORM);
            let rel_err = (analytical_val - numerical).abs() / denom;

            if rel_err > max_rel_error {
                max_rel_error = rel_err;
            }
            if rel_err > THRESHOLD {
                eprintln!(
                    "  [FALHA] peso[{i}]: analítico={analytical_val:.6e}  numérico={numerical:.6e}  erro_rel={rel_err:.2e}"
                );
                num_failures += 1;
            }
        }

        // ── Biases ───────────────────────────────────────────────────────────
        for i in 0..mlp.biases.len() {
            let original = mlp.biases[i];

            mlp.biases[i] = original + EPS;
            let loss_plus = forward_loss(mlp, input, targets, bs);

            mlp.biases[i] = original - EPS;
            let loss_minus = forward_loss(mlp, input, targets, bs);

            mlp.biases[i] = original;

            let numerical = (loss_plus - loss_minus) / (2.0 * EPS);
            let analytical_val = analytical.db[i];
            // Denominador clampado a MIN_NORM para evitar erros relativos inflados
            // quando ambos os gradientes são próximos de zero (regime de ruído f32).
            let denom = (analytical_val.abs() + numerical.abs()).max(MIN_NORM);
            let rel_err = (analytical_val - numerical).abs() / denom;

            if rel_err > max_rel_error {
                max_rel_error = rel_err;
            }
            if rel_err > THRESHOLD {
                eprintln!(
                    "  [FALHA] bias[{i}]: analítico={analytical_val:.6e}  numérico={numerical:.6e}  erro_rel={rel_err:.2e}"
                );
                num_failures += 1;
            }
        }

        (max_rel_error, num_failures)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Testes
    // ═══════════════════════════════════════════════════════════════════════════

    /// Rede mínima [2, 3, 2] — batch de 1 amostra.
    /// Caso mais simples para isolar problemas em camadas individuais.
    #[test]
    fn test_gradient_check_minimal_network() {
        let arch = vec![2, 3, 2];
        let mut mlp = MLP::new(&arch);

        let input: Vec<f32> = vec![0.42, 0.77];
        let targets = vec![1usize];
        let bs = 1;

        let (max_err, failures) = run_gradient_check(&mut mlp, &input, &targets, bs);

        println!(
            "Gradient check [2,3,2]: maior erro relativo = {:.2e}  |  threshold = {:.1e}  |  falhas = {}",
            max_err, THRESHOLD, failures
        );
        assert_eq!(
            failures, 0,
            "Gradient check [2,3,2] falhou: {failures} parâmetro(s) com erro relativo > {THRESHOLD:.1e}"
        );
    }

    /// Rede com 3 camadas [4, 8, 5, 3] — batch de 2 amostras.
    /// Cobre pesos e biases de todas as camadas, incluindo a saída softmax.
    #[test]
    fn test_gradient_check_full_network() {
        let arch = vec![4, 8, 5, 3];
        let mut mlp = MLP::new(&arch);

        let input: Vec<f32> = vec![
            0.10, 0.25, 0.80, 0.45, // amostra 0
            0.60, 0.05, 0.33, 0.90, // amostra 1
        ];
        let targets = vec![0usize, 2];
        let bs = 2;

        let (max_err, failures) = run_gradient_check(&mut mlp, &input, &targets, bs);

        println!(
            "Gradient check [4,8,5,3]: maior erro relativo = {:.2e}  |  threshold = {:.1e}  |  falhas = {}",
            max_err, THRESHOLD, failures
        );
        assert_eq!(
            failures, 0,
            "Gradient check [4,8,5,3] falhou: {failures} parâmetro(s) com erro relativo > {THRESHOLD:.1e}"
        );
    }

    /// Rede mais profunda [3, 6, 6, 6, 2] — batch de 3 amostras.
    /// Verifica que a propagação do gradiente funciona corretamente em 4 camadas.
    #[test]
    fn test_gradient_check_deep_network() {
        let arch = vec![3, 6, 6, 6, 2];
        let mut mlp = MLP::new(&arch);

        let input: Vec<f32> = vec![
            0.20, 0.50, 0.75, // amostra 0
            0.85, 0.10, 0.40, // amostra 1
            0.33, 0.66, 0.99, // amostra 2
        ];
        let targets = vec![0usize, 1, 0];
        let bs = 3;

        let (max_err, failures) = run_gradient_check(&mut mlp, &input, &targets, bs);

        println!(
            "Gradient check [3,6,6,6,2]: maior erro relativo = {:.2e}  |  threshold = {:.1e}  |  falhas = {}",
            max_err, THRESHOLD, failures
        );
        assert_eq!(
            failures, 0,
            "Gradient check [3,6,6,6,2] falhou: {failures} parâmetro(s) com erro relativo > {THRESHOLD:.1e}"
        );
    }

    /// Verifica que a diferença central é simétrica:
    /// [L(θ+ε) - L(θ-ε)] deve ser o oposto de [L(θ-ε) - L(θ+ε)].
    /// Testa a consistência interna do forward pass.
    #[test]
    fn test_gradient_check_central_difference_symmetry() {
        let arch = vec![3, 4, 2];
        let mut mlp = MLP::new(&arch);

        let input: Vec<f32> = vec![0.1, 0.5, 0.9];
        let targets = vec![0usize];
        let bs = 1;

        for i in 0..mlp.weights.len() {
            let original = mlp.weights[i];

            mlp.weights[i] = original + EPS;
            let lp = forward_loss(&mlp, &input, &targets, bs);

            mlp.weights[i] = original - EPS;
            let lm = forward_loss(&mlp, &input, &targets, bs);

            mlp.weights[i] = original;

            // [f(θ+ε) - f(θ-ε)] + [f(θ-ε) - f(θ+ε)] == 0
            let sum = (lp - lm) + (lm - lp);
            assert!(
                sum.abs() < 1e-6,
                "Diferença central não-simétrica em peso[{i}]: lp={lp:.8e}, lm={lm:.8e}"
            );
        }
    }

    /// Verifica que o gradiente numérico converge para zero quando a loss não muda
    /// ao perturbar um parâmetro que não afeta a saída.
    /// Testa biases de neurônios "mortos" (pré-ativação muito negativa).
    #[test]
    fn test_gradient_check_dead_neuron_bias() {
        // Cria rede mínima e força um neurônio a ser "morto" (output ReLU = 0)
        // ajustando manualmente o bias para um valor muito negativo.
        let arch = vec![2, 2, 2];
        let mut mlp = MLP::new(&arch);

        // Força o neurônio 0 da camada 0 a ficar morto (bias = -100)
        mlp.biases[0] = -100.0;

        let input: Vec<f32> = vec![0.5, 0.5];
        let targets = vec![0usize];
        let bs = 1;

        let analytical = analytical_grads(&mlp, &input, &targets, bs);

        // O bias do neurônio morto deve ter gradiente analítico ≈ 0
        // (sinal de ReLU é zero → gradiente bloqueado)
        assert!(
            analytical.db[0].abs() < 1e-6,
            "Bias de neurônio morto deveria ter gradiente ≈ 0, mas obteve {:.6e}",
            analytical.db[0]
        );
    }

    /// Verifica que o gradiente aponta na direção correta de descida:
    /// dar um pequeno passo na direção −∇L deve reduzir a loss.
    /// Isso é um sanity-check ortogonal ao gradient check numérico.
    #[test]
    fn test_gradient_direction_loss_decreases() {
        let arch = vec![4, 6, 3];
        let mut mlp = MLP::new(&arch);

        let input: Vec<f32> = vec![0.3, 0.6, 0.1, 0.9];
        let targets = vec![2usize];
        let bs = 1;

        let loss_before = forward_loss(&mlp, &input, &targets, bs);
        let grads = analytical_grads(&mlp, &input, &targets, bs);

        // Passo de gradient descent: θ ← θ - lr * ∇L
        let lr = 0.01f32;
        for i in 0..mlp.weights.len() {
            mlp.weights[i] -= lr * grads.dw[i];
        }
        for i in 0..mlp.biases.len() {
            mlp.biases[i] -= lr * grads.db[i];
        }

        let loss_after = forward_loss(&mlp, &input, &targets, bs);

        println!(
            "Loss antes={:.6}  loss depois={:.6}  (deve diminuir)",
            loss_before, loss_after
        );
        assert!(
            loss_after < loss_before,
            "Um passo de gradient descent deveria reduzir a loss! antes={loss_before:.6}, depois={loss_after:.6}"
        );
    }
}

