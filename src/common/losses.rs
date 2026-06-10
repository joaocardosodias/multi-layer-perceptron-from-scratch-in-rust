/// Calcula a perda de entropia cruzada (Cross-Entropy Loss) para uma única amostra.
/// Recebe as `probs` (probabilidades preditas para cada classe) e `target_class` (o índice da classe verdadeira).
/// Adiciona um pequeno valor (`1e-10`) para evitar o cálculo do logaritmo de zero.
pub fn cross_entropy(probs: &[f32], target_class: usize) -> f32 {
    let p = probs[target_class];
    let p_safe = p.max(1e-10).min(1.0);
    -p_safe.ln()
}
