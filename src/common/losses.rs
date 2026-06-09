pub fn cross_entropy(probs: &[f32], target_class: usize) -> f32 {
    let p = probs[target_class];
    let p_safe = p.max(1e-10).min(1.0);
    -p_safe.ln()
}
