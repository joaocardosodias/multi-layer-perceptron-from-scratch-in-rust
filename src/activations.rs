pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}
pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        return 1.0;
    } else {
        return 0.0;
    }
}
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
