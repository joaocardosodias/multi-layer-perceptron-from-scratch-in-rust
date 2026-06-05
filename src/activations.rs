#[inline]
pub fn relu(x: f32) -> f32 { x.max(0.0) }

pub fn relu_derivative(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

pub fn softmax_into(logits: &[f32], out: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for i in 0..logits.len() {
        let e = (logits[i] - max).exp();
        out[i] = e;
        sum += e;
    }
    let inv_sum = 1.0 / sum;
    for i in 0..out.len() {
        out[i] *= inv_sum;
    }
}
