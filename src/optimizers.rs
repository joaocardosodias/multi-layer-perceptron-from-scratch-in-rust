use crate::network::Gradients;

pub fn sgd_update(
    weights: &mut Vec<Vec<f32>>,
    biases: &mut Vec<Vec<f32>>,
    gradients: &Gradients,
    learning_rate: f32,
) {
    for layer_idx in 0..weights.len() {
        let w = &mut weights[layer_idx];
        let b = &mut biases[layer_idx];
        let dw = &gradients.dw[layer_idx];
        let db = &gradients.db[layer_idx];

        for i in 0..w.len() {
            w[i] -= learning_rate * dw[i];
        }
        for i in 0..b.len() {
            b[i] -= learning_rate * db[i];
        }
    }
}
