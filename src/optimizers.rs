use crate::network::Gradients;
pub fn sgd_update(
    weights: &mut Vec<Vec<Vec<f64>>>,
    biases: &mut Vec<Vec<f64>>,
    gradients: &Gradients,
    learning_rate: f64,
) {
    for layer_idx in 0..weights.len() {
        for neuron_idx in 0..weights[layer_idx].len() {
            for w_idx in 0..weights[layer_idx][neuron_idx].len() {
                weights[layer_idx][neuron_idx][w_idx] -=
                    learning_rate * gradients.dw[layer_idx][neuron_idx][w_idx];
            }
            biases[layer_idx][neuron_idx] -= learning_rate * gradients.db[layer_idx][neuron_idx];
        }
    }
}
