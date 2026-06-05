use crate::network::MLP;
use crate::losses::cross_entropy;

pub fn shuffle(indices: &mut [usize]) {
    use std::time::SystemTime;
    let mut seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    for i in (1..indices.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((seed >> 33) as usize) % (i + 1);
        indices.swap(i, j);
    }
}

pub fn argmax(v: &[f64]) -> usize {
    let mut max_idx = 0;
    for i in 1..v.len() {
        if v[i] > v[max_idx] {
            max_idx = i;
        }
    }
    max_idx
}

pub fn zero_gradients_dw(weights: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
    weights.iter().map(|layer| {
        layer.iter().map(|neuron| {
            vec![0.0; neuron.len()]
        }).collect()
    }).collect()
}

pub fn zero_gradients_db(biases: &[Vec<f64>]) -> Vec<Vec<f64>> {
    biases.iter().map(|layer| vec![0.0; layer.len()]).collect()
}

pub fn accumulate_gradients(acc: &mut Vec<Vec<Vec<f64>>>, grads: &[Vec<Vec<f64>>]) {
    for (a_layer, g_layer) in acc.iter_mut().zip(grads.iter()) {
        for (a_neuron, g_neuron) in a_layer.iter_mut().zip(g_layer.iter()) {
            for (a_w, g_w) in a_neuron.iter_mut().zip(g_neuron.iter()) {
                *a_w += g_w;
            }
        }
    }
}

pub fn accumulate_gradients_db(acc: &mut Vec<Vec<f64>>, grads: &[Vec<f64>]) {
    for (a_layer, g_layer) in acc.iter_mut().zip(grads.iter()) {
        for (a_b, g_b) in a_layer.iter_mut().zip(g_layer.iter()) {
            *a_b += g_b;
        }
    }
}

pub fn scale_gradients(grads: &mut Vec<Vec<Vec<f64>>>, scale: f64) {
    for layer in grads.iter_mut() {
        for neuron in layer.iter_mut() {
            for w in neuron.iter_mut() {
                *w *= scale;
            }
        }
    }
}

pub fn scale_gradients_db(grads: &mut Vec<Vec<f64>>, scale: f64) {
    for layer in grads.iter_mut() {
        for b in layer.iter_mut() {
            *b *= scale;
        }
    }
}

pub fn evaluate(mlp: &MLP, images: &[Vec<f64>], labels: &[usize]) -> (f64, f64) {
    let mut correct = 0;
    let mut total_loss = 0.0;

    for (img, label) in images.iter().zip(labels.iter()) {
        let (probs, _) = mlp.forward(img);
        if argmax(&probs) == *label {
            correct += 1;
        }
        total_loss += cross_entropy(&probs, *label);
    }

    (correct as f64 / images.len() as f64, total_loss / images.len() as f64)
}
