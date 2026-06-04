use crate::activations::*;
use crate::linalg::{mul_matrix_vec, add_vec_vec, outer_product, transpose, mul_vec_vec};

static mut SEED: u64 = 42;
fn rand_uniform() -> f64 {
    unsafe {
        SEED = SEED
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((SEED >> 33) as f64) / (u32::MAX as f64)
    }
}
fn rand_normal() -> f64 {
    let mut sum = 0.0;
    for _ in 0..12 {
        sum += rand_uniform();
    }
    sum - 6.0
}

pub struct MLP {
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
}

pub struct ForwardCache {
    pub pre_activations: Vec<Vec<f64>>,
    pub activations: Vec<Vec<f64>>,
}

impl ForwardCache {
    pub fn new() -> Self {
        ForwardCache {
            pre_activations: Vec::new(),
            activations: Vec::new(),
        }
    }
}

pub struct Gradients {
    pub dw: Vec<Vec<Vec<f64>>>,
    pub db: Vec<Vec<f64>>,
}

impl MLP {
    pub fn new(architecture: &[usize]) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for i in 0..(architecture.len() - 1) {
            let n_in = architecture[i];
            let n_out = architecture[i + 1];
            let std_dev = (2.0 / n_in as f64).sqrt();
            let mut layer_w = Vec::with_capacity(n_out);
            for _ in 0..n_out {
                let mut neuron_w = Vec::with_capacity(n_in);
                for _ in 0..n_in {
                    neuron_w.push(rand_normal() * std_dev);
                }
                layer_w.push(neuron_w);
            }
            weights.push(layer_w);
            biases.push(vec![0.0; n_out]);
        }
        MLP { weights, biases }
    }

    pub fn forward(&self, start_input: &[f64]) -> (Vec<f64>, ForwardCache) {
        let mut cache = ForwardCache::new();
        cache.activations.push(start_input.to_vec());
        for i in 0..self.weights.len() {
            let input = &cache.activations[i];
            let z = add_vec_vec(&mul_matrix_vec(&self.weights[i], input), &self.biases[i]);
            cache.pre_activations.push(z.clone());
            let a = if i == self.weights.len() - 1 {
                softmax(&z)
            } else {
                relu_vec(&z)
            };
            cache.activations.push(a);
        }
        (cache.activations.last().unwrap().clone(), cache)
    }

    pub fn backward(&self, cache: &ForwardCache, probs: &[f64], target: usize) -> Gradients {
        let num_layers = self.weights.len();
        let mut dw = Vec::with_capacity(num_layers);
        let mut db = Vec::with_capacity(num_layers);

        let mut delta = probs.to_vec();
        delta[target] -= 1.0;

        for l in (0..num_layers).rev() {
            let a_prev = &cache.activations[l];
            dw.push(outer_product(&delta, a_prev));
            db.push(delta.clone());

            if l > 0 {
                let w_t = transpose(&self.weights[l]);
                let delta_propagated = mul_matrix_vec(&w_t, &delta);
                let z_prev = &cache.pre_activations[l - 1];
                let relu_derivs: Vec<f64> = z_prev.iter().map(|&z| relu_derivative(z)).collect();
                delta = mul_vec_vec(&delta_propagated, &relu_derivs);
            }
        }
        
        dw.reverse();
        db.reverse();

        Gradients { dw, db }
    }
}
