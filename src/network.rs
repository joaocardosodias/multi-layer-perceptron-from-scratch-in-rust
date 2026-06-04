use crate::linalg::*;
use crate::activations::*;

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
impl ForwardCache{
  pub fn new()->Self{
    ForwardCache { pre_activations: Vec::new(), activations: Vec::new() }
  }
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
    pub fn forward(&self,input:&[f64])->(Vec<f64>,ForwardCache){
      let mut cache=ForwardCache::new();
      
    }
}
