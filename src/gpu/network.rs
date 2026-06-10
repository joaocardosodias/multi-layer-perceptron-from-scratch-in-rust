use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::sync::Arc;

use crate::error::GpuError;
use crate::kernels::*;
use crate::linalg::BlasHandle;

/// Estrutura principal da rede neural Multi-Layer Perceptron na GPU.
/// Armazena os pesos e vieses diretamente na memória do dispositivo (VRAM) via `CudaSlice`.
#[allow(dead_code)]
pub struct MLP {
    pub weights: CudaSlice<f32>,
    pub w_offsets: Vec<usize>,
    pub biases: CudaSlice<f32>,
    pub b_offsets: Vec<usize>,
    pub dims: Vec<(usize, usize)>,
    pub dev: Arc<CudaDevice>,
}

/// Cache de tensores intermediários pré-alocados na GPU para processamento de batches.
/// Mantém pre-ativações (Z), ativações (A) e erros propagados (deltas) na VRAM,
/// evitando overhead de alocação dinâmica durante o loop de treinamento rápido.
pub struct BatchCache {
    pub pre_activations: CudaSlice<f32>,
    pub p_offsets: Vec<usize>,
    pub activations: CudaSlice<f32>,
    pub a_offsets: Vec<usize>,
    pub deltas: Vec<CudaSlice<f32>>,
}

impl BatchCache {
    /// Pre-aloca a memória necessária na GPU com base na arquitetura da rede e tamanho do batch.
    pub fn new(
        dev: &Arc<CudaDevice>,
        dims: &[(usize, usize)],
        batch_size: usize,
    ) -> Result<Self, GpuError> {
        let mut pre_activations_data = Vec::new();
        let mut p_offsets = Vec::new();
        let mut activations_data = Vec::new();
        let mut a_offsets = Vec::new();
        let mut deltas = Vec::new();

        a_offsets.push(0);
        activations_data.resize(batch_size * dims[0].1, 0.0);

        for &(r, _) in dims {
            p_offsets.push(pre_activations_data.len());
            pre_activations_data.resize(pre_activations_data.len() + batch_size * r, 0.0);

            a_offsets.push(activations_data.len());
            activations_data.resize(activations_data.len() + batch_size * r, 0.0);

            deltas.push(dev.alloc_zeros::<f32>(batch_size * r)?);
        }
        p_offsets.push(pre_activations_data.len());
        a_offsets.push(activations_data.len());

        let pre_activations = dev.htod_sync_copy(&pre_activations_data)?;
        let activations = dev.htod_sync_copy(&activations_data)?;

        Ok(BatchCache {
            pre_activations,
            p_offsets,
            activations,
            a_offsets,
            deltas,
        })
    }
}

/// Mantém os tensores de gradientes acumulados para pesos e vieses alocados na GPU.
pub struct Gradients {
    pub dw: CudaSlice<f32>,
    pub w_offsets: Vec<usize>,
    pub db: CudaSlice<f32>,
    pub b_offsets: Vec<usize>,
    pub dev: Arc<CudaDevice>,
}

impl Gradients {
    /// Instancia a estrutura alocando tensores zerados (`alloc_zeros`) na VRAM.
    pub fn new(dev: &Arc<CudaDevice>, mlp: &MLP) -> Result<Self, GpuError> {
        let dw = dev.alloc_zeros::<f32>(mlp.weights.len())?;
        let db = dev.alloc_zeros::<f32>(mlp.biases.len())?;
        Ok(Gradients {
            dw,
            w_offsets: mlp.w_offsets.clone(),
            db,
            b_offsets: mlp.b_offsets.clone(),
            dev: dev.clone(),
        })
    }

    /// Preenche rapidamente a memória dos tensores de gradiente com zeros na GPU.
    pub fn zero(&mut self) -> Result<(), GpuError> {
        self.dev.memset_zeros(&mut self.dw)?;
        self.dev.memset_zeros(&mut self.db)?;
        Ok(())
    }
}

impl MLP {
    /// Constrói a MLP, inicializando os pesos na RAM do host usando distribuição Normal
    /// e em seguida realizando a cópia (Host-to-Device) para a VRAM da placa de vídeo.
    pub fn new(dev: &Arc<CudaDevice>, architecture: &[usize]) -> Result<Self, GpuError> {
        let mut weights = Vec::new();
        let mut w_offsets = Vec::new();
        let mut biases = Vec::new();
        let mut b_offsets = Vec::new();
        let mut dims = Vec::new();

        let mut rng = StdRng::seed_from_u64(42);

        for i in 0..(architecture.len() - 1) {
            let n_in = architecture[i];
            let n_out = architecture[i + 1];
            let std_dev = (2.0 / n_in as f32).sqrt();
            let normal = Normal::new(0.0, std_dev).unwrap();

            w_offsets.push(weights.len());
            for _ in 0..(n_out * n_in) {
                weights.push(normal.sample(&mut rng) as f32);
            }

            b_offsets.push(biases.len());
            for _ in 0..n_out {
                biases.push(0.0f32);
            }

            dims.push((n_out, n_in));
        }
        w_offsets.push(weights.len());
        b_offsets.push(biases.len());

        let weights = dev.htod_sync_copy(&weights)?;
        let biases = dev.htod_sync_copy(&biases)?;

        Ok(MLP {
            weights,
            w_offsets,
            biases,
            b_offsets,
            dims,
            dev: dev.clone(),
        })
    }

    /// Processa o Forward Pass de um batch puramente na GPU.
    /// Combina operações super otimizadas da NVIDIA (cuBLAS `gemm_tb`) com kernels CUDA customizados
    /// para adição de bias (`bias_add`), ativações (`relu`, `softmax`) e regularização (`dropout`).
    pub fn forward_batch(
        &self,
        input: &cudarc::driver::CudaView<f32>,
        cache: &mut BatchCache,
        bs: usize,
        is_training: bool,
        kernels: &Kernels,
        blas: &BlasHandle,
        dropout_keep: f32,
    ) -> Result<(), GpuError> {
        for i in 0..self.dims.len() {
            let (rows, cols) = self.dims[i];
            let w_off = self.w_offsets[i];
            let b_off = self.b_offsets[i];
            let p_off = cache.p_offsets[i];
            let a_off = cache.a_offsets[i + 1];

            let w_slice = self.weights.slice(w_off..w_off + rows * cols);
            let b_slice = self.biases.slice(b_off..b_off + rows);

            {
                let a_prev_view = if i == 0 {
                    input.slice(0..bs * cols)
                } else {
                    cache
                        .activations
                        .slice(cache.a_offsets[i]..cache.a_offsets[i] + bs * cols)
                };

                let mut z_slice = cache.pre_activations.slice_mut(p_off..p_off + bs * rows);

                blas.gemm_tb(
                    bs,
                    cols,
                    rows,
                    1.0,
                    &a_prev_view,
                    &w_slice,
                    0.0,
                    &mut z_slice,
                )?;

                launch_bias_add(&kernels.bias_add, &mut z_slice, &b_slice, bs, rows)?;
            }

            let mut a_slice = cache.activations.slice_mut(a_off..a_off + bs * rows);
            let mut z_slice = cache.pre_activations.slice_mut(p_off..p_off + bs * rows);

            if i == self.dims.len() - 1 {
                launch_softmax(&kernels.softmax, &mut z_slice, &mut a_slice, bs, rows)?;
            } else {
                launch_relu(&kernels.relu, &mut z_slice, &mut a_slice, bs * rows)?;
                if is_training {
                    let seed = fastrand::u32(..);
                    launch_dropout(
                        &kernels.dropout,
                        &mut a_slice,
                        bs * rows,
                        dropout_keep,
                        seed,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Processa o Backward Pass de um batch na GPU para computar os gradientes locais.
    /// Lança um kernel unificado para o erro cruzado da Softmax (`softmax_crossentropy_backward`), 
    /// extrai gradientes matriciais via cuBLAS (`gemm_ta`, `gemm`),
    /// e propaga pela ReLU reversa e acumulação vetorial de linhas de erro.
    pub fn backward_batch(
        &self,
        cache: &mut BatchCache,
        targets: &cudarc::driver::CudaView<i32>,
        acc_grads: &mut Gradients,
        bs: usize,
        kernels: &Kernels,
        blas: &BlasHandle,
        label_smoothing: f32,
    ) -> Result<(), GpuError> {
        let num_layers = self.dims.len();
        let inv_bs = 1.0 / bs as f32;

        let out_dim = self.dims[num_layers - 1].0;
        let a_last_off = cache.a_offsets[num_layers];
        let targets_view = targets.slice(0..bs);

        {
            let probs = cache
                .activations
                .slice(a_last_off..a_last_off + bs * out_dim);
            let mut delta_out = cache.deltas[num_layers - 1].slice_mut(0..bs * out_dim);
            launch_softmax_crossentropy_backward(
                &kernels.softmax_crossentropy_backward,
                &probs,
                &mut delta_out,
                &targets_view,
                bs,
                out_dim,
                label_smoothing,
            )?;
        }

        {
            let (rows, cols) = self.dims[num_layers - 1];
            let a_prev_off = cache.a_offsets[num_layers - 1];
            let a_prev = cache.activations.slice(a_prev_off..a_prev_off + bs * cols);
            let delta = cache.deltas[num_layers - 1].slice(0..bs * out_dim);
            let mut dw = acc_grads
                .dw
                .slice_mut(acc_grads.w_offsets[num_layers - 1]..);
            blas.gemm_ta(rows, bs, cols, inv_bs, &delta, &a_prev, 1.0, &mut dw)?;
        }
        {
            let mut delta = cache.deltas[num_layers - 1].slice_mut(0..bs * out_dim);
            let mut db = acc_grads
                .db
                .slice_mut(acc_grads.b_offsets[num_layers - 1]..);
            launch_sum_rows(&kernels.sum_rows, &mut delta, &mut db, bs, out_dim)?;
        }

        for l in (0..num_layers - 1).rev() {
            let (rows_next, _) = self.dims[l + 1];
            let next_dim = self.dims[l].0;
            let w_next_off = self.w_offsets[l + 1];
            let w_next = self
                .weights
                .slice(w_next_off..w_next_off + rows_next * next_dim);

            {
                let (left, right) = cache.deltas.split_at_mut(l + 1);
                let delta_next = right[0].slice(0..bs * rows_next);
                let mut delta_curr = left[l].slice_mut(0..bs * next_dim);
                blas.gemm(
                    bs,
                    rows_next,
                    next_dim,
                    1.0,
                    &delta_next,
                    &w_next,
                    0.0,
                    &mut delta_curr,
                )?;
            }

            {
                let z_prev_off = cache.p_offsets[l];
                let z_prev = cache
                    .pre_activations
                    .slice(z_prev_off..z_prev_off + bs * next_dim);
                let mut delta_curr = cache.deltas[l].slice_mut(0..bs * next_dim);
                launch_relu_backward(
                    &kernels.relu_backward,
                    &z_prev,
                    &mut delta_curr,
                    bs * next_dim,
                )?;
            }

            {
                let (r, c) = self.dims[l];
                let a_prev_off = cache.a_offsets[l];
                let a_prev = cache.activations.slice(a_prev_off..a_prev_off + bs * c);
                let delta_curr = cache.deltas[l].slice(0..bs * next_dim);
                let mut dw = acc_grads.dw.slice_mut(acc_grads.w_offsets[l]..);
                blas.gemm_ta(r, bs, c, inv_bs, &delta_curr, &a_prev, 1.0, &mut dw)?;
            }

            {
                let mut delta_curr = cache.deltas[l].slice_mut(0..bs * next_dim);
                let mut db = acc_grads.db.slice_mut(acc_grads.b_offsets[l]..);
                launch_sum_rows(&kernels.sum_rows, &mut delta_curr, &mut db, bs, next_dim)?;
            }
        }

        Ok(())
    }
}
