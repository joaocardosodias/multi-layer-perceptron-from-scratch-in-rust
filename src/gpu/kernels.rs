use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig, CudaView, CudaViewMut, LaunchAsync};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use crate::error::GpuError;

macro_rules! compile_kernel {
    ($dev:expr, $name:expr, $code:expr) => {{
        let ptx = compile_ptx($code)?;
        $dev.load_ptx(ptx, $name, &[$name])?;
        $dev.get_func($name, $name).ok_or_else(|| GpuError::Other(format!("Kernel {} not found", $name)))?
    }};
}

pub struct Kernels {
    pub bias_add: CudaFunction,
    pub relu: CudaFunction,
    pub relu_backward: CudaFunction,
    pub dropout: CudaFunction,
    pub softmax: CudaFunction,
    pub softmax_crossentropy_backward: CudaFunction,
    pub adam_update: CudaFunction,
    pub sum_rows: CudaFunction,
}

impl Kernels {
    pub fn new(dev: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        let bias_add                      = compile_kernel!(dev, "bias_add",                    BIAS_ADD);
        let relu                          = compile_kernel!(dev, "relu",                        RELU);
        let relu_backward                 = compile_kernel!(dev, "relu_backward",               RELU_BACKWARD);
        let dropout                       = compile_kernel!(dev, "dropout",                     DROPOUT);
        let softmax                       = compile_kernel!(dev, "softmax",                     SOFTMAX);
        let softmax_crossentropy_backward = compile_kernel!(dev, "softmax_crossentropy_backward", SOFTMAX_CROSSENTROPY_BACKWARD);
        let adam_update                   = compile_kernel!(dev, "adam_update",                 ADAM_UPDATE);
        let sum_rows                      = compile_kernel!(dev, "sum_rows",                    SUM_ROWS);

        Ok(Kernels {
            bias_add, relu, relu_backward, dropout, softmax,
            softmax_crossentropy_backward, adam_update, sum_rows,
        })
    }
}

fn launch_cfg(n: usize) -> LaunchConfig {
    let block_size: u32 = 256;
    let grid_size = ((n as u32) + block_size - 1) / block_size;
    LaunchConfig { block_dim: (block_size, 1, 1), grid_dim: (grid_size, 1, 1), shared_mem_bytes: 0 }
}

pub fn launch_bias_add(
    f: &CudaFunction,
    out: &mut CudaViewMut<f32>,
    b: &CudaView<f32>,
    rows: usize, cols: usize,
) -> Result<(), GpuError> {
    let cfg = launch_cfg(rows * cols);
    unsafe { f.clone().launch(cfg, (out, b, rows as i32, cols as i32))? };
    Ok(())
}

pub fn launch_relu(
    f: &CudaFunction,
    z: &mut CudaViewMut<f32>,
    a: &mut CudaViewMut<f32>,
    n: usize,
) -> Result<(), GpuError> {
    let cfg = launch_cfg(n);
    unsafe { f.clone().launch(cfg, (z, a, n as i32))? };
    Ok(())
}

pub fn launch_relu_backward(
    f: &CudaFunction,
    z: &CudaView<f32>,
    delta: &mut CudaViewMut<f32>,
    n: usize,
) -> Result<(), GpuError> {
    let cfg = launch_cfg(n);
    unsafe { f.clone().launch(cfg, (z, delta, n as i32))? };
    Ok(())
}

pub fn launch_dropout(
    f: &CudaFunction,
    a: &mut CudaViewMut<f32>,
    n: usize,
    p_keep: f32, seed: u32,
) -> Result<(), GpuError> {
    let cfg = launch_cfg(n);
    unsafe { f.clone().launch(cfg, (a, n as i32, p_keep, seed))? };
    Ok(())
}

pub fn launch_softmax(
    f: &CudaFunction,
    z: &mut CudaViewMut<f32>,
    a: &mut CudaViewMut<f32>,
    rows: usize, cols: usize,
) -> Result<(), GpuError> {
    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim:  (rows as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { f.clone().launch(cfg, (z, a, rows as i32, cols as i32))? };
    Ok(())
}

pub fn launch_softmax_crossentropy_backward(
    f: &CudaFunction,
    probs: &CudaView<f32>,
    delta: &mut CudaViewMut<f32>,
    targets: &CudaView<i32>,
    bs: usize, out_dim: usize,
) -> Result<(), GpuError> {
    let cfg = launch_cfg(bs * out_dim);
    unsafe { f.clone().launch(cfg, (probs, delta, targets, bs as i32, out_dim as i32))? };
    Ok(())
}

pub fn launch_adam_update(
    f: &CudaFunction,
    w: &mut cudarc::driver::CudaSlice<f32>,
    m: &mut cudarc::driver::CudaSlice<f32>,
    v: &mut cudarc::driver::CudaSlice<f32>,
    g: &cudarc::driver::CudaSlice<f32>,
    n: usize,
    lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, t: i32,
) -> Result<(), GpuError> {
    let cfg = launch_cfg(n);
    unsafe { f.clone().launch(cfg, (w, m, v, g, n as i32, lr, beta1, beta2, eps, weight_decay, t))? };
    Ok(())
}

pub fn launch_sum_rows(
    f: &CudaFunction,
    delta: &mut CudaViewMut<f32>,
    db: &mut CudaViewMut<f32>,
    rows: usize, cols: usize,
) -> Result<(), GpuError> {
    let block: u32 = 256;
    let shared     = block * 4;
    let cfg = LaunchConfig {
        block_dim: (block, 1, 1),
        grid_dim:  (cols as u32, 1, 1),
        shared_mem_bytes: shared,
    };
    unsafe { f.clone().launch(cfg, (delta, db, rows as i32, cols as i32))? };
    Ok(())
}

const BIAS_ADD: &str = r#"
extern "C" __global__ void bias_add(float* out, const float* b, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        out[idx] += __ldg(&b[col]);
    }
}
"#;

const RELU: &str = r#"
extern "C" __global__ void relu(const float* z, float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = __ldg(&z[i]);
        a[i] = v > 0.0f ? v : 0.0f;
    }
}
"#;

const RELU_BACKWARD: &str = r#"
extern "C" __global__ void relu_backward(const float* z, float* delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (__ldg(&z[i]) <= 0.0f) delta[i] = 0.0f;
    }
}
"#;

const DROPOUT: &str = r#"
extern "C" __global__ void dropout(float* a, int n, float p_keep, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int s = seed ^ (unsigned int)(i * 2654435761u);
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        float r     = (s & 0x7FFFFFFFu) * (1.0f / 2147483647.0f);
        float scale = 1.0f / p_keep;
        a[i] = r < p_keep ? a[i] * scale : 0.0f;
    }
}
"#;

const SOFTMAX: &str = r#"
extern "C" __global__ void softmax(const float* z, float* a, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        const float* zrow = z + row * cols;
        float*       arow = a + row * cols;
        float max_val = -1e20f;
        for (int c = 0; c < cols; c++) max_val = fmaxf(max_val, __ldg(&zrow[c]));
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            float e = expf(__ldg(&zrow[c]) - max_val);
            arow[c] = e; sum += e;
        }
        float inv = 1.0f / sum;
        for (int c = 0; c < cols; c++) arow[c] *= inv;
    }
}
"#;

const SOFTMAX_CROSSENTROPY_BACKWARD: &str = r#"
extern "C" __global__ void softmax_crossentropy_backward(
        const float* probs, float* delta, const int* targets, int bs, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bs * out_dim) {
        int s = idx / out_dim;
        int r = idx % out_dim;
        delta[idx] = __ldg(&probs[idx]) - (r == __ldg(&targets[s]) ? 1.0f : 0.0f);
    }
}
"#;

const ADAM_UPDATE: &str = r#"
extern "C" __global__ void adam_update(
        float* w, float* m, float* v, const float* g,
        int n, float lr, float beta1, float beta2, float eps,
        float weight_decay, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float grad = __ldg(&g[i]);
        float mi   = beta1 * m[i] + (1.0f - beta1) * grad;
        float vi   = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        m[i] = mi; v[i] = vi;
        float bc1  = 1.0f - powf(beta1, (float)t);
        float bc2  = 1.0f - powf(beta2, (float)t);
        float step = lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
        w[i] = w[i] * (1.0f - lr * weight_decay) - step;
    }
}
"#;

const SUM_ROWS: &str = r#"
extern "C" __global__ void sum_rows(const float* delta, float* db, int rows, int cols) {
    extern __shared__ float sdata[];
    int col  = blockIdx.x;
    int tid  = threadIdx.x;
    int bdim = blockDim.x;
    if (col >= cols) return;
    float acc = 0.0f;
    for (int r = tid; r < rows; r += bdim) acc += __ldg(&delta[r * cols + col]);
    sdata[tid] = acc;
    __syncthreads();
    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) db[col] += sdata[0] / (float)rows;
}
"#;
