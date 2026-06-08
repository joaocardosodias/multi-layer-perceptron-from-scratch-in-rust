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
    pub gather_and_augment: CudaFunction,
    pub count_correct: CudaFunction,
    pub compute_loss: CudaFunction,
    pub gather_labels: CudaFunction,
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
        let gather_and_augment            = compile_kernel!(dev, "gather_and_augment",          GATHER_AND_AUGMENT);
        let count_correct                 = compile_kernel!(dev, "count_correct",               COUNT_CORRECT);
        let compute_loss                  = compile_kernel!(dev, "compute_loss",                COMPUTE_LOSS);
        let gather_labels                 = compile_kernel!(dev, "gather_labels",               GATHER_LABELS);

        Ok(Kernels {
            bias_add, relu, relu_backward, dropout, softmax,
            softmax_crossentropy_backward, adam_update, sum_rows,
            gather_and_augment, count_correct, compute_loss,
            gather_labels,
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
    label_smoothing: f32,
) -> Result<(), GpuError> {
    let cfg = launch_cfg(bs * out_dim);
    unsafe { f.clone().launch(cfg, (probs, delta, targets, bs as i32, out_dim as i32, label_smoothing))? };
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

pub fn launch_gather_and_augment(
    f: &CudaFunction,
    all_images: &CudaView<f32>,
    indices: &CudaView<i32>,
    batch: &mut CudaViewMut<f32>,
    batch_size: usize,
    seed: u32,
    p_keep: f32,
) -> Result<(), GpuError> {
    let cfg = launch_cfg(batch_size * 784);
    unsafe { f.clone().launch(cfg, (all_images, indices, batch, batch_size as i32, seed, p_keep))? };
    Ok(())
}

pub fn launch_gather_labels(
    f: &CudaFunction,
    all_labels: &CudaView<i32>,
    indices: &CudaView<i32>,
    batch: &mut CudaViewMut<i32>,
    batch_size: usize,
) -> Result<(), GpuError> {
    let cfg = launch_cfg(batch_size);
    unsafe { f.clone().launch(cfg, (all_labels, indices, batch, batch_size as i32))? };
    Ok(())
}

pub fn launch_count_correct(
    f: &CudaFunction,
    probs: &CudaView<f32>,
    labels: &CudaView<i32>,
    correct: &mut CudaViewMut<i32>,
    batch_size: usize,
) -> Result<(), GpuError> {
    let cfg = LaunchConfig {
        block_dim: (1, 1, 1),
        grid_dim:  (batch_size as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { f.clone().launch(cfg, (probs, labels, correct, batch_size as i32))? };
    Ok(())
}

pub fn launch_compute_loss(
    f: &CudaFunction,
    probs: &CudaView<f32>,
    labels: &CudaView<i32>,
    loss: &mut CudaViewMut<f32>,
    batch_size: usize, out_dim: usize,
) -> Result<(), GpuError> {
    let cfg = launch_cfg(batch_size);
    unsafe { f.clone().launch(cfg, (probs, labels, loss, batch_size as i32, out_dim as i32))? };
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
        const float* probs, float* delta, const int* targets, int bs, int out_dim, float label_smoothing) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bs * out_dim) {
        int s = idx / out_dim;
        int r = idx % out_dim;
        float target_val = (r == __ldg(&targets[s])) ? (1.0f - label_smoothing) : (label_smoothing / (out_dim - 1));
        delta[idx] = __ldg(&probs[idx]) - target_val;
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

const GATHER_AND_AUGMENT: &str = r#"
extern "C" __global__ void gather_and_augment(const float* all_images, const int* indices, float* batch, int batch_size, unsigned int seed, float p_keep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * 784;
    if (idx >= total) return;

    int sample = idx / 784;
    int pixel  = idx % 784;
    int x = pixel % 28;
    int y = pixel / 28;

    int img_idx = __ldg(&indices[sample]);
    float orig = __ldg(&all_images[img_idx * 784 + pixel]);

    // RNG para decidir se aplica augmentation
    unsigned int s1 = seed ^ (sample * 2654435761u);
    s1 ^= s1 << 13; s1 ^= s1 >> 17; s1 ^= s1 << 5;
    float r_keep = (s1 & 0x7FFFFFFFu) * (1.0f / 2147483647.0f);

    if (r_keep > p_keep) {
        batch[idx] = orig;
        return;
    }

    // ==== AFFINE (igual a CPU) ====
    unsigned int s2 = s1 ^ 0x9e3779b9u;
    s2 ^= s2 << 13; s2 ^= s2 >> 17; s2 ^= s2 << 5;
    float angle_deg = (s2 & 0x7FFFFFFFu) * (20.0f / 2147483647.0f) - 10.0f;

    unsigned int s3 = s2 ^ 0x85ebca6bu;
    s3 ^= s3 << 13; s3 ^= s3 >> 17; s3 ^= s3 << 5;
    float tx = (s3 & 0x7FFFFFFFu) * (3.0f / 2147483647.0f) - 1.5f;

    unsigned int s4 = s3 ^ 0xc2b2ae35u;
    s4 ^= s4 << 13; s4 ^= s4 >> 17; s4 ^= s4 << 5;
    float ty = (s4 & 0x7FFFFFFFu) * (3.0f / 2147483647.0f) - 1.5f;

    float angle_rad = angle_deg * 3.14159265f / 180.0f;
    float cos_a = cosf(angle_rad);
    float sin_a = sinf(angle_rad);
    float cx = 13.5f, cy = 13.5f;

    float dx = x - cx;
    float dy = y - cy;
    float src_x = dx * cos_a + dy * sin_a - tx + cx;
    float src_y = -dx * sin_a + dy * cos_a - ty + cy;

    // ==== ELASTIC DISTORTION (aproximado via smooth noise) ====
    // Gerar offset aleatório correlacionado espacialmente
    unsigned int sex = s1 ^ (x * 1664525u);
    sex ^= sex << 13; sex ^= sex >> 17; sex ^= sex << 5;
    float off_x = (sex & 0x7FFFFFFFu) * (2.0f / 2147483647.0f) - 1.0f;

    unsigned int sey = s1 ^ (y * 1103515245u);
    sey ^= sey << 13; sey ^= sey >> 17; sey ^= sey << 5;
    float off_y = (sey & 0x7FFFFFFFu) * (2.0f / 2147483647.0f) - 1.0f;

    // Aplicar offsets (alpha=36, sigma=5 aproximado)
    src_x += off_x * 1.8f;
    src_y += off_y * 1.8f;

    if (src_x >= 0.0f && src_x < 27.0f && src_y >= 0.0f && src_y < 27.0f) {
        int x0 = (int)floorf(src_x);
        int y0 = (int)floorf(src_y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float wx = src_x - x0;
        float wy = src_y - y0;

        float v00 = __ldg(&all_images[img_idx * 784 + y0 * 28 + x0]);
        float v10 = __ldg(&all_images[img_idx * 784 + y0 * 28 + x1]);
        float v01 = __ldg(&all_images[img_idx * 784 + y1 * 28 + x0]);
        float v11 = __ldg(&all_images[img_idx * 784 + y1 * 28 + x1]);

        batch[idx] = (1.0f - wx) * (1.0f - wy) * v00
                    + wx * (1.0f - wy) * v10
                    + (1.0f - wx) * wy * v01
                    + wx * wy * v11;
    } else {
        batch[idx] = 0.0f;
    }
}
"#;

const GATHER_LABELS: &str = r#"
extern "C" __global__ void gather_labels(const int* all_labels, const int* indices, int* batch, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        int idx = __ldg(&indices[i]);
        batch[i] = __ldg(&all_labels[idx]);
    }
}
"#;

const COUNT_CORRECT: &str = r#"
extern "C" __global__ void count_correct(const float* probs, const int* labels, int* correct, int batch_size) {
    int s = blockIdx.x;
    if (s < batch_size) {
        int best = 0;
        float best_val = probs[s * 10];
        for (int i = 1; i < 10; i++) {
            float v = probs[s * 10 + i];
            if (v > best_val) { best_val = v; best = i; }
        }
        if (best == labels[s]) atomicAdd(correct, 1);
    }
}
"#;

const COMPUTE_LOSS: &str = r#"
extern "C" __global__ void compute_loss(const float* probs, const int* labels, float* loss, int batch_size, int out_dim) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < batch_size) {
        int target = labels[s];
        float p = probs[s * out_dim + target];
        p = fmaxf(p, 1e-10f);
        atomicAdd(loss, -logf(p));
    }
}
"#;
