/// Aplica a função de ativação ReLU (Rectified Linear Unit) a um vetor de entrada.
/// Retorna um novo vetor onde todos os valores negativos são substituídos por 0.
pub fn relu_forward(z: &[f32]) -> Vec<f32> {
    z.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
}

/// Calcula o gradiente (backward pass) da função ReLU.
/// Recebe os valores da camada atual `z` e os gradientes da camada seguinte `delta`.
pub fn relu_backward(z: &[f32], delta: &[f32]) -> Vec<f32> {
    z.iter()
        .zip(delta.iter())
        .map(|(&z_val, &d_val)| if z_val > 0.0 { d_val } else { 0.0 })
        .collect()
}

/// Aplica a função Softmax a um vetor de valores (logits).
/// Converte as pontuações brutas em uma distribuição de probabilidade (valores entre 0 e 1 que somam 1).
pub fn softmax_forward(z: &[f32]) -> Vec<f32> {
    let max_z = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_z: Vec<f32> = z.iter().map(|&v| (v - max_z).exp()).collect();
    let sum_exp: f32 = exp_z.iter().sum();
    exp_z.iter().map(|&v| v / sum_exp).collect()
}

/// Calcula a função de perda entropia cruzada (Cross-Entropy Loss).
/// Recebe as `probs` (probabilidades preditas pelo Softmax) e o `target` (índice da classe real).
pub fn cross_entropy_loss(probs: &[f32], target: usize) -> f32 {
    -probs[target].max(1e-15).ln()
}

/// Calcula o gradiente combinado da função Softmax e da perda de Entropia Cruzada.
/// É uma operação simplificada e numericamente mais estável do que calculá-las separadamente.
pub fn softmax_crossentropy_backward(probs: &[f32], target: usize) -> Vec<f32> {
    let mut delta = probs.to_vec();
    delta[target] -= 1.0;
    delta
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_eq_vec(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y))
    }

    #[test]
    fn test_relu_forward_positive() {
        let z = vec![1.0, 2.0, 3.0, 4.0];
        let result = relu_forward(&z);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_relu_forward_negative() {
        let z = vec![-1.0, -2.0, -3.0, -4.0];
        let result = relu_forward(&z);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_forward_mixed() {
        let z = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = relu_forward(&z);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_forward_zero() {
        let z = vec![0.0];
        let result = relu_forward(&z);
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn test_relu_backward_positive() {
        let z = vec![1.0, 2.0, 3.0];
        let delta = vec![0.5, 1.0, 1.5];
        let result = relu_backward(&z, &delta);
        assert_eq!(result, vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_relu_backward_negative() {
        let z = vec![-1.0, -2.0, -3.0];
        let delta = vec![0.5, 1.0, 1.5];
        let result = relu_backward(&z, &delta);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_backward_mixed() {
        let z = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let delta = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let result = relu_backward(&z, &delta);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_softmax_forward_basic() {
        let z = vec![1.0, 2.0, 3.0];
        let result = softmax_forward(&z);

        let sum: f32 = result.iter().sum();
        assert!(approx_eq(sum, 1.0));

        assert!(result.iter().all(|&v| v >= 0.0 && v <= 1.0));

        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_softmax_forward_uniform() {
        let z = vec![1.0, 1.0, 1.0];
        let result = softmax_forward(&z);

        let expected = 1.0 / 3.0;
        assert!(result.iter().all(|&v| approx_eq(v, expected)));
    }

    #[test]
    fn test_softmax_forward_large_values() {
        let z = vec![1000.0, 1001.0, 1002.0];
        let result = softmax_forward(&z);

        let sum: f32 = result.iter().sum();
        assert!(approx_eq(sum, 1.0));

        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_softmax_forward_negative_values() {
        let z = vec![-10.0, -5.0, 0.0];
        let result = softmax_forward(&z);

        let sum: f32 = result.iter().sum();
        assert!(approx_eq(sum, 1.0));

        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_cross_entropy_loss_perfect() {
        let probs = vec![1.0, 0.0, 0.0];
        let loss = cross_entropy_loss(&probs, 0);
        assert!(approx_eq(loss, 0.0));
    }

    #[test]
    fn test_cross_entropy_loss_uniform() {
        let probs = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let loss = cross_entropy_loss(&probs, 0);
        let expected = -(1.0 / 3.0_f32).ln();
        assert!(approx_eq(loss, expected));
    }

    #[test]
    fn test_cross_entropy_loss_different_targets() {
        let probs = vec![0.1, 0.3, 0.6];

        let loss0 = cross_entropy_loss(&probs, 0);
        let loss1 = cross_entropy_loss(&probs, 1);
        let loss2 = cross_entropy_loss(&probs, 2);

        assert!(loss0 > loss1);
        assert!(loss1 > loss2);
    }

    #[test]
    fn test_softmax_crossentropy_backward_basic() {
        let probs = vec![0.2, 0.3, 0.5];
        let delta = softmax_crossentropy_backward(&probs, 1);

        assert!(approx_eq(delta[0], 0.2));
        assert!(approx_eq(delta[1], 0.3 - 1.0));
        assert!(approx_eq(delta[2], 0.5));
    }

    #[test]
    fn test_softmax_crossentropy_backward_sum_zero() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let delta = softmax_crossentropy_backward(&probs, 2);

        let sum: f32 = delta.iter().sum();
        assert!(approx_eq(sum, 0.0));
    }

    #[test]
    fn test_softmax_crossentropy_backward_all_targets() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];

        for target in 0..4 {
            let delta = softmax_crossentropy_backward(&probs, target);

            let sum: f32 = delta.iter().sum();
            assert!(approx_eq(sum, 0.0));

            assert!(delta[target] < 0.0);

            for (i, &d) in delta.iter().enumerate() {
                if i != target {
                    assert!(d > 0.0);
                }
            }
        }
    }

    #[test]
    fn test_relu_gradient_check() {
        let z = vec![1.0, -1.0, 0.5, -0.5];
        let delta = vec![1.0, 1.0, 1.0, 1.0];
        let analytical_grad = relu_backward(&z, &delta);

        let epsilon = 1e-4;
        let tolerance = 1e-3;

        for (i, &z_val) in z.iter().enumerate() {
            let z_plus = {
                let mut z_p = z.clone();
                z_p[i] += epsilon;
                relu_forward(&z_p)
            };
            let z_minus = {
                let mut z_m = z.clone();
                z_m[i] -= epsilon;
                relu_forward(&z_m)
            };

            let numerical_grad = (z_plus[i] - z_minus[i]) / (2.0 * epsilon);

            if z_val.abs() > epsilon {
                assert!(
                    (analytical_grad[i] - numerical_grad).abs() < tolerance,
                    "Gradiente ReLU mismatch em {}: analítico={}, numérico={}",
                    i,
                    analytical_grad[i],
                    numerical_grad
                );
            }
        }
    }

    #[test]
    fn test_softmax_gradient_check() {
        let z = vec![1.0, 2.0, 3.0];
        let target = 1;

        let probs = softmax_forward(&z);
        let analytical_grad = softmax_crossentropy_backward(&probs, target);

        let epsilon = 1e-4;
        let tolerance = 1e-3;

        for i in 0..z.len() {
            let z_plus = {
                let mut z_p = z.clone();
                z_p[i] += epsilon;
                let p = softmax_forward(&z_p);
                cross_entropy_loss(&p, target)
            };
            let z_minus = {
                let mut z_m = z.clone();
                z_m[i] -= epsilon;
                let p = softmax_forward(&z_m);
                cross_entropy_loss(&p, target)
            };

            let numerical_grad = (z_plus - z_minus) / (2.0 * epsilon);

            assert!(
                (analytical_grad[i] - numerical_grad).abs() < tolerance,
                "Gradiente mismatch em {}: analítico={}, numérico={}",
                i,
                analytical_grad[i],
                numerical_grad
            );
        }
    }

    #[test]
    fn test_relu_empty() {
        let z: Vec<f32> = vec![];
        let result = relu_forward(&z);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_single_element() {
        let z = vec![5.0];
        let result = softmax_forward(&z);
        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0], 1.0));
    }

    #[test]
    fn test_softmax_two_elements() {
        let z = vec![0.0, 0.0];
        let result = softmax_forward(&z);
        assert!(approx_eq(result[0], 0.5));
        assert!(approx_eq(result[1], 0.5));
    }
}
