pub fn mat_vec_mul(matrix: &[Vec<f64>], in_vector: &[f64]) -> Vec<f64> {
    let mut out_vector = Vec::new();
    for i in 0..matrix.len() {
        let mut soma = 0.0;
        for j in 0..matrix[i].len() {
            soma += matrix[i][j] * in_vector[j];
        }
        out_vector.push(soma);
        soma = 0.0;
    }
    out_vector
}
