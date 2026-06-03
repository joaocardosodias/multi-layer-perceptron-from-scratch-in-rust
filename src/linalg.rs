pub fn mul_matrix_vec(matrix: &[Vec<f64>], in_vector: &[f64]) -> Vec<f64> {
    let mut out_vector = Vec::new();
    for i in 0..matrix.len() {
        let mut sum = 0.0;
        for j in 0..matrix[i].len() {
            sum += matrix[i][j] * in_vector[j];
        }
        out_vector.push(sum);
        sum = 0.0;
    }
    out_vector
}
pub fn add_vec_vec(vec1: &[f64], vec2: &[f64]) -> Vec<f64> {
    let mut out_vector = Vec::new();
    for i in 0..vec1.len() {
        out_vector.push(vec1[i] + vec2[i]);
    }
    out_vector
}
pub fn mul_vec_vec(vec1: &[f64], vec2: &[f64]) -> Vec<f64> {
    let mut out_vector = Vec::new();
    for i in 0..vec1.len() {
        out_vector.push(vec1[i] * vec2[i]);
    }
    out_vector
}
pub fn mul_matrix_matrix(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = matrix1.len();
    let n = matrix2[0].len();
    let k = matrix2.len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i][j] +=matrix1[i][p] * matrix2[p][j];
            }
        }
    }
    c
}