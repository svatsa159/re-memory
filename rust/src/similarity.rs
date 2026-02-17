//! Fast Similarity Computations
//!
//! High-performance cosine similarity for dense vectors.

use pyo3::prelude::*;

/// Compute cosine similarity between two dense vectors.
///
/// Args:
///     vec_a: First vector
///     vec_b: Second vector
///
/// Returns:
///     Cosine similarity (-1..1)
#[pyfunction]
pub fn cosine_similarity(vec_a: Vec<f64>, vec_b: Vec<f64>) -> PyResult<f64> {
    if vec_a.len() != vec_b.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Vectors must have equal length",
        ));
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (a, b) in vec_a.iter().zip(vec_b.iter()) {
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        return Ok(0.0);
    }

    Ok(dot / denom)
}

/// Compute cosine similarity between a query vector and multiple target vectors.
///
/// Args:
///     query: The query vector
///     targets: List of target vectors
///
/// Returns:
///     List of (index, similarity) tuples sorted by similarity descending
#[pyfunction]
pub fn batch_cosine_similarity(
    query: Vec<f64>,
    targets: Vec<Vec<f64>>,
) -> PyResult<Vec<(usize, f64)>> {
    let query_norm: f64 = query.iter().map(|x| x * x).sum::<f64>().sqrt();
    if query_norm < 1e-10 {
        return Ok(vec![]);
    }

    let mut results: Vec<(usize, f64)> = targets
        .iter()
        .enumerate()
        .map(|(i, target)| {
            if target.len() != query.len() {
                return (i, 0.0);
            }
            let dot: f64 = query.iter().zip(target.iter()).map(|(a, b)| a * b).sum();
            let target_norm: f64 = target.iter().map(|x| x * x).sum::<f64>().sqrt();
            let denom = query_norm * target_norm;
            if denom < 1e-10 {
                (i, 0.0)
            } else {
                (i, dot / denom)
            }
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(v.clone(), v).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(a, b).unwrap();
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(a, b).unwrap();
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_batch_sorted_descending() {
        let query = vec![1.0, 0.0, 0.0];
        let targets = vec![
            vec![0.0, 1.0, 0.0], // orthogonal
            vec![1.0, 0.0, 0.0], // identical
            vec![0.5, 0.5, 0.0], // partial
        ];
        let results = batch_cosine_similarity(query, targets).unwrap();
        assert_eq!(results[0].0, 1); // identical is first
        assert!(results[0].1 > results[1].1); // sorted descending
    }

    #[test]
    fn test_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(cosine_similarity(a, b).is_err());
    }
}
