//! CA3: Pattern Completion
//!
//! CA3 implements attractor dynamics for pattern completion — given a partial
//! or noisy input cue, it converges to the closest stored memory pattern.
//!
//! Implementation: Modern Hopfield Network (exponential energy function).
//! Unlike classical Hopfield nets (quadratic energy, low capacity), modern
//! Hopfield nets use exponential interaction functions, achieving exponential
//! storage capacity and one-step retrieval.
//!
//! Reference: Ramsauer et al., "Hopfield Networks is All You Need" (2020)

use pyo3::prelude::*;

/// Modern Hopfield Network retrieval: given a query pattern, find the best
/// matching stored pattern using softmax attention over stored memories.
///
/// Args:
///     query: The query/cue vector (dense embedding)
///     stored_patterns: Matrix of stored memory patterns (list of dense vectors)
///     beta: Inverse temperature parameter (higher = sharper retrieval, default: 8.0)
///     top_k: Number of top matches to return (default: 5)
///
/// Returns:
///     List of (index, similarity_score) tuples for top-k matches
#[pyfunction]
#[pyo3(signature = (query, stored_patterns, beta=8.0, top_k=5))]
pub fn pattern_complete(
    query: Vec<f64>,
    stored_patterns: Vec<Vec<f64>>,
    beta: f64,
    top_k: usize,
) -> PyResult<Vec<(usize, f64)>> {
    if stored_patterns.is_empty() {
        return Ok(vec![]);
    }

    let query_norm = l2_norm(&query);
    if query_norm < 1e-10 {
        return Ok(vec![]);
    }

    // Compute scaled dot-product similarities (energy)
    let energies: Vec<f64> = stored_patterns
        .iter()
        .map(|pattern| {
            let pattern_norm = l2_norm(pattern);
            if pattern_norm < 1e-10 {
                return f64::NEG_INFINITY;
            }
            let dot: f64 = query.iter().zip(pattern.iter()).map(|(a, b)| a * b).sum();
            beta * dot / (query_norm * pattern_norm)
        })
        .collect();

    // Softmax normalization for retrieval probabilities
    let max_energy = energies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let exp_energies: Vec<f64> = energies.iter().map(|e| (e - max_energy).exp()).collect();
    let sum_exp: f64 = exp_energies.iter().sum();

    let probabilities: Vec<f64> = if sum_exp > 0.0 {
        exp_energies.iter().map(|e| e / sum_exp).collect()
    } else {
        vec![0.0; energies.len()]
    };

    // Sort by probability and return top-k
    let mut indexed: Vec<(usize, f64)> = probabilities.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    Ok(indexed)
}

/// Batch pattern completion for multiple queries.
#[pyfunction]
#[pyo3(signature = (queries, stored_patterns, beta=8.0, top_k=5))]
pub fn batch_pattern_complete(
    queries: Vec<Vec<f64>>,
    stored_patterns: Vec<Vec<f64>>,
    beta: f64,
    top_k: usize,
) -> PyResult<Vec<Vec<(usize, f64)>>> {
    queries
        .into_iter()
        .map(|q| pattern_complete(q, stored_patterns.clone(), beta, top_k))
        .collect()
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_patterns() {
        let query = vec![1.0, 0.0, 0.0];
        let result = pattern_complete(query, vec![], 8.0, 5).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_exact_match_highest() {
        let query = vec![1.0, 0.0, 0.0];
        let patterns = vec![
            vec![1.0, 0.0, 0.0], // exact match
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = pattern_complete(query, patterns, 8.0, 3).unwrap();
        assert_eq!(result[0].0, 0); // index 0 should be best match
        assert!(result[0].1 > result[1].1); // with highest score
    }

    #[test]
    fn test_top_k_limit() {
        let query = vec![1.0, 0.5];
        let patterns = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let result = pattern_complete(query, patterns, 8.0, 2).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let query = vec![0.5, 0.3, 0.7];
        let patterns = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];
        let result = pattern_complete(query, patterns, 8.0, 3).unwrap();
        let total: f64 = result.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
