//! Dentate Gyrus: Pattern Separation
//!
//! The dentate gyrus orthogonalizes similar inputs into sparse, distinct codes.
//! This prevents catastrophic interference — even highly similar experiences
//! get unique fingerprints in memory.
//!
//! Implementation: Sparse random projection + winner-take-all (WTA) activation.
//! A dense embedding vector is projected through a fixed random matrix into a
//! high-dimensional sparse space. Only the top-k activations survive (WTA),
//! producing a binary sparse code.

use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// Generate a sparse binary code from a dense embedding via random projection + WTA.
///
/// Args:
///     embedding: Dense input vector (e.g., 768-dim from embedding model)
///     sparse_dim: Dimensionality of the sparse output space (default: 2048)
///     sparsity_k: Number of active (1) bits in the output (default: 64)
///     seed: Random seed for reproducible projections (default: 42)
///
/// Returns:
///     List of indices of active bits in the sparse code
#[pyfunction]
#[pyo3(signature = (embedding, sparse_dim=2048, sparsity_k=64, seed=42))]
pub fn pattern_separate(
    embedding: Vec<f64>,
    sparse_dim: usize,
    sparsity_k: usize,
    seed: u64,
) -> PyResult<Vec<usize>> {
    let input_dim = embedding.len();
    let mut rng = StdRng::seed_from_u64(seed);

    // Random projection: project input into sparse_dim space
    // Uses a fixed random matrix seeded deterministically
    let mut activations = vec![0.0f64; sparse_dim];
    for j in 0..sparse_dim {
        let mut sum = 0.0;
        for i in 0..input_dim {
            // Sparse random projection matrix (ternary: -1, 0, +1)
            let r: f64 = match rng.gen_range(0u8..6) {
                0 => -1.0,
                5 => 1.0,
                _ => 0.0, // 2/3 chance of zero — sparse projection
            };
            sum += embedding[i] * r;
        }
        activations[j] = sum;
    }

    // Winner-take-all: keep only the top-k activations
    let mut indexed: Vec<(usize, f64)> = activations.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let active_indices: Vec<usize> = indexed.into_iter().take(sparsity_k).map(|(i, _)| i).collect();

    Ok(active_indices)
}

/// Batch pattern separation for multiple embeddings.
#[pyfunction]
#[pyo3(signature = (embeddings, sparse_dim=2048, sparsity_k=64, seed=42))]
pub fn batch_pattern_separate(
    embeddings: Vec<Vec<f64>>,
    sparse_dim: usize,
    sparsity_k: usize,
    seed: u64,
) -> PyResult<Vec<Vec<usize>>> {
    embeddings
        .into_iter()
        .map(|emb| pattern_separate(emb, sparse_dim, sparsity_k, seed))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_separate_output_size() {
        let embedding = vec![0.1; 768];
        let result = pattern_separate(embedding, 2048, 64, 42).unwrap();
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_pattern_separate_deterministic() {
        let embedding = vec![0.5; 768];
        let r1 = pattern_separate(embedding.clone(), 2048, 64, 42).unwrap();
        let r2 = pattern_separate(embedding, 2048, 64, 42).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_different_inputs_different_codes() {
        let emb1 = vec![0.1; 768];
        let emb2 = vec![0.9; 768];
        let r1 = pattern_separate(emb1, 2048, 64, 42).unwrap();
        let r2 = pattern_separate(emb2, 2048, 64, 42).unwrap();
        // Different inputs should produce different sparse codes
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_indices_within_bounds() {
        let embedding = vec![0.3; 768];
        let result = pattern_separate(embedding, 2048, 64, 42).unwrap();
        for idx in &result {
            assert!(*idx < 2048);
        }
    }
}
