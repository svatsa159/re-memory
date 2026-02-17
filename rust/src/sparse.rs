//! Sparse Vector Operations
//!
//! Operations on sparse binary codes produced by the Dentate Gyrus.
//! Sparse codes are represented as sorted lists of active indices.

use pyo3::prelude::*;

/// Compute Hamming distance between two sparse binary codes.
/// (Number of positions where the codes differ)
///
/// Args:
///     code_a: Sorted list of active indices
///     code_b: Sorted list of active indices
///     total_dim: Total dimensionality of the sparse space
///
/// Returns:
///     Hamming distance (number of differing bits)
#[pyfunction]
#[pyo3(signature = (code_a, code_b, total_dim=2048))]
pub fn hamming_distance(code_a: Vec<usize>, code_b: Vec<usize>, total_dim: usize) -> PyResult<u32> {
    let _ = total_dim; // used conceptually; active bits define the codes
    let set_a: std::collections::HashSet<usize> = code_a.into_iter().collect();
    let set_b: std::collections::HashSet<usize> = code_b.into_iter().collect();

    let union_size = set_a.union(&set_b).count();
    let intersection_size = set_a.intersection(&set_b).count();

    // Hamming distance = bits in A not in B + bits in B not in A
    Ok((union_size - intersection_size) as u32)
}

/// Compute Jaccard similarity between two sparse binary codes.
/// J(A, B) = |A ∩ B| / |A ∪ B|
///
/// Args:
///     code_a: Sorted list of active indices
///     code_b: Sorted list of active indices
///
/// Returns:
///     Jaccard similarity (0..1)
#[pyfunction]
pub fn jaccard_similarity(code_a: Vec<usize>, code_b: Vec<usize>) -> PyResult<f64> {
    if code_a.is_empty() && code_b.is_empty() {
        return Ok(1.0);
    }

    let set_a: std::collections::HashSet<usize> = code_a.into_iter().collect();
    let set_b: std::collections::HashSet<usize> = code_b.into_iter().collect();

    let intersection = set_a.intersection(&set_b).count() as f64;
    let union = set_a.union(&set_b).count() as f64;

    if union == 0.0 {
        Ok(1.0)
    } else {
        Ok(intersection / union)
    }
}

/// Compute the overlap coefficient between two sparse codes.
/// O(A, B) = |A ∩ B| / min(|A|, |B|)
///
/// More lenient than Jaccard — useful for partial cue matching.
///
/// Args:
///     code_a: Sorted list of active indices
///     code_b: Sorted list of active indices
///
/// Returns:
///     Overlap coefficient (0..1)
#[pyfunction]
pub fn sparse_overlap(code_a: Vec<usize>, code_b: Vec<usize>) -> PyResult<f64> {
    if code_a.is_empty() || code_b.is_empty() {
        return Ok(0.0);
    }

    let set_a: std::collections::HashSet<usize> = code_a.into_iter().collect();
    let set_b: std::collections::HashSet<usize> = code_b.into_iter().collect();

    let intersection = set_a.intersection(&set_b).count() as f64;
    let min_size = (set_a.len().min(set_b.len())) as f64;

    if min_size == 0.0 {
        Ok(0.0)
    } else {
        Ok(intersection / min_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_codes_zero_hamming() {
        let code = vec![1, 5, 10, 20];
        let dist = hamming_distance(code.clone(), code, 2048).unwrap();
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_disjoint_codes_max_hamming() {
        let a = vec![0, 1, 2];
        let b = vec![3, 4, 5];
        let dist = hamming_distance(a, b, 2048).unwrap();
        assert_eq!(dist, 6); // all 6 bits differ
    }

    #[test]
    fn test_jaccard_identical() {
        let code = vec![1, 2, 3];
        let j = jaccard_similarity(code.clone(), code).unwrap();
        assert!((j - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = vec![0, 1, 2];
        let b = vec![3, 4, 5];
        let j = jaccard_similarity(a, b).unwrap();
        assert!((j - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_overlap_partial() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![3, 4, 5];
        let o = sparse_overlap(a, b).unwrap();
        assert!((o - 1.0).abs() < 1e-6); // b is subset of a
    }
}
