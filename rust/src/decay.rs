//! Decay: Ebbinghaus Forgetting Curve + Time-Decay Mathematics
//!
//! Implements the forgetting curve R = e^(-t/S) where:
//!   R = retention (0..1)
//!   t = time since last access
//!   S = stability (strength of memory)
//!
//! Memories that are accessed more frequently have higher stability (S),
//! meaning they decay slower. This mirrors synaptic consolidation — repeated
//! recall strengthens memory traces.

use pyo3::prelude::*;

/// Compute retention probability using Ebbinghaus forgetting curve.
///
/// Args:
///     elapsed_hours: Time since last access in hours
///     stability: Memory stability factor (higher = slower decay)
///     base_confidence: Initial confidence when last accessed (0..1)
///
/// Returns:
///     Current retention/confidence level (0..1)
#[pyfunction]
#[pyo3(signature = (elapsed_hours, stability=1.0, base_confidence=1.0))]
pub fn ebbinghaus_retention(
    elapsed_hours: f64,
    stability: f64,
    base_confidence: f64,
) -> PyResult<f64> {
    if stability <= 0.0 {
        return Ok(0.0);
    }
    if elapsed_hours <= 0.0 {
        return Ok(base_confidence);
    }

    let retention = base_confidence * (-elapsed_hours / stability).exp();
    Ok(retention.clamp(0.0, 1.0))
}

/// Compute decayed confidence for a memory given its access history.
///
/// Args:
///     base_confidence: Original confidence score (0..1)
///     elapsed_hours: Hours since last access
///     access_count: Total number of times this memory was accessed
///     importance: Importance/salience score (0..1), boosts stability
///     decay_rate: Global decay rate multiplier (higher = faster forgetting)
///
/// Returns:
///     Decayed confidence score (0..1)
#[pyfunction]
#[pyo3(signature = (base_confidence, elapsed_hours, access_count=1, importance=0.5, decay_rate=0.1))]
pub fn compute_confidence_decay(
    base_confidence: f64,
    elapsed_hours: f64,
    access_count: u32,
    importance: f64,
    decay_rate: f64,
) -> PyResult<f64> {
    // Stability increases with access count and importance
    // S = (1 + log(1 + access_count)) * (0.5 + importance) / decay_rate
    let access_boost = (1.0 + (1.0 + access_count as f64).ln()).max(1.0);
    let importance_factor = (0.5 + importance).clamp(0.5, 1.5);
    let stability = access_boost * importance_factor / decay_rate.max(0.01);

    ebbinghaus_retention(elapsed_hours, stability, base_confidence)
}

/// Compute a time-decay score for ranking memories in retrieval.
/// Combines relevance similarity with temporal decay.
///
/// Args:
///     similarity: Cosine similarity or other relevance score (0..1)
///     elapsed_hours: Hours since memory was created/accessed
///     recency_weight: How much to weight recency vs. relevance (0..1)
///     half_life_hours: Half-life for time decay (default: 168 = 1 week)
///
/// Returns:
///     Combined score (0..1)
#[pyfunction]
#[pyo3(signature = (similarity, elapsed_hours, recency_weight=0.3, half_life_hours=168.0))]
pub fn time_decay_score(
    similarity: f64,
    elapsed_hours: f64,
    recency_weight: f64,
    half_life_hours: f64,
) -> PyResult<f64> {
    // Exponential time decay with configurable half-life
    let decay = if half_life_hours > 0.0 {
        (-(0.693 * elapsed_hours) / half_life_hours).exp() // ln(2) ≈ 0.693
    } else {
        1.0
    };

    // Weighted combination of relevance and recency
    let w = recency_weight.clamp(0.0, 1.0);
    let score = (1.0 - w) * similarity + w * decay;

    Ok(score.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_decay_at_zero_time() {
        let r = ebbinghaus_retention(0.0, 1.0, 0.9).unwrap();
        assert!((r - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_decay_over_time() {
        let r1 = ebbinghaus_retention(1.0, 10.0, 1.0).unwrap();
        let r2 = ebbinghaus_retention(100.0, 10.0, 1.0).unwrap();
        assert!(r1 > r2); // more time = more decay
    }

    #[test]
    fn test_higher_stability_slower_decay() {
        let r_low = ebbinghaus_retention(24.0, 10.0, 1.0).unwrap();
        let r_high = ebbinghaus_retention(24.0, 100.0, 1.0).unwrap();
        assert!(r_high > r_low); // higher stability = slower decay
    }

    #[test]
    fn test_confidence_decay_with_access() {
        let r_few = compute_confidence_decay(1.0, 24.0, 1, 0.5, 0.1).unwrap();
        let r_many = compute_confidence_decay(1.0, 24.0, 100, 0.5, 0.1).unwrap();
        assert!(r_many > r_few); // more accesses = slower decay
    }

    #[test]
    fn test_time_decay_score_recent_high() {
        let s1 = time_decay_score(0.8, 1.0, 0.3, 168.0).unwrap();
        let s2 = time_decay_score(0.8, 1000.0, 0.3, 168.0).unwrap();
        assert!(s1 > s2); // recent memory scores higher
    }

    #[test]
    fn test_retention_clamped() {
        let r = ebbinghaus_retention(1.0, 10.0, 1.5).unwrap();
        assert!(r <= 1.0);
    }
}
