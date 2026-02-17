use pyo3::prelude::*;

mod ca3;
mod decay;
mod dentate_gyrus;
mod similarity;
mod sparse;

/// re-memory Rust core: performance-critical brain components
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dentate_gyrus::pattern_separate, m)?)?;
    m.add_function(wrap_pyfunction!(dentate_gyrus::batch_pattern_separate, m)?)?;
    m.add_function(wrap_pyfunction!(ca3::pattern_complete, m)?)?;
    m.add_function(wrap_pyfunction!(ca3::batch_pattern_complete, m)?)?;
    m.add_function(wrap_pyfunction!(decay::ebbinghaus_retention, m)?)?;
    m.add_function(wrap_pyfunction!(decay::compute_confidence_decay, m)?)?;
    m.add_function(wrap_pyfunction!(decay::time_decay_score, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(sparse::sparse_overlap, m)?)?;
    Ok(())
}
