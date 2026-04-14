use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use sonara::filters as rs;

#[pyfunction]
#[pyo3(name = "mel", signature = (*, sr=22050.0, n_fft=2048, n_mels=128, fmin=0.0, fmax=0.0, htk=false, norm="slaney"))]
pub fn py_mel<'py>(
    py: Python<'py>,
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
    htk: bool,
    norm: &str,
) -> Bound<'py, PyArray2<f64>> {
    rs::mel(sr, n_fft, n_mels, fmin, fmax, htk, norm).into_pyarray(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_mel, m)?)?;
    Ok(())
}
