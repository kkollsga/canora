use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use sonara::analyze as rs;
use crate::error::IntoPyResult;

fn result_to_dict<'py>(py: Python<'py>, r: &rs::TrackAnalysis) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("duration_sec", r.duration_sec)?;
    d.set_item("bpm", r.bpm)?;
    d.set_item("n_beats", r.beats.len())?;
    d.set_item("beats", r.beats.clone())?;
    d.set_item("onset_frames", r.onset_frames.clone())?;
    d.set_item("rms_mean", r.rms_mean)?;
    d.set_item("rms_max", r.rms_max)?;
    d.set_item("dynamic_range_db", r.dynamic_range_db)?;
    d.set_item("spectral_centroid_mean", r.spectral_centroid_mean)?;
    d.set_item("zero_crossing_rate", r.zero_crossing_rate)?;
    d.set_item("onset_density", r.onset_density)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(name = "analyze_file", signature = (path, *, sr=22050))]
pub fn py_analyze_file<'py>(py: Python<'py>, path: &str, sr: u32) -> PyResult<Bound<'py, PyDict>> {
    let result = rs::analyze_file(Path::new(path), sr).into_pyresult()?;
    result_to_dict(py, &result)
}

#[pyfunction]
#[pyo3(name = "analyze_signal", signature = (y, *, sr=22050))]
pub fn py_analyze_signal<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    sr: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let result = rs::analyze_signal(y.as_array(), sr).into_pyresult()?;
    result_to_dict(py, &result)
}

#[pyfunction]
#[pyo3(name = "analyze_batch", signature = (paths, *, sr=22050))]
pub fn py_analyze_batch<'py>(py: Python<'py>, paths: Vec<String>, sr: u32) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let path_refs: Vec<&Path> = paths.iter().map(|p| Path::new(p.as_str())).collect();
    let results = rs::analyze_batch(&path_refs, sr);
    results
        .into_iter()
        .map(|r| {
            let analysis = r.into_pyresult()?;
            result_to_dict(py, &analysis)
        })
        .collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_analyze_file, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_signal, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_batch, m)?)?;
    Ok(())
}
