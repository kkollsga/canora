use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use sonara::beat as rs;
use crate::error::IntoPyResult;

#[pyfunction]
#[pyo3(name = "beat_track", signature = (*, y=None, onset_envelope=None, sr=22050, hop_length=512, start_bpm=120.0, tightness=100.0, trim=true))]
pub fn py_beat_track(
    y: Option<PyReadonlyArray1<'_, f32>>,
    onset_envelope: Option<PyReadonlyArray1<'_, f32>>,
    sr: u32, hop_length: usize, start_bpm: f32, tightness: f32, trim: bool,
) -> PyResult<(f32, Vec<usize>)> {
    let yv = y.as_ref().map(|a| a.as_array());
    let ev = onset_envelope.as_ref().map(|a| a.as_array());
    rs::beat_track(yv, ev, sr, hop_length, start_bpm, tightness, trim).into_pyresult()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_beat_track, m)?)?;
    Ok(())
}
