use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use sonara::core::convert as rs;
use crate::error::IntoPyResult;

// Scalar conversions
#[pyfunction] #[pyo3(name = "hz_to_mel", signature = (freq, *, htk=false))]
pub fn py_hz_to_mel(freq: f64, htk: bool) -> f64 { rs::hz_to_mel(freq, htk) }

#[pyfunction] #[pyo3(name = "mel_to_hz", signature = (mel, *, htk=false))]
pub fn py_mel_to_hz(mel: f64, htk: bool) -> f64 { rs::mel_to_hz(mel, htk) }

#[pyfunction] #[pyo3(name = "hz_to_midi")]
pub fn py_hz_to_midi(freq: f64) -> f64 { rs::hz_to_midi(freq) }

#[pyfunction] #[pyo3(name = "midi_to_hz")]
pub fn py_midi_to_hz(midi: f64) -> f64 { rs::midi_to_hz(midi) }

#[pyfunction] #[pyo3(name = "note_to_hz")]
pub fn py_note_to_hz(note: &str) -> PyResult<f64> { rs::note_to_hz(note).into_pyresult() }

#[pyfunction] #[pyo3(name = "note_to_midi")]
pub fn py_note_to_midi(note: &str) -> PyResult<f64> { rs::note_to_midi(note).into_pyresult() }

#[pyfunction] #[pyo3(name = "midi_to_note")]
pub fn py_midi_to_note(midi: f64) -> String { rs::midi_to_note(midi) }

#[pyfunction] #[pyo3(name = "hz_to_note")]
pub fn py_hz_to_note(freq: f64) -> String { rs::hz_to_note(freq) }

#[pyfunction] #[pyo3(name = "hz_to_octs", signature = (freq, *, tuning=0.0, bins_per_octave=12))]
pub fn py_hz_to_octs(freq: f64, tuning: f64, bins_per_octave: usize) -> f64 { rs::hz_to_octs(freq, tuning, bins_per_octave) }

#[pyfunction] #[pyo3(name = "octs_to_hz", signature = (octs, *, tuning=0.0, bins_per_octave=12))]
pub fn py_octs_to_hz(octs: f64, tuning: f64, bins_per_octave: usize) -> f64 { rs::octs_to_hz(octs, tuning, bins_per_octave) }

#[pyfunction] #[pyo3(name = "A4_to_tuning", signature = (a4, *, bins_per_octave=12))]
pub fn py_a4_to_tuning(a4: f64, bins_per_octave: usize) -> f64 { rs::a4_to_tuning(a4, bins_per_octave) }

#[pyfunction] #[pyo3(name = "tuning_to_A4", signature = (tuning, *, bins_per_octave=12))]
pub fn py_tuning_to_a4(tuning: f64, bins_per_octave: usize) -> f64 { rs::tuning_to_a4(tuning, bins_per_octave) }

// Svara conversions
#[pyfunction] #[pyo3(name = "hz_to_svara_h", signature = (freq, *, sa=261.63, abbr=true))]
pub fn py_hz_to_svara_h(freq: f64, sa: f64, abbr: bool) -> String { rs::hz_to_svara_h(freq, sa, abbr) }

#[pyfunction] #[pyo3(name = "hz_to_svara_c", signature = (freq, *, sa=261.63, abbr=true))]
pub fn py_hz_to_svara_c(freq: f64, sa: f64, abbr: bool) -> String { rs::hz_to_svara_c(freq, sa, abbr) }

#[pyfunction] #[pyo3(name = "midi_to_svara_h", signature = (midi, *, sa=60.0, abbr=true))]
pub fn py_midi_to_svara_h(midi: f64, sa: f64, abbr: bool) -> String { rs::midi_to_svara_h(midi, sa, abbr) }

#[pyfunction] #[pyo3(name = "midi_to_svara_c", signature = (midi, *, sa=60.0, abbr=true))]
pub fn py_midi_to_svara_c(midi: f64, sa: f64, abbr: bool) -> String { rs::midi_to_svara_c(midi, sa, abbr) }

#[pyfunction] #[pyo3(name = "note_to_svara_h", signature = (note, *, sa="C4", abbr=true))]
pub fn py_note_to_svara_h(note: &str, sa: &str, abbr: bool) -> PyResult<String> { rs::note_to_svara_h(note, sa, abbr).into_pyresult() }

#[pyfunction] #[pyo3(name = "note_to_svara_c", signature = (note, *, sa="C4", abbr=true))]
pub fn py_note_to_svara_c(note: &str, sa: &str, abbr: bool) -> PyResult<String> { rs::note_to_svara_c(note, sa, abbr).into_pyresult() }

#[pyfunction] #[pyo3(name = "hz_to_fjs", signature = (freq, *, ref_freq=261.63))]
pub fn py_hz_to_fjs(freq: f64, ref_freq: f64) -> String { rs::hz_to_fjs(freq, ref_freq) }

// Array generators
#[pyfunction] #[pyo3(name = "fft_frequencies", signature = (*, sr=22050.0, n_fft=2048))]
pub fn py_fft_frequencies<'py>(py: Python<'py>, sr: f64, n_fft: usize) -> Bound<'py, PyArray1<f64>> { rs::fft_frequencies(sr, n_fft).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "mel_frequencies", signature = (*, n_mels=128, fmin=0.0, fmax=11025.0, htk=false))]
pub fn py_mel_frequencies<'py>(py: Python<'py>, n_mels: usize, fmin: f64, fmax: f64, htk: bool) -> Bound<'py, PyArray1<f64>> { rs::mel_frequencies(n_mels, fmin, fmax, htk).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "cqt_frequencies", signature = (n_bins, *, fmin=32.7032, bins_per_octave=12))]
pub fn py_cqt_frequencies<'py>(py: Python<'py>, n_bins: usize, fmin: f64, bins_per_octave: usize) -> Bound<'py, PyArray1<f64>> { rs::cqt_frequencies(n_bins, fmin, bins_per_octave).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "tempo_frequencies", signature = (n_bins, *, hop_length=512, sr=22050.0))]
pub fn py_tempo_frequencies<'py>(py: Python<'py>, n_bins: usize, hop_length: usize, sr: f64) -> Bound<'py, PyArray1<f64>> { rs::tempo_frequencies(n_bins, hop_length, sr).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "fourier_tempo_frequencies", signature = (*, sr=22050.0, win_length=384, hop_length=512))]
pub fn py_fourier_tempo_frequencies<'py>(py: Python<'py>, sr: f64, win_length: usize, hop_length: usize) -> Bound<'py, PyArray1<f64>> { rs::fourier_tempo_frequencies(sr, win_length, hop_length).into_pyarray(py) }

// Frame/sample/time conversions
#[pyfunction] #[pyo3(name = "frames_to_samples", signature = (frames, *, hop_length=512))]
pub fn py_frames_to_samples(frames: Vec<usize>, hop_length: usize) -> Vec<usize> { rs::frames_to_samples(&frames, hop_length) }

#[pyfunction] #[pyo3(name = "frames_to_time", signature = (frames, *, sr=22050.0, hop_length=512))]
pub fn py_frames_to_time(frames: Vec<usize>, sr: f64, hop_length: usize) -> Vec<f64> { rs::frames_to_time(&frames, sr, hop_length) }

#[pyfunction] #[pyo3(name = "samples_to_frames", signature = (samples, *, hop_length=512))]
pub fn py_samples_to_frames(samples: Vec<usize>, hop_length: usize) -> Vec<usize> { rs::samples_to_frames(&samples, hop_length) }

#[pyfunction] #[pyo3(name = "samples_to_time", signature = (samples, *, sr=22050.0))]
pub fn py_samples_to_time(samples: Vec<usize>, sr: f64) -> Vec<f64> { rs::samples_to_time(&samples, sr) }

#[pyfunction] #[pyo3(name = "time_to_frames", signature = (times, *, sr=22050.0, hop_length=512))]
pub fn py_time_to_frames(times: Vec<f64>, sr: f64, hop_length: usize) -> Vec<usize> { rs::time_to_frames(&times, sr, hop_length) }

#[pyfunction] #[pyo3(name = "time_to_samples", signature = (times, *, sr=22050.0))]
pub fn py_time_to_samples(times: Vec<f64>, sr: f64) -> Vec<usize> { rs::time_to_samples(&times, sr) }

#[pyfunction] #[pyo3(name = "blocks_to_frames", signature = (blocks, *, block_length, hop_length=512))]
pub fn py_blocks_to_frames(blocks: Vec<usize>, block_length: usize, hop_length: usize) -> Vec<usize> { rs::blocks_to_frames(&blocks, block_length, hop_length) }

#[pyfunction] #[pyo3(name = "blocks_to_samples", signature = (blocks, *, block_length))]
pub fn py_blocks_to_samples(blocks: Vec<usize>, block_length: usize) -> Vec<usize> { rs::blocks_to_samples(&blocks, block_length) }

#[pyfunction] #[pyo3(name = "blocks_to_time", signature = (blocks, *, block_length, sr=22050.0))]
pub fn py_blocks_to_time(blocks: Vec<usize>, block_length: usize, sr: f64) -> Vec<f64> { rs::blocks_to_time(&blocks, block_length, sr) }

// Weighting
#[pyfunction] #[pyo3(name = "A_weighting")]
pub fn py_a_weighting<'py>(py: Python<'py>, frequencies: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> { frequencies.as_array().mapv(rs::a_weighting).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "B_weighting")]
pub fn py_b_weighting<'py>(py: Python<'py>, frequencies: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> { frequencies.as_array().mapv(rs::b_weighting).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "C_weighting")]
pub fn py_c_weighting<'py>(py: Python<'py>, frequencies: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> { frequencies.as_array().mapv(rs::c_weighting).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "D_weighting")]
pub fn py_d_weighting<'py>(py: Python<'py>, frequencies: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> { frequencies.as_array().mapv(rs::d_weighting).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "Z_weighting")]
pub fn py_z_weighting<'py>(py: Python<'py>, frequencies: PyReadonlyArray1<'py, f64>) -> Bound<'py, PyArray1<f64>> { frequencies.as_array().mapv(rs::z_weighting).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "frequency_weighting", signature = (frequencies, *, kind="A"))]
pub fn py_frequency_weighting<'py>(py: Python<'py>, frequencies: PyReadonlyArray1<'py, f64>, kind: &str) -> PyResult<Bound<'py, PyArray1<f64>>> { rs::frequency_weighting(frequencies.as_array(), kind).map(|r| r.into_pyarray(py)).into_pyresult() }

#[pyfunction] #[pyo3(name = "multi_frequency_weighting")]
pub fn py_multi_frequency_weighting<'py>(py: Python<'py>, frequencies: PyReadonlyArray1<'py, f64>, kinds: Vec<String>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    let kind_refs: Vec<&str> = kinds.iter().map(|s| s.as_str()).collect();
    rs::multi_frequency_weighting(frequencies.as_array(), &kind_refs).map(|r| r.into_pyarray(py)).into_pyresult()
}

// Notation
#[pyfunction] #[pyo3(name = "key_to_notes")]
pub fn py_key_to_notes(key: &str) -> PyResult<Vec<String>> { sonara::core::notation::key_to_notes(key).into_pyresult() }

#[pyfunction] #[pyo3(name = "key_to_degrees")]
pub fn py_key_to_degrees(key: &str) -> PyResult<Vec<usize>> { sonara::core::notation::key_to_degrees(key).into_pyresult() }

#[pyfunction] #[pyo3(name = "mela_to_degrees")]
pub fn py_mela_to_degrees(mela: usize) -> PyResult<Vec<usize>> { sonara::core::notation::mela_to_degrees(mela).into_pyresult() }

#[pyfunction] #[pyo3(name = "mela_to_svara")]
pub fn py_mela_to_svara(mela: usize) -> PyResult<Vec<String>> { sonara::core::notation::mela_to_svara(mela).into_pyresult() }

#[pyfunction] #[pyo3(name = "thaat_to_degrees")]
pub fn py_thaat_to_degrees(thaat: &str) -> PyResult<Vec<usize>> { sonara::core::notation::thaat_to_degrees(thaat).into_pyresult() }

#[pyfunction] #[pyo3(name = "list_mela")]
pub fn py_list_mela() -> Vec<String> { sonara::core::notation::list_mela() }

#[pyfunction] #[pyo3(name = "list_thaat")]
pub fn py_list_thaat() -> Vec<String> { sonara::core::notation::list_thaat() }

#[pyfunction] #[pyo3(name = "fifths_to_note")]
pub fn py_fifths_to_note(fifths: i32) -> String { sonara::core::notation::fifths_to_note(fifths) }

#[pyfunction] #[pyo3(name = "interval_to_fjs")]
pub fn py_interval_to_fjs(interval: f64) -> String { sonara::core::notation::interval_to_fjs(interval) }

#[pyfunction] #[pyo3(name = "interval_frequencies", signature = (n_bins, *, fmin=32.7, intervals, bins_per_octave=12))]
pub fn py_interval_frequencies<'py>(py: Python<'py>, n_bins: usize, fmin: f64, intervals: Vec<f64>, bins_per_octave: usize) -> Bound<'py, PyArray1<f64>> { sonara::core::intervals::interval_frequencies(n_bins, fmin, &intervals, bins_per_octave).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "pythagorean_intervals", signature = (*, bins_per_octave=12))]
pub fn py_pythagorean_intervals<'py>(py: Python<'py>, bins_per_octave: usize) -> Bound<'py, PyArray1<f64>> { sonara::core::intervals::pythagorean_intervals(bins_per_octave).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "plimit_intervals", signature = (*, p=5, bins_per_octave=12))]
pub fn py_plimit_intervals<'py>(py: Python<'py>, p: usize, bins_per_octave: usize) -> Bound<'py, PyArray1<f64>> { sonara::core::intervals::plimit_intervals(p, bins_per_octave).into_pyarray(py) }

// Utility
#[pyfunction] #[pyo3(name = "samples_like", signature = (n_frames, *, hop_length=512))]
pub fn py_samples_like<'py>(py: Python<'py>, n_frames: usize, hop_length: usize) -> Bound<'py, numpy::PyArray1<usize>> { rs::samples_like(n_frames, hop_length).into_pyarray(py) }

#[pyfunction] #[pyo3(name = "times_like", signature = (n_frames, *, sr=22050.0, hop_length=512))]
pub fn py_times_like<'py>(py: Python<'py>, n_frames: usize, sr: f64, hop_length: usize) -> Bound<'py, PyArray1<f64>> { rs::times_like(n_frames, sr, hop_length).into_pyarray(py) }

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // All scalar conversions
    m.add_function(wrap_pyfunction!(py_hz_to_mel, m)?)?;
    m.add_function(wrap_pyfunction!(py_mel_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(py_hz_to_midi, m)?)?;
    m.add_function(wrap_pyfunction!(py_midi_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(py_note_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(py_note_to_midi, m)?)?;
    m.add_function(wrap_pyfunction!(py_midi_to_note, m)?)?;
    m.add_function(wrap_pyfunction!(py_hz_to_note, m)?)?;
    m.add_function(wrap_pyfunction!(py_hz_to_octs, m)?)?;
    m.add_function(wrap_pyfunction!(py_octs_to_hz, m)?)?;
    m.add_function(wrap_pyfunction!(py_a4_to_tuning, m)?)?;
    m.add_function(wrap_pyfunction!(py_tuning_to_a4, m)?)?;
    m.add_function(wrap_pyfunction!(py_hz_to_svara_h, m)?)?;
    m.add_function(wrap_pyfunction!(py_hz_to_svara_c, m)?)?;
    m.add_function(wrap_pyfunction!(py_midi_to_svara_h, m)?)?;
    m.add_function(wrap_pyfunction!(py_midi_to_svara_c, m)?)?;
    m.add_function(wrap_pyfunction!(py_note_to_svara_h, m)?)?;
    m.add_function(wrap_pyfunction!(py_note_to_svara_c, m)?)?;
    m.add_function(wrap_pyfunction!(py_hz_to_fjs, m)?)?;
    // Array generators
    m.add_function(wrap_pyfunction!(py_fft_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(py_mel_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(py_cqt_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(py_tempo_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(py_fourier_tempo_frequencies, m)?)?;
    // Frame conversions
    m.add_function(wrap_pyfunction!(py_frames_to_samples, m)?)?;
    m.add_function(wrap_pyfunction!(py_frames_to_time, m)?)?;
    m.add_function(wrap_pyfunction!(py_samples_to_frames, m)?)?;
    m.add_function(wrap_pyfunction!(py_samples_to_time, m)?)?;
    m.add_function(wrap_pyfunction!(py_time_to_frames, m)?)?;
    m.add_function(wrap_pyfunction!(py_time_to_samples, m)?)?;
    m.add_function(wrap_pyfunction!(py_blocks_to_frames, m)?)?;
    m.add_function(wrap_pyfunction!(py_blocks_to_samples, m)?)?;
    m.add_function(wrap_pyfunction!(py_blocks_to_time, m)?)?;
    // Weighting
    m.add_function(wrap_pyfunction!(py_a_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(py_b_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(py_c_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(py_d_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(py_z_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(py_frequency_weighting, m)?)?;
    m.add_function(wrap_pyfunction!(py_multi_frequency_weighting, m)?)?;
    // Notation
    m.add_function(wrap_pyfunction!(py_key_to_notes, m)?)?;
    m.add_function(wrap_pyfunction!(py_key_to_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(py_mela_to_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(py_mela_to_svara, m)?)?;
    m.add_function(wrap_pyfunction!(py_thaat_to_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(py_list_mela, m)?)?;
    m.add_function(wrap_pyfunction!(py_list_thaat, m)?)?;
    m.add_function(wrap_pyfunction!(py_fifths_to_note, m)?)?;
    m.add_function(wrap_pyfunction!(py_interval_to_fjs, m)?)?;
    m.add_function(wrap_pyfunction!(py_interval_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(py_pythagorean_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(py_plimit_intervals, m)?)?;
    // Utility
    m.add_function(wrap_pyfunction!(py_samples_like, m)?)?;
    m.add_function(wrap_pyfunction!(py_times_like, m)?)?;
    Ok(())
}
