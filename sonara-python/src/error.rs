use sonara::SonaraError;
use pyo3::exceptions;
use pyo3::PyErr;

/// Convert a SonaraError into a PyErr.
/// We use a function instead of `impl From` to avoid the orphan rule.
pub fn to_pyerr(err: SonaraError) -> PyErr {
    match &err {
        SonaraError::InvalidParameter { .. }
        | SonaraError::ShapeMismatch { .. }
        | SonaraError::InvalidAudio(_)
        | SonaraError::InsufficientData { .. } => {
            exceptions::PyValueError::new_err(err.to_string())
        }
        SonaraError::AudioFile(_) | SonaraError::Decode(_) => {
            exceptions::PyIOError::new_err(err.to_string())
        }
        SonaraError::UnsupportedFormat(_) => {
            exceptions::PyNotImplementedError::new_err(err.to_string())
        }
        _ => exceptions::PyRuntimeError::new_err(err.to_string()),
    }
}

/// Extension trait to convert sonara Result to PyResult.
pub trait IntoPyResult<T> {
    fn into_pyresult(self) -> pyo3::PyResult<T>;
}

impl<T> IntoPyResult<T> for sonara::Result<T> {
    fn into_pyresult(self) -> pyo3::PyResult<T> {
        self.map_err(to_pyerr)
    }
}
