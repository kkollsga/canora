use canora::CanoraError;
use pyo3::exceptions;
use pyo3::PyErr;

/// Convert a CanoraError into a PyErr.
/// We use a function instead of `impl From` to avoid the orphan rule.
pub fn to_pyerr(err: CanoraError) -> PyErr {
    match &err {
        CanoraError::InvalidParameter { .. }
        | CanoraError::ShapeMismatch { .. }
        | CanoraError::InvalidAudio(_)
        | CanoraError::InsufficientData { .. } => {
            exceptions::PyValueError::new_err(err.to_string())
        }
        CanoraError::AudioFile(_) | CanoraError::Decode(_) => {
            exceptions::PyIOError::new_err(err.to_string())
        }
        CanoraError::UnsupportedFormat(_) => {
            exceptions::PyNotImplementedError::new_err(err.to_string())
        }
        _ => exceptions::PyRuntimeError::new_err(err.to_string()),
    }
}

/// Extension trait to convert canora Result to PyResult.
pub trait IntoPyResult<T> {
    fn into_pyresult(self) -> pyo3::PyResult<T>;
}

impl<T> IntoPyResult<T> for canora::Result<T> {
    fn into_pyresult(self) -> pyo3::PyResult<T> {
        self.map_err(to_pyerr)
    }
}
