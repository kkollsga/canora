pub mod rhythm;
pub mod spectral;

use pyo3::prelude::*;

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "feature")?;
    spectral::register(&m)?;
    rhythm::register(&m)?;
    parent.add_submodule(&m)?;
    Ok(())
}
