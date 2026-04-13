use pyo3::prelude::*;

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "util")?;
    // Utility bindings will be added as needed
    parent.add_submodule(&m)?;
    Ok(())
}
