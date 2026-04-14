pub mod audio;
pub mod convert;
pub mod spectrum;

use pyo3::prelude::*;

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "core")?;
    audio::register(&m)?;
    convert::register(&m)?;
    spectrum::register(&m)?;
    parent.add_submodule(&m)?;
    Ok(())
}
