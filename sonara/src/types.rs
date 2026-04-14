use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD};
use num_complex::Complex;

// ---- Scalar precision ----
/// Default floating-point type for all internal computation.
/// Using f32 for performance: halves memory bandwidth, matches native decode
/// format (Symphonia decodes to f32), and is standard for audio analysis
/// (aubio, essentia, Core Audio all default to f32).
pub type Float = f32;

/// Complex float used for spectral representations.
pub type ComplexFloat = Complex<f32>;

// ---- Audio domain ----
/// Mono audio signal: 1-D array of f32 samples in [-1.0, 1.0].
pub type AudioBuffer = Array1<Float>;

/// Borrowed view of a mono audio signal.
pub type AudioBufferView<'a> = ArrayView1<'a, Float>;

/// Multi-channel audio: shape (n_channels, n_samples).
pub type MultiChannelAudio = Array2<Float>;

// ---- Spectral domain ----
/// Complex STFT matrix: shape (1 + n_fft/2, n_frames).
pub type Stft = Array2<ComplexFloat>;

/// Borrowed view of an STFT matrix.
pub type StftView<'a> = ArrayView2<'a, ComplexFloat>;

/// Real-valued spectrogram (magnitude, power, mel, etc.): shape (n_bins, n_frames).
pub type Spectrogram = Array2<Float>;

/// Borrowed view of a real spectrogram.
pub type SpectrogramView<'a> = ArrayView2<'a, Float>;

// ---- Filter domain ----
/// Filter bank matrix: shape (n_filters, n_fft_bins).
pub type FilterBank = Array2<Float>;

// ---- Dynamic shapes ----
/// Dynamic-dimensional real array.
pub type ArrayDynFloat = ArrayD<Float>;

/// Borrowed view of a dynamic-dimensional real array.
#[allow(dead_code)]
pub type ArrayDynFloatView<'a> = ArrayViewD<'a, Float>;

/// Dynamic-dimensional complex array.
pub type ArrayDynComplex = ArrayD<ComplexFloat>;

// ---- Window specification ----
/// How a window function is specified — mirrors librosa's `_WindowSpec`.
#[derive(Debug, Clone)]
pub enum WindowSpec {
    /// Named window: "hann", "hamming", "blackman", etc.
    Named(String),
    /// Pre-computed window samples.
    Array(Array1<Float>),
    /// Parameterized window: (name, parameter), e.g. ("kaiser", 14.0).
    Parameterized(String, Float),
}

// ---- Pad mode ----
/// Padding modes matching numpy's `np.pad` (subset used by librosa).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Constant,
    Edge,
    Reflect,
    Symmetric,
    LinearRamp,
}

impl Default for PadMode {
    fn default() -> Self {
        PadMode::Constant
    }
}

impl PadMode {
    /// Parse a pad mode string (case-insensitive).
    pub fn from_str(s: &str) -> crate::Result<Self> {
        match s.to_lowercase().as_str() {
            "constant" | "zero" => Ok(PadMode::Constant),
            "edge" => Ok(PadMode::Edge),
            "reflect" => Ok(PadMode::Reflect),
            "symmetric" => Ok(PadMode::Symmetric),
            "linear_ramp" => Ok(PadMode::LinearRamp),
            _ => Err(crate::SonaraError::InvalidParameter {
                param: "pad_mode",
                reason: format!("unsupported pad mode: '{s}'"),
            }),
        }
    }
}
