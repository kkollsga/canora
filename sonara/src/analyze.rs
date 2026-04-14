//! Fused audio analysis pipeline.
//!
//! Computes all common audio features in a single optimized pass,
//! eliminating redundant STFT computation.
//!
//! ## Modes
//!
//! - **Compact** (default): core signal features — tempo, beats, onsets, RMS,
//!   centroid, ZCR, dynamic range. ~0.6ms per 10s track.
//! - **Playlist**: everything needed for playlist generation — adds spectral
//!   features, MFCCs, chroma, plus perceptual features (energy, danceability,
//!   key, valence, acousticness). ~3ms per 10s track.
//! - **Full**: same features as Playlist (currently identical, reserved for
//!   future additions like per-frame arrays or segment-level analysis).
//!
//! ## Accuracy
//!
//! The `accurate` flag trades speed for precision on select features:
//! - **Chroma**: mel-band approximation (±1 semitone) → proper chroma filterbank
//! - **Spectral contrast**: mel sub-bands → log-spaced frequency bands on magnitude spectrum
//! - **Danceability**: beat heuristic → Detrended Fluctuation Analysis (Streich & Herrera 2005)
//! - Requires storing the full power spectrogram (~2x memory for the FFT pass).

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::Path;

use ndarray::{s, Array1, Array2};
use rayon::prelude::*;

use crate::core::{audio, convert, fft, spectrum};
use crate::dsp::windows;
use crate::error::{SonaraError, Result};
use crate::filters;
use crate::perceptual;
use crate::types::*;
use crate::util::utils;

/// Minimum number of frames to justify rayon thread overhead.
const PARALLEL_THRESHOLD: usize = 32;

// ============================================================
// Analysis mode & feature selection
// ============================================================

/// Analysis depth — controls which features are computed.
///
/// Use `AnalysisMode::Compact` for fast scanning, `Playlist` for music discovery
/// and playlist generation, or `Full` for comprehensive analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisMode {
    /// Core signal features only: tempo, beats, onsets, RMS, centroid, ZCR,
    /// dynamic range. (~0.6ms per 10s track)
    Compact,
    /// All features for playlist generation: adds spectral bandwidth/rolloff/
    /// flatness/contrast, MFCCs, chroma, plus perceptual features (energy,
    /// danceability, key, valence, acousticness). (~3ms per 10s track)
    Playlist,
    /// All available features. Currently identical to Playlist; reserved for
    /// future additions (per-frame arrays, segment analysis). (~3ms per 10s track)
    Full,
}

impl AnalysisMode {
    /// Parse mode from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "compact" => Some(Self::Compact),
            "playlist" => Some(Self::Playlist),
            "full" => Some(Self::Full),
            _ => None,
        }
    }
}

impl Default for AnalysisMode {
    fn default() -> Self {
        Self::Compact
    }
}

/// Configuration for a single analysis run.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Analysis depth — which feature groups to compute.
    pub mode: AnalysisMode,
    /// Use accurate (but slower) algorithms for chroma, spectral contrast,
    /// and danceability. Requires storing the full power spectrogram.
    pub accurate: bool,
    /// Optional: override which features to include, regardless of mode.
    /// When `Some`, only the listed features are computed.
    /// Valid feature names (case-insensitive):
    ///
    /// **Core signal:**
    /// `bpm`, `beats`, `onsets`, `rms`, `dynamic_range`, `centroid`, `zcr`, `onset_density`
    ///
    /// **Spectral:**
    /// `bandwidth`, `rolloff`, `flatness`, `contrast`, `mfcc`, `chroma`
    ///
    /// **Perceptual:**
    /// `energy`, `danceability`, `key`, `valence`, `acousticness`
    ///
    /// Note: `duration` is always included. Some features depend on others
    /// (e.g., `key` requires `chroma`, `valence` requires `key`); dependencies
    /// are resolved automatically.
    pub features: Option<HashSet<String>>,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            mode: AnalysisMode::Compact,
            accurate: false,
            features: None,
        }
    }
}

impl AnalysisConfig {
    /// Check if a feature should be computed.
    fn wants(&self, name: &str) -> bool {
        if let Some(ref features) = self.features {
            // Explicit feature list — check if requested
            features.contains(name)
        } else {
            // Mode-based defaults
            match self.mode {
                AnalysisMode::Compact => false, // only core (handled separately)
                AnalysisMode::Playlist | AnalysisMode::Full => true,
            }
        }
    }

    /// Check if extended features (anything beyond compact) are needed.
    fn needs_extended(&self) -> bool {
        if let Some(ref features) = self.features {
            // If any non-core feature is requested
            const EXTENDED_FEATURES: &[&str] = &[
                "bandwidth", "rolloff", "flatness", "contrast", "mfcc", "chroma",
                "energy", "danceability", "key", "valence", "acousticness",
            ];
            EXTENDED_FEATURES.iter().any(|&f| features.contains(f))
        } else {
            self.mode != AnalysisMode::Compact
        }
    }

    /// Check if the power spectrogram needs to be stored (for accurate chroma/contrast).
    pub fn needs_power_spec(&self) -> bool {
        self.accurate && self.needs_extended()
    }
}

/// Cached mel filterbank and sparse representation.
struct MelCache {
    key: (u32, usize, usize), // (sr, n_fft, n_mels)
    sparse_mel: Vec<(usize, Vec<Float>)>,
    freqs: Array1<Float>,
    win_padded: Array1<Float>,
}

thread_local! {
    static MEL_CACHE: RefCell<Option<MelCache>> = const { RefCell::new(None) };
}

/// Complete analysis result for a single track.
///
/// Core fields are always populated. Extended/perceptual fields are `Some`
/// only when the selected mode or feature list includes them.
pub struct TrackAnalysis {
    // -- Basic (always computed) --
    pub duration_sec: Float,
    pub bpm: Float,
    pub beats: Vec<usize>,
    pub onset_frames: Vec<usize>,
    pub rms_mean: Float,
    pub rms_max: Float,
    pub loudness_lufs: Float,
    pub dynamic_range_db: Float,
    pub spectral_centroid_mean: Float,
    pub zero_crossing_rate: Float,
    pub onset_density: Float,

    // -- Extended (extended or full) --
    pub spectral_bandwidth_mean: Option<Float>,
    pub spectral_rolloff_mean: Option<Float>,
    pub spectral_flatness_mean: Option<Float>,
    pub spectral_contrast_mean: Option<Vec<Float>>,
    pub mfcc_mean: Option<Vec<Float>>,
    pub chroma_mean: Option<Vec<Float>>,

    // -- Perceptual (extended or full) --
    pub energy: Option<Float>,
    pub danceability: Option<Float>,
    pub key: Option<String>,
    pub key_confidence: Option<Float>,
    pub valence: Option<Float>,
    pub acousticness: Option<Float>,

    // -- Tier 3 placeholders (future ML models) --
    /// Requires ML model. See essentia's TensorFlow-based classifiers.
    pub mood_happy: Option<Float>,
    pub mood_aggressive: Option<Float>,
    pub mood_relaxed: Option<Float>,
    pub mood_sad: Option<Float>,
    pub instrumentalness: Option<Float>,
    pub genre: Option<String>,
}

/// Per-frame results from the fused FFT pass.
struct FrameResult {
    mel_col: Vec<Float>,
    centroid: Float,
    rms: Float,
    bandwidth: Float,
    rolloff: Float,
    flatness: Float,
    power_col: Option<Vec<Float>>, // saved for full mode (chroma + contrast)
}

// ============================================================
// Public API
// ============================================================

/// Analyze a track from a file path with the given configuration.
pub fn analyze_file(path: &Path, sr: u32, config: &AnalysisConfig) -> Result<TrackAnalysis> {
    let (y, actual_sr) = audio::load(path, sr, true, 0.0, 0.0)?;
    analyze_signal(y.view(), actual_sr, config)
}

/// Analyze a pre-loaded audio signal with the given configuration.
pub fn analyze_signal(
    y: ndarray::ArrayView1<Float>,
    sr: u32,
    config: &AnalysisConfig,
) -> Result<TrackAnalysis> {
    let extended = config.needs_extended();
    let accurate = config.accurate && extended;
    analyze_signal_inner(y, sr, extended, accurate, config)
}

/// Analyze multiple files in parallel.
pub fn analyze_batch(paths: &[&Path], sr: u32, config: &AnalysisConfig) -> Vec<Result<TrackAnalysis>> {
    paths.par_iter().map(|path| analyze_file(path, sr, config)).collect()
}

// ============================================================
// Core implementation
// ============================================================

fn analyze_signal_inner(
    y: ndarray::ArrayView1<Float>,
    sr: u32,
    extended: bool,
    accurate: bool,
    config: &AnalysisConfig,
) -> Result<TrackAnalysis> {
    let sr_f = sr as Float;
    let n_fft = 2048;
    let hop_length = 512;
    let n_mels = 128;
    let n_bins = n_fft / 2 + 1;

    let duration_sec = y.len() as Float / sr_f;

    // ================================================================
    // SETUP: mel filterbank, window, padding (cached across calls)
    // ================================================================

    let cache_key = (sr, n_fft, n_mels);

    let (sparse_mel, freqs, win_padded) = MEL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(ref c) = *cache {
            if c.key == cache_key {
                return (c.sparse_mel.clone(), c.freqs.clone(), c.win_padded.clone());
            }
        }

        let mel_fb = filters::mel(sr_f, n_fft, n_mels, 0.0, sr_f / 2.0, false, "slaney");
        let sparse: Vec<(usize, Vec<Float>)> = (0..n_mels)
            .map(|m| {
                let row = mel_fb.row(m);
                let first = row.iter().position(|&v| v > 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&v| v > 0.0).unwrap_or(0);
                if first > last { (0, vec![]) }
                else { (first, row.slice(s![first..=last]).to_vec()) }
            })
            .collect();
        let f = convert::fft_frequencies(sr_f, n_fft);
        let win = windows::get_window(&WindowSpec::Named("hann".into()), n_fft, true)
            .expect("hann window");
        let wp = utils::pad_center(win.view(), n_fft).expect("pad_center");

        *cache = Some(MelCache {
            key: cache_key,
            sparse_mel: sparse.clone(),
            freqs: f.clone(),
            win_padded: wp.clone(),
        });

        (sparse, f, wp)
    });

    let pad = n_fft / 2;
    let mut y_padded = Array1::<Float>::zeros(y.len() + 2 * pad);
    y_padded.slice_mut(s![pad..pad + y.len()]).assign(&y);
    let n = y_padded.len();
    if n < n_fft {
        return Err(SonaraError::InsufficientData { needed: n_fft, got: n });
    }
    let y_raw = y_padded.as_slice().unwrap();
    let win_raw = win_padded.as_slice().unwrap();

    // ================================================================
    // SINGLE PASS: FFT → mel + centroid + rms + (extended features)
    // In full mode, also saves the power spectrum for accurate chroma/contrast.
    // ================================================================

    let n_frames = 1 + (n - n_fft) / hop_length;
    let mut mel_spec = Array2::<Float>::zeros((n_mels, n_frames));
    let mut centroids = Array1::<Float>::zeros(n_frames);
    let mut rms_frames = Array1::<Float>::zeros(n_frames);
    let mut bandwidths = if extended { Array1::<Float>::zeros(n_frames) } else { Array1::zeros(0) };
    let mut rolloffs = if extended { Array1::<Float>::zeros(n_frames) } else { Array1::zeros(0) };
    let mut flatnesses = if extended { Array1::<Float>::zeros(n_frames) } else { Array1::zeros(0) };
    // Full mode: store power spectrogram for accurate chroma + contrast
    let mut power_spec = if accurate {
        Array2::<Float>::zeros((n_bins, n_frames))
    } else {
        Array2::zeros((0, 0))
    };

    let freqs_raw = freqs.as_slice().unwrap();
    let roll_percent: Float = 0.85;

    let compute_frame = |t: usize| -> FrameResult {
        let start = t * hop_length;
        let mut fft_in = vec![0.0_f32; n_fft];
        for i in 0..n_fft { fft_in[i] = y_raw[start + i] * win_raw[i]; }
        let mut fft_out = vec![num_complex::Complex::new(0.0, 0.0); n_bins];
        fft::rfft(&mut fft_in, &mut fft_out).expect("FFT failed");

        let mut cent_num = 0.0_f32;
        let mut cent_den = 0.0_f32;
        let mut power_col = vec![0.0_f32; n_bins];

        for i in 0..n_bins {
            let pwr = fft_out[i].norm_sqr();
            power_col[i] = pwr;
            let mag = pwr.sqrt();
            cent_num += freqs_raw[i] * mag;
            cent_den += mag;
        }

        let centroid = if cent_den > 0.0 { cent_num / cent_den } else { 0.0 };

        // RMS from time-domain (not windowed FFT)
        let mut sum_sq = 0.0_f32;
        for i in 0..n_fft { sum_sq += y_raw[start + i] * y_raw[start + i]; }
        let rms = (sum_sq / n_fft as Float).sqrt();

        let (bandwidth, rolloff, flatness) = if extended {
            let bw = if cent_den > 0.0 {
                let mut bw_num = 0.0_f32;
                for i in 0..n_bins {
                    let mag = power_col[i].sqrt();
                    let dev = freqs_raw[i] - centroid;
                    bw_num += mag * dev * dev;
                }
                (bw_num / cent_den).sqrt()
            } else { 0.0 };

            let total_mag: Float = (0..n_bins).map(|i| power_col[i].sqrt()).sum();
            let threshold = roll_percent * total_mag;
            let mut cumsum = 0.0_f32;
            let mut ro = 0.0_f32;
            for i in 0..n_bins {
                cumsum += power_col[i].sqrt();
                if cumsum >= threshold {
                    ro = freqs_raw[i];
                    break;
                }
            }

            let amin: Float = 1e-10;
            let mut log_sum = 0.0_f32;
            let mut arith_sum = 0.0_f32;
            for i in 0..n_bins {
                let v = power_col[i].max(amin);
                log_sum += v.ln();
                arith_sum += v;
            }
            let geo_mean = (log_sum / n_bins as Float).exp();
            let arith_mean = arith_sum / n_bins as Float;
            let fl = if arith_mean > 0.0 { geo_mean / arith_mean } else { 0.0 };

            (bw, ro, fl)
        } else {
            (0.0, 0.0, 0.0)
        };

        // Sparse mel projection
        let mel_col: Vec<Float> = sparse_mel.iter().map(|(start_bin, weights)| {
            let mut sum = 0.0;
            for (k, &w) in weights.iter().enumerate() { sum += w * power_col[start_bin + k]; }
            sum
        }).collect();

        let saved_power = if accurate { Some(power_col) } else { None };

        FrameResult { mel_col, centroid, rms, bandwidth, rolloff, flatness, power_col: saved_power }
    };

    if n_frames >= PARALLEL_THRESHOLD {
        let frame_results: Vec<FrameResult> = (0..n_frames)
            .into_par_iter()
            .map(|t| compute_frame(t))
            .collect();

        for (t, fr) in frame_results.into_iter().enumerate() {
            centroids[t] = fr.centroid;
            rms_frames[t] = fr.rms;
            if extended {
                bandwidths[t] = fr.bandwidth;
                rolloffs[t] = fr.rolloff;
                flatnesses[t] = fr.flatness;
            }
            for (m, val) in fr.mel_col.into_iter().enumerate() {
                mel_spec[(m, t)] = val;
            }
            if let Some(pc) = fr.power_col {
                for (i, val) in pc.into_iter().enumerate() {
                    power_spec[(i, t)] = val;
                }
            }
        }
    } else {
        for t in 0..n_frames {
            let fr = compute_frame(t);
            centroids[t] = fr.centroid;
            rms_frames[t] = fr.rms;
            if extended {
                bandwidths[t] = fr.bandwidth;
                rolloffs[t] = fr.rolloff;
                flatnesses[t] = fr.flatness;
            }
            for (m, val) in fr.mel_col.into_iter().enumerate() {
                mel_spec[(m, t)] = val;
            }
            if let Some(pc) = fr.power_col {
                for (i, val) in pc.into_iter().enumerate() {
                    power_spec[(i, t)] = val;
                }
            }
        }
    }

    // ================================================================
    // ONSET STRENGTH from mel spectrogram (no additional FFT)
    // ================================================================

    let s_db = spectrum::power_to_db(mel_spec.view(), 1.0, 1e-10, Some(80.0));
    let lag = 1usize;

    let out_frames = if n_frames > lag { n_frames - lag } else { 0 };
    let mut onset_env = Array1::<Float>::zeros(out_frames);
    for t in 0..out_frames {
        let mut sum = 0.0;
        for m in 0..n_mels {
            sum += (s_db[(m, t + lag)] - s_db[(m, t)]).max(0.0);
        }
        onset_env[t] = sum / n_mels as Float;
    }

    let pad_left = lag + n_fft / (2 * hop_length);
    let total_oenv_frames = out_frames + pad_left;
    let mut oenv_padded = Array1::<Float>::zeros(total_oenv_frames);
    for t in 0..out_frames { oenv_padded[pad_left + t] = onset_env[t]; }

    // ================================================================
    // BEAT TRACKING + ONSET DETECTION
    // ================================================================

    let (bpm, beats) = crate::beat::beat_track(
        None, Some(oenv_padded.view()), sr, hop_length, 120.0, 100.0, true,
    )?;

    let onset_frames = crate::onset::onset_detect(
        None, Some(oenv_padded.view()), sr, hop_length, false, 0.07, 0,
    )?;

    // ================================================================
    // Zero crossings (trivial, time-domain)
    // ================================================================

    let zc = audio::zero_crossings(y, 0.0);
    let zcr = zc.iter().filter(|&&v| v).count() as Float / y.len() as Float;

    // ================================================================
    // LUFS integrated loudness (ITU-R BS.1770-4, K-weighted)
    // ================================================================

    let loudness_lufs = perceptual::loudness_lufs(y, sr);

    // ================================================================
    // EXTENDED: MFCCs from mel spectrogram (DCT, no extra FFT)
    // ================================================================

    let n_mfcc = 13;
    let mfcc_mean = if extended {
        let n_dct = n_mels;
        let mut mfcc_avg = vec![0.0_f32; n_mfcc];
        for t in 0..n_frames {
            for k in 0..n_mfcc {
                let mut sum = 0.0_f32;
                for m in 0..n_dct {
                    sum += s_db[(m, t)] * (std::f32::consts::PI * k as Float * (2 * m + 1) as Float / (2 * n_dct) as Float).cos();
                }
                let norm = if k == 0 { (1.0 / n_dct as Float).sqrt() } else { (2.0 / n_dct as Float).sqrt() };
                mfcc_avg[k] += sum * norm;
            }
        }
        for v in mfcc_avg.iter_mut() { *v /= n_frames.max(1) as Float; }
        Some(mfcc_avg)
    } else {
        None
    };

    // ================================================================
    // CHROMA: mel-approximated (extended) or accurate via filterbank (full)
    // ================================================================

    let chroma_mean = if accurate && n_frames > 0 {
        // Full mode: use the proper chroma filterbank on the saved power spectrogram.
        // This is the same method as the standalone chroma_stft function.
        let chroma_fb = filters::chroma(sr_f, n_fft, 12, 0.0);
        let mut chroma_avg = vec![0.0_f32; 12];

        for t in 0..n_frames {
            let mut frame_chroma = [0.0_f32; 12];
            for c in 0..12 {
                let mut sum = 0.0;
                for f in 0..n_bins.min(chroma_fb.ncols()) {
                    sum += chroma_fb[(c, f)] * power_spec[(f, t)];
                }
                frame_chroma[c] = sum;
            }
            // L-inf normalize
            let max_val = frame_chroma.iter().copied().fold(0.0_f32, Float::max);
            if max_val > 0.0 {
                for v in frame_chroma.iter_mut() { *v /= max_val; }
            }
            for (i, &v) in frame_chroma.iter().enumerate() {
                chroma_avg[i] += v;
            }
        }
        for v in chroma_avg.iter_mut() { *v /= n_frames as Float; }
        Some(chroma_avg)
    } else if extended && n_frames > 0 {
        // Extended mode: fast mel-to-chroma approximation (±1 bin accuracy)
        let mut chroma_avg = vec![0.0_f32; 12];
        let c0: Float = 16.352;
        let mel_hi = convert::hz_to_mel(sr_f / 2.0, false);
        let mel_chroma_map: Vec<(usize, usize, Float)> = (0..n_mels)
            .map(|m| {
                let mel_center = mel_hi * (m as Float + 0.5) / n_mels as Float;
                let freq = convert::mel_to_hz(mel_center, false);
                if freq > c0 {
                    let chroma_val = ((12.0 * (freq / c0).log2()) % 12.0 + 12.0) % 12.0;
                    let bin_lo = chroma_val.floor() as usize % 12;
                    let bin_hi = (bin_lo + 1) % 12;
                    let frac = chroma_val - chroma_val.floor();
                    (bin_lo, bin_hi, frac)
                } else {
                    (0, 0, 0.0)
                }
            })
            .collect();

        for t in 0..n_frames {
            let mut frame_chroma = [0.0_f32; 12];
            for m in 0..n_mels {
                let (lo, hi, frac) = mel_chroma_map[m];
                let energy = mel_spec[(m, t)];
                frame_chroma[lo] += energy * (1.0 - frac);
                frame_chroma[hi] += energy * frac;
            }
            let max_val = frame_chroma.iter().copied().fold(0.0_f32, Float::max);
            if max_val > 0.0 {
                for v in frame_chroma.iter_mut() { *v /= max_val; }
            }
            for (i, &v) in frame_chroma.iter().enumerate() {
                chroma_avg[i] += v;
            }
        }
        for v in chroma_avg.iter_mut() { *v /= n_frames as Float; }
        Some(chroma_avg)
    } else {
        None
    };

    // ================================================================
    // SPECTRAL CONTRAST: mel-approximated (extended) or accurate (full)
    // ================================================================

    let n_contrast_bands = 6;
    let spectral_contrast_mean = if accurate && n_frames > 0 {
        // Full mode: proper log-spaced frequency bands on the magnitude spectrum
        // (same algorithm as the standalone spectral_contrast function)
        let fmin: Float = 200.0;
        let fmax = sr_f / 2.0;
        let quantile: Float = 0.02;

        let mut band_edges = vec![fmin];
        for i in 1..=n_contrast_bands {
            band_edges.push(fmin * (fmax / fmin).powf(i as Float / n_contrast_bands as Float));
        }

        // Pre-compute bin ranges for each band
        let band_bins: Vec<(usize, usize)> = (0..n_contrast_bands)
            .map(|b| {
                let lo = band_edges[b];
                let hi = band_edges[b + 1];
                let start = freqs_raw.iter().position(|&f| f >= lo).unwrap_or(0);
                let end = freqs_raw.iter().position(|&f| f >= hi).unwrap_or(n_bins);
                (start, end)
            })
            .collect();

        let mut contrast_avg = vec![0.0_f32; n_contrast_bands + 1];

        for t in 0..n_frames {
            for (b, &(start_bin, end_bin)) in band_bins.iter().enumerate() {
                if start_bin >= end_bin { continue; }
                let mut band_vals: Vec<Float> = (start_bin..end_bin)
                    .map(|f| power_spec[(f, t)].sqrt().max(1e-10))
                    .collect();
                band_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let bn = band_vals.len();
                let q_idx = ((bn as Float * quantile) as usize).min(bn - 1);
                let valley = band_vals[q_idx];
                let peak = band_vals[(bn - 1).saturating_sub(q_idx)];
                contrast_avg[b] += peak.log10() - valley.log10();
            }
            let mean_mag: Float = (0..n_bins).map(|f| power_spec[(f, t)].sqrt()).sum::<Float>() / n_bins as Float;
            contrast_avg[n_contrast_bands] += mean_mag.max(1e-10).log10();
        }
        for v in contrast_avg.iter_mut() { *v /= n_frames as Float; }
        Some(contrast_avg)
    } else if extended && n_frames > 0 {
        // Extended mode: approximate using mel sub-bands
        let mut contrast_avg = vec![0.0_f32; n_contrast_bands + 1];
        let bands_per_group = n_mels / n_contrast_bands;

        for t in 0..n_frames {
            for b in 0..n_contrast_bands {
                let start_m = b * bands_per_group;
                let end_m = ((b + 1) * bands_per_group).min(n_mels);
                let mut band_vals: Vec<Float> = (start_m..end_m)
                    .map(|m| mel_spec[(m, t)].max(1e-10))
                    .collect();
                band_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let bn = band_vals.len();
                if bn > 0 {
                    let q = (bn as Float * 0.02) as usize;
                    let valley = band_vals[q.min(bn - 1)];
                    let peak = band_vals[(bn - 1).saturating_sub(q)];
                    contrast_avg[b] += peak.log10() - valley.log10();
                }
            }
            let mean_val: Float = (0..n_mels).map(|m| mel_spec[(m, t)]).sum::<Float>() / n_mels as Float;
            contrast_avg[n_contrast_bands] += mean_val.max(1e-10).log10();
        }
        for v in contrast_avg.iter_mut() { *v /= n_frames as Float; }
        Some(contrast_avg)
    } else {
        None
    };

    // ================================================================
    // Aggregate results
    // ================================================================

    let rms_mean = rms_frames.iter().sum::<Float>() / rms_frames.len() as Float;
    let rms_max = rms_frames.iter().copied().fold(0.0_f32, Float::max);

    let rms_nonzero: Vec<Float> = rms_frames.iter().copied().filter(|&v| v > 1e-10).collect();
    let dynamic_range_db = if rms_nonzero.len() > 10 {
        let mut sorted = rms_nonzero.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p5 = sorted[sorted.len() * 5 / 100];
        let p95 = sorted[sorted.len() * 95 / 100];
        if p5 > 0.0 { 20.0 * (p95 / p5).log10() } else { 0.0 }
    } else {
        0.0
    };

    let centroid_mean = centroids.iter().sum::<Float>() / centroids.len().max(1) as Float;
    let onset_density = onset_frames.len() as Float / duration_sec;

    let spectral_bandwidth_mean = if extended {
        Some(bandwidths.iter().sum::<Float>() / bandwidths.len().max(1) as Float)
    } else { None };

    let spectral_rolloff_mean = if extended {
        Some(rolloffs.iter().sum::<Float>() / rolloffs.len().max(1) as Float)
    } else { None };

    let spectral_flatness_mean = if extended {
        Some(flatnesses.iter().sum::<Float>() / flatnesses.len().max(1) as Float)
    } else { None };

    // ================================================================
    // PERCEPTUAL FEATURES (from already-computed scalars, ~0 extra cost)
    // ================================================================

    let bw_mean = spectral_bandwidth_mean.unwrap_or(0.0);
    let fl_mean = spectral_flatness_mean.unwrap_or(0.0);
    let ro_mean = spectral_rolloff_mean.unwrap_or(0.0);

    let wants_energy = extended && config.wants("energy");
    let wants_dance = extended && config.wants("danceability");
    let wants_key = extended && config.wants("key");
    let wants_valence = extended && (config.wants("valence") || config.wants("key"));
    let wants_acoustic = extended && config.wants("acousticness");

    let energy = if wants_energy {
        Some(perceptual::energy(rms_mean, centroid_mean, onset_density, bw_mean))
    } else { None };

    let danceability = if wants_dance && accurate {
        Some(perceptual::danceability_dfa(y, sr))
    } else if wants_dance {
        Some(perceptual::danceability_heuristic(bpm, &beats, onset_density))
    } else { None };

    // Key detection requires chroma (resolved as dependency)
    let key_result = if wants_key || wants_valence {
        chroma_mean.as_ref().map(|c| perceptual::detect_key(c))
    } else { None };

    let valence = if config.wants("valence") {
        key_result.as_ref().map(|kr| perceptual::valence(kr, bpm, centroid_mean))
    } else { None };

    let acousticness = if wants_acoustic {
        Some(perceptual::acousticness(fl_mean, ro_mean, onset_density))
    } else { None };

    let key = key_result.as_ref().map(|kr| perceptual::format_key(kr));
    let key_confidence = key_result.as_ref().map(|kr| kr.confidence);

    Ok(TrackAnalysis {
        duration_sec,
        bpm,
        beats,
        onset_frames,
        rms_mean,
        rms_max,
        loudness_lufs,
        dynamic_range_db,
        spectral_centroid_mean: centroid_mean,
        zero_crossing_rate: zcr,
        onset_density,
        spectral_bandwidth_mean,
        spectral_rolloff_mean,
        spectral_flatness_mean,
        spectral_contrast_mean,
        mfcc_mean,
        chroma_mean,
        energy,
        danceability,
        key,
        key_confidence,
        valence,
        acousticness,
        // Tier 3 placeholders — requires ML models
        mood_happy: None,
        mood_aggressive: None,
        mood_relaxed: None,
        mood_sad: None,
        instrumentalness: None,
        genre: None,
    })
}

// ============================================================
// Convenience constructors
// ============================================================

/// Shorthand for compact mode analysis.
pub fn compact() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Compact, accurate: false, features: None }
}

/// Shorthand for playlist mode analysis.
pub fn playlist() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Playlist, accurate: false, features: None }
}

/// Shorthand for playlist mode with accurate algorithms.
pub fn playlist_accurate() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Playlist, accurate: true, features: None }
}

/// Shorthand for full mode analysis.
pub fn full() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Full, accurate: false, features: None }
}

/// Shorthand for full mode with accurate algorithms.
pub fn full_accurate() -> AnalysisConfig {
    AnalysisConfig { mode: AnalysisMode::Full, accurate: true, features: None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq: Float, sr: u32, dur: Float) -> Array1<Float> {
        let n = (sr as Float * dur) as usize;
        Array1::from_shape_fn(n, |i| (2.0 * PI * freq * i as Float / sr as Float).sin())
    }

    #[test]
    fn test_analyze_compact() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert!(result.duration_sec > 1.9 && result.duration_sec < 2.1);
        assert!(result.bpm > 30.0 && result.bpm < 320.0);
        assert!(result.rms_mean > 0.0);
        assert!(result.spectral_centroid_mean > 0.0);
        // Compact: no extended features
        assert!(result.spectral_bandwidth_mean.is_none());
        assert!(result.mfcc_mean.is_none());
        assert!(result.energy.is_none());
    }

    #[test]
    fn test_analyze_playlist() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &playlist()).unwrap();
        assert!(result.spectral_bandwidth_mean.unwrap() > 0.0);
        assert!(result.mfcc_mean.unwrap().len() == 13);
        assert!(result.chroma_mean.unwrap().len() == 12);
        assert!(result.energy.unwrap() >= 0.0);
        assert!(result.danceability.unwrap() >= 0.0);
        assert!(result.key.is_some());
        assert!(result.valence.unwrap() >= 0.0);
        assert!(result.acousticness.unwrap() >= 0.0);
    }

    #[test]
    fn test_analyze_accurate_chroma() {
        let y = sine(440.0, 22050, 2.0);
        let result = analyze_signal(y.view(), 22050, &playlist_accurate()).unwrap();
        // Accurate chroma should map A440 to bin 9
        let chroma = result.chroma_mean.unwrap();
        assert_eq!(chroma.len(), 12);
        let max_bin = chroma.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;
        assert_eq!(max_bin, 9, "A440 should map to chroma bin 9 (A), got {}", max_bin);
    }

    #[test]
    fn test_analyze_custom_features() {
        let y = sine(440.0, 22050, 2.0);
        let config = AnalysisConfig {
            mode: AnalysisMode::Compact,
            accurate: false,
            features: Some(["energy", "key", "chroma"].iter().map(|s| s.to_string()).collect()),
        };
        let result = analyze_signal(y.view(), 22050, &config).unwrap();
        // Requested features should be present
        assert!(result.energy.is_some());
        assert!(result.key.is_some());
        assert!(result.chroma_mean.is_some());
        // Non-requested extended features should be absent
        assert!(result.danceability.is_none());
        assert!(result.acousticness.is_none());
    }

    #[test]
    fn test_analyze_click_train() {
        let sr = 22050u32;
        let n = (4.0 * sr as Float) as usize;
        let interval = (60.0 / 120.0 * sr as Float) as usize;
        let mut y = Array1::<Float>::zeros(n);
        let mut pos = 0;
        while pos < n {
            for i in 0..100.min(n - pos) {
                y[pos + i] = (2.0 * PI * 1000.0 * i as Float / sr as Float).sin();
            }
            pos += interval;
        }
        let result = analyze_signal(y.view(), sr, &compact()).unwrap();
        assert!(result.bpm > 50.0 && result.bpm < 250.0);
        assert!(result.onset_frames.len() >= 3);
    }

    #[test]
    fn test_analyze_features_reasonable() {
        let y = Array1::from_shape_fn(44100, |i| {
            (2.0 * PI * 440.0 * i as Float / 22050.0).sin() * 0.5
        });
        let result = analyze_signal(y.view(), 22050, &compact()).unwrap();
        assert!(result.rms_mean > 0.1 && result.rms_mean < 0.6,
            "RMS {} unexpected", result.rms_mean);
        assert!(result.spectral_centroid_mean > 300.0 && result.spectral_centroid_mean < 600.0,
            "Centroid {} unexpected", result.spectral_centroid_mean);
    }

    #[test]
    fn test_analyze_playlist_sine_vs_noise() {
        let sine_sig = sine(440.0, 22050, 2.0);
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let noise = Array1::from_shape_fn(44100, |i| {
            let mut h = DefaultHasher::new();
            (i as u64 ^ 0xDEADBEEF).hash(&mut h);
            (h.finish() as Float / u64::MAX as Float) * 2.0 - 1.0
        });

        let cfg = playlist();
        let r_sine = analyze_signal(sine_sig.view(), 22050, &cfg).unwrap();
        let r_noise = analyze_signal(noise.view(), 22050, &cfg).unwrap();

        assert!(r_sine.spectral_flatness_mean.unwrap() < r_noise.spectral_flatness_mean.unwrap());
        assert!(r_sine.spectral_bandwidth_mean.unwrap() < r_noise.spectral_bandwidth_mean.unwrap());
    }
}
