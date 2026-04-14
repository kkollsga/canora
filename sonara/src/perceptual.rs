//! Perceptual audio features: LUFS loudness, energy, danceability, key, valence, acousticness.
//!
//! These are higher-level features derived from the signal-level measurements
//! already computed by the fused analysis pipeline. No additional FFT work.
//!
//! Tier 0 (standardized): LUFS integrated loudness (ITU-R BS.1770-4)
//! Tier 1 (signal-grounded): energy, danceability, key detection
//! Tier 2 (heuristic approximations): valence, acousticness
//! Tier 3 (requires ML, future): mood_*, instrumentalness, genre

use ndarray::ArrayView1;

use crate::types::Float;

// ============================================================
// LUFS Integrated Loudness (ITU-R BS.1770-4 / EBU R128)
// ============================================================

/// K-weighting filter coefficients for a given sample rate.
///
/// Two cascaded biquad sections:
/// 1. High-shelf (~+4dB above 1500 Hz) — head-related transfer function
/// 2. High-pass at ~38 Hz — removes subsonic content
///
/// Coefficients are computed via bilinear transform from the analog prototypes
/// defined in ITU-R BS.1770-4.
struct KWeightCoeffs {
    // Stage 1: high-shelf (b0, b1, b2, a1, a2) — a0 normalized to 1.0
    s1_b: [Float; 3],
    s1_a: [Float; 2], // a1, a2 (a0 = 1.0)
    // Stage 2: high-pass
    s2_b: [Float; 3],
    s2_a: [Float; 2],
}

impl KWeightCoeffs {
    fn for_sample_rate(sr: u32) -> Self {
        let sr = sr as f64;

        // Stage 1: High shelf filter
        // Pre-warped analog prototype from BS.1770-4
        let f0: f64 = 1681.974450955533;
        let g: f64 = 3.999843853572914; // ~+4 dB
        let q: f64 = 0.7071752369554196;

        let k = (std::f64::consts::PI * f0 / sr).tan();
        let vh = (10.0_f64).powf(g / 20.0);
        let vb = vh.powf(0.4996667741545416);

        let a0 = 1.0 + k / q + k * k;
        let s1_b0 = ((vh + vb * k / q + k * k) / a0) as Float;
        let s1_b1 = (2.0 * (k * k - vh) / a0) as Float;
        let s1_b2 = ((vh - vb * k / q + k * k) / a0) as Float;
        let s1_a1 = (2.0 * (k * k - 1.0) / a0) as Float;
        let s1_a2 = ((1.0 - k / q + k * k) / a0) as Float;

        // Stage 2: High-pass filter at ~38 Hz
        let f0: f64 = 38.13547087602444;
        let q: f64 = 0.5003270373238773;

        let k = (std::f64::consts::PI * f0 / sr).tan();
        let a0 = 1.0 + k / q + k * k;
        let s2_b0 = (1.0 / a0) as Float;
        let s2_b1 = (-2.0 / a0) as Float;
        let s2_b2 = (1.0 / a0) as Float;
        let s2_a1 = (2.0 * (k * k - 1.0) / a0) as Float;
        let s2_a2 = ((1.0 - k / q + k * k) / a0) as Float;

        KWeightCoeffs {
            s1_b: [s1_b0, s1_b1, s1_b2],
            s1_a: [s1_a1, s1_a2],
            s2_b: [s2_b0, s2_b1, s2_b2],
            s2_a: [s2_a1, s2_a2],
        }
    }
}

/// Compute integrated LUFS loudness per ITU-R BS.1770-4 / EBU R128.
///
/// This is the industry standard for loudness measurement, used by Spotify,
/// YouTube, and broadcast. It applies K-weighting (models human loudness
/// perception) then computes mean-square energy.
///
/// Returns the integrated loudness in LUFS (typically -60 to 0 for music).
/// Silence returns -70.0 (the EBU R128 "absolute gate" threshold).
///
/// Performance: ~0.2-0.5ms for a 3-minute track (two biquad IIR passes).
pub fn loudness_lufs(y: ArrayView1<Float>, sr: u32) -> Float {
    let n = y.len();
    if n == 0 {
        return -70.0;
    }

    let c = KWeightCoeffs::for_sample_rate(sr);
    let raw = y.as_slice().unwrap();

    // Apply K-weighting: two cascaded biquad sections (Direct Form II Transposed)
    // We process both stages in a single pass to stay cache-friendly.
    let mut s1_z1: Float = 0.0;
    let mut s1_z2: Float = 0.0;
    let mut s2_z1: Float = 0.0;
    let mut s2_z2: Float = 0.0;
    let mut sum_sq: Float = 0.0;

    for i in 0..n {
        // Stage 1: high shelf
        let x = raw[i];
        let y1 = c.s1_b[0] * x + s1_z1;
        s1_z1 = c.s1_b[1] * x - c.s1_a[0] * y1 + s1_z2;
        s1_z2 = c.s1_b[2] * x - c.s1_a[1] * y1;

        // Stage 2: high pass
        let y2 = c.s2_b[0] * y1 + s2_z1;
        s2_z1 = c.s2_b[1] * y1 - c.s2_a[0] * y2 + s2_z2;
        s2_z2 = c.s2_b[2] * y1 - c.s2_a[1] * y2;

        sum_sq += y2 * y2;
    }

    let mean_sq = sum_sq / n as Float;

    if mean_sq < 1e-20 {
        -70.0 // EBU R128 absolute gate
    } else {
        -0.691 + 10.0 * mean_sq.log10()
    }
}

// ============================================================
// Key detection types
// ============================================================

/// Musical key detection result.
pub struct KeyResult {
    /// Root note: "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    pub key: &'static str,
    /// "major" or "minor"
    pub mode: &'static str,
    /// Pearson correlation strength (0.0 - 1.0). Higher = more confident.
    pub confidence: Float,
}

const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

// Temperley MIREX 2005 key profiles — corpus-derived, better for popular music
// than the original Krumhansl profiles. Essentia also recommends corpus-derived
// profiles (edma/bgate) over Krumhansl for non-classical music.
// Source: essentia key.cpp, D. Temperley "What's Key for Key?" (1999/2005)
const KEY_PROFILE_MAJOR: [Float; 12] = [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400];
const KEY_PROFILE_MINOR: [Float; 12] = [0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330];

// ============================================================
// Tier 1: Energy
// ============================================================

/// Compute perceptual energy (0.0 - 1.0).
///
/// Normalized combination of loudness, spectral brightness, rhythmic activity,
/// and frequency spread. Loosely modeled on Spotify's energy descriptor.
///
/// - Loud, bright, rhythmically active music → high energy
/// - Quiet, dark, sparse music → low energy
pub fn energy(
    rms_mean: Float,
    spectral_centroid_mean: Float,
    onset_density: Float,
    spectral_bandwidth_mean: Float,
) -> Float {
    // Normalize each feature to [0, 1] via empirical music ranges
    let norm_rms = (rms_mean / 0.5).clamp(0.0, 1.0);              // compressed pop reaches 0.5+
    let norm_centroid = ((spectral_centroid_mean - 500.0) / 4500.0).clamp(0.0, 1.0);
    let norm_onset = (onset_density / 10.0).clamp(0.0, 1.0);      // complex rhythms exceed 8
    let norm_bw = ((spectral_bandwidth_mean - 500.0) / 3500.0).clamp(0.0, 1.0);

    // Weighted combination
    let weighted = 0.35 * norm_rms + 0.25 * norm_centroid + 0.25 * norm_onset + 0.15 * norm_bw;

    // Gentler sigmoid centered lower — reaches 0.9+ on energetic music
    1.0 / (1.0 + (-4.0 * (weighted - 0.45)).exp())
}

// ============================================================
// Tier 1: Danceability
// ============================================================

/// Fast danceability estimate from beat regularity and tempo (0.0 - 1.0).
///
/// Uses heuristics: regular beats + tempo in 100-140 BPM range + moderate
/// onset density → high danceability. No extra signal processing needed.
pub fn danceability_heuristic(bpm: Float, beats: &[usize], onset_density: Float) -> Float {
    // Beat regularity: coefficient of variation of inter-beat intervals
    let beat_reg = if beats.len() >= 3 {
        let intervals: Vec<Float> = beats.windows(2)
            .map(|w| (w[1] - w[0]) as Float)
            .collect();
        let mean_interval = intervals.iter().sum::<Float>() / intervals.len() as Float;
        if mean_interval > 0.0 {
            let std_interval = (intervals.iter()
                .map(|&i| (i - mean_interval).powi(2))
                .sum::<Float>() / intervals.len() as Float)
                .sqrt();
            let cv = std_interval / mean_interval;
            1.0 - cv.clamp(0.0, 1.0) // low CV = regular beats = danceable
        } else {
            0.0
        }
    } else {
        0.0 // too few beats to judge
    };

    // Tempo sweet spot: Gaussian centered at 120 BPM
    let tempo_score = (-0.5 * ((bpm - 120.0) / 30.0).powi(2)).exp();

    // Onset density sweet spot: 2-6 onsets/sec is danceable
    let onset_score = (-0.5 * ((onset_density - 4.0) / 2.0).powi(2)).exp();

    0.4 * beat_reg + 0.35 * tempo_score + 0.25 * onset_score
}

/// Accurate danceability via Detrended Fluctuation Analysis (DFA).
///
/// Based on Streich & Herrera 2005, same algorithm as essentia's Danceability.
/// Operates on the raw audio signal. Returns 0.0 - 1.0 (normalized from raw DFA
/// values which typically range 0 to ~3).
pub fn danceability_dfa(y: ArrayView1<Float>, sr: u32) -> Float {
    let frame_size = (0.01 * sr as Float) as usize; // 10ms frames
    let n_samples = y.len();
    let n_frames = n_samples / frame_size;

    if n_frames < 10 {
        return 0.0;
    }

    let raw = y.as_slice().unwrap();

    // Step 1: Compute stddev per 10ms frame
    let mut s = vec![0.0_f32; n_frames];
    for i in 0..n_frames {
        let start = i * frame_size;
        let end = ((i + 1) * frame_size).min(n_samples);
        let n = (end - start) as Float;
        let mean: Float = raw[start..end].iter().sum::<Float>() / n;
        let var: Float = raw[start..end].iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / (n - 1.0).max(1.0);
        s[i] = var.sqrt();
    }

    // Step 2: Subtract mean and integrate
    let mean_s: Float = s.iter().sum::<Float>() / n_frames as Float;
    for v in s.iter_mut() {
        *v -= mean_s;
    }
    for i in 1..n_frames {
        s[i] += s[i - 1];
    }

    // Step 3: Compute DFA for each tau
    // Tau values: geometrically spaced from 30 (0.3s) to min(800, n_frames) (×1.1)
    let min_tau = 30usize;
    let max_tau = 800.min(n_frames);
    let mut taus = Vec::new();
    let mut tau = min_tau as Float;
    while (tau as usize) <= max_tau {
        taus.push(tau as usize);
        tau *= 1.1;
    }

    if taus.len() < 2 {
        return 0.0;
    }

    let mut f_values: Vec<Float> = Vec::with_capacity(taus.len());

    for &tau in &taus {
        if n_frames < tau {
            break;
        }

        let jump = (tau / 50).max(1);
        let mut total_error = 0.0_f32;
        let mut n_blocks = 0usize;

        let mut k = 0;
        while k + tau <= n_frames {
            total_error += residual_error(&s, k, k + tau);
            n_blocks += 1;
            k += jump;
        }

        if n_blocks > 0 {
            f_values.push((total_error / n_blocks as Float).sqrt());
        } else {
            f_values.push(0.0);
        }
    }

    // Step 4: Compute DFA exponent from log-log slope
    let n_f = f_values.len();
    if n_f < 2 {
        return 0.0;
    }

    let mut dfa_sum = 0.0_f32;
    let mut n_valid = 0usize;

    for i in 0..n_f - 1 {
        if f_values[i + 1] > 0.0 && f_values[i] > 0.0 {
            let dfa_i = (f_values[i + 1] / f_values[i]).log10()
                / ((taus[i + 1] as Float + 3.0) / (taus[i] as Float + 3.0)).log10();
            dfa_sum += dfa_i;
            n_valid += 1;
        }
    }

    if n_valid == 0 {
        return 0.0;
    }

    let dfa_exponent = dfa_sum / n_valid as Float;
    let raw_danceability = if dfa_exponent > 0.0 { 1.0 / dfa_exponent } else { 0.0 };

    // Normalize to 0-1 (typical range is 0 to ~3)
    (raw_danceability / 3.0).clamp(0.0, 1.0)
}

/// Least-squares residual error for a segment (used by DFA).
/// Formula from Mathworld: ssyy - ssxy^2 / ssxx
fn residual_error(array: &[Float], start: usize, end: usize) -> Float {
    let size = end - start;
    let mean_x = (size - 1) as Float * 0.5;
    let mean_y: Float = array[start..end].iter().sum::<Float>() / size as Float;

    let mut ssxx = 0.0_f32;
    let mut ssyy = 0.0_f32;
    let mut ssxy = 0.0_f32;

    for i in 0..size {
        let dx = i as Float - mean_x;
        let dy = array[start + i] - mean_y;
        ssxx += dx * dx;
        ssxy += dx * dy;
        ssyy += dy * dy;
    }

    if ssxx > 0.0 {
        (ssyy - ssxy * ssxy / ssxx) / size as Float
    } else {
        0.0
    }
}

// ============================================================
// Tier 1: Key detection
// ============================================================

/// Detect musical key from a chroma vector using the Krumhansl-Schmuckler algorithm.
///
/// The chroma vector should have 12 elements (C, C#, D, ..., B).
/// Returns the best-matching key, mode (major/minor), and correlation confidence.
pub fn detect_key(chroma: &[Float]) -> KeyResult {
    if chroma.len() != 12 {
        return KeyResult { key: "C", mode: "major", confidence: 0.0 };
    }

    let mut best_key = 0usize;
    let mut best_mode = "major";
    let mut best_corr: Float = -2.0;
    let mut second_best: Float = -2.0;

    for shift in 0..12 {
        let corr_major = pearson_correlation(chroma, &KEY_PROFILE_MAJOR, shift);
        if corr_major > best_corr {
            second_best = best_corr;
            best_corr = corr_major;
            best_key = shift;
            best_mode = "major";
        } else if corr_major > second_best {
            second_best = corr_major;
        }

        let corr_minor = pearson_correlation(chroma, &KEY_PROFILE_MINOR, shift);
        if corr_minor > best_corr {
            second_best = best_corr;
            best_corr = corr_minor;
            best_key = shift;
            best_mode = "minor";
        } else if corr_minor > second_best {
            second_best = corr_minor;
        }
    }

    // Confidence: how much the best key stands out from the second best
    let confidence = if best_corr > -1.0 {
        ((best_corr - second_best) / best_corr.abs().max(0.001)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    KeyResult {
        key: NOTE_NAMES[best_key],
        mode: best_mode,
        confidence,
    }
}

/// Format key result as a string like "C major" or "A minor".
pub fn format_key(result: &KeyResult) -> String {
    format!("{} {}", result.key, result.mode)
}

/// Pearson correlation between chroma (12 values) and a profile rotated by `shift`.
fn pearson_correlation(chroma: &[Float], profile: &[Float; 12], shift: usize) -> Float {
    let n = 12;
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;

    for i in 0..n {
        sum_x += chroma[(i + shift) % n];
        sum_y += profile[i];
    }
    let mean_x = sum_x / n as Float;
    let mean_y = sum_y / n as Float;

    let mut cov = 0.0_f32;
    let mut var_x = 0.0_f32;
    let mut var_y = 0.0_f32;

    for i in 0..n {
        let dx = chroma[(i + shift) % n] - mean_x;
        let dy = profile[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom > 0.0 { cov / denom } else { 0.0 }
}

// ============================================================
// Tier 2: Valence
// ============================================================

/// Heuristic valence/mood estimate (0.0 = sad/dark, 1.0 = happy/bright).
///
/// Combines musical mode (major=happy), tempo (fast=happy), and spectral
/// brightness (bright=happy). This is an approximation — true mood perception
/// is subjective and context-dependent.
pub fn valence(key_result: &KeyResult, bpm: Float, spectral_centroid_mean: Float) -> Float {
    // Mode contribution
    let mode_score = if key_result.confidence < 0.05 {
        0.5 // low confidence → neutral
    } else if key_result.mode == "major" {
        0.7
    } else {
        0.3
    };

    // Tempo contribution (faster = happier, range 60-180)
    let tempo_score = ((bpm - 60.0) / 120.0).clamp(0.0, 1.0);

    // Brightness contribution (brighter = happier)
    let brightness = ((spectral_centroid_mean - 1000.0) / 3000.0).clamp(0.0, 1.0);

    0.45 * mode_score + 0.30 * tempo_score + 0.25 * brightness
}

// ============================================================
// Tier 2: Acousticness
// ============================================================

/// Heuristic acousticness estimate (0.0 = electronic/synthetic, 1.0 = acoustic).
///
/// Acoustic music tends to be more tonal (low spectral flatness), have less
/// high-frequency energy (low rolloff), and fewer percussive onsets.
pub fn acousticness(
    spectral_flatness_mean: Float,
    spectral_rolloff_mean: Float,
    onset_density: Float,
) -> Float {
    // Low flatness = tonal = more acoustic
    let tonal_score = (1.0 - (spectral_flatness_mean * 5.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);

    // Low rolloff = less high-frequency energy = more acoustic
    let hf_score = (1.0 - ((spectral_rolloff_mean - 2000.0) / 6000.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);

    // Fewer onsets = calmer = more acoustic
    let calm_score = (1.0 - (onset_density / 6.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);

    0.45 * tonal_score + 0.30 * hf_score + 0.25 * calm_score
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_energy_loud_vs_quiet() {
        let loud = energy(0.35, 3000.0, 5.0, 2500.0);
        let quiet = energy(0.05, 800.0, 1.0, 600.0);
        assert!(loud > quiet, "loud energy {} should be > quiet energy {}", loud, quiet);
        assert!(loud > 0.5, "loud signal energy should be > 0.5, got {}", loud);
        assert!(quiet < 0.5, "quiet signal energy should be < 0.5, got {}", quiet);
    }

    #[test]
    fn test_energy_range() {
        // Extreme low
        let low = energy(0.0, 0.0, 0.0, 0.0);
        assert!(low >= 0.0 && low <= 1.0);
        // Extreme high
        let high = energy(1.0, 10000.0, 20.0, 8000.0);
        assert!(high >= 0.0 && high <= 1.0);
    }

    #[test]
    fn test_danceability_heuristic_regular_beats() {
        // Regular 120 BPM with consistent intervals
        let beats: Vec<usize> = (0..16).map(|i| i * 43).collect(); // ~43 frames apart
        let d = danceability_heuristic(120.0, &beats, 4.0);
        assert!(d > 0.5, "Regular 120 BPM should be danceable, got {}", d);
    }

    #[test]
    fn test_danceability_heuristic_irregular() {
        // Irregular beats, weird tempo
        let beats = vec![0, 10, 50, 52, 100, 200];
        let d = danceability_heuristic(45.0, &beats, 1.0);
        assert!(d < 0.4, "Irregular slow music should not be very danceable, got {}", d);
    }

    #[test]
    fn test_danceability_dfa_basic() {
        // Generate a simple signal and verify DFA returns a reasonable value
        let n = 22050 * 3; // 3 seconds
        let y = Array1::from_shape_fn(n, |i| {
            (2.0 * std::f32::consts::PI * 440.0 * i as Float / 22050.0).sin()
        });
        let d = danceability_dfa(y.view(), 22050);
        assert!(d >= 0.0 && d <= 1.0, "DFA danceability should be in [0,1], got {}", d);
    }

    #[test]
    fn test_key_detection_a440() {
        // A major chord: A(9), C#(1), E(4) → should detect A
        let mut chroma = [0.01_f32; 12];
        chroma[9] = 1.0; // A
        chroma[1] = 0.6; // C#
        chroma[4] = 0.5; // E
        let result = detect_key(&chroma);
        assert_eq!(result.key, "A", "A major chroma should detect key A, got {}", result.key);
    }

    #[test]
    fn test_key_detection_c_major() {
        // C, E, G prominent → C major
        let mut chroma = [0.01_f32; 12];
        chroma[0] = 1.0; // C
        chroma[4] = 0.7; // E
        chroma[7] = 0.5; // G
        let result = detect_key(&chroma);
        assert_eq!(result.key, "C", "C major chroma should detect C, got {}", result.key);
        assert_eq!(result.mode, "major", "C major chroma should detect major, got {}", result.mode);
    }

    #[test]
    fn test_key_detection_a_minor() {
        // A, C, E prominent → A minor
        let mut chroma = [0.01_f32; 12];
        chroma[9] = 1.0; // A
        chroma[0] = 0.7; // C
        chroma[4] = 0.5; // E
        let result = detect_key(&chroma);
        assert_eq!(result.key, "A", "A minor chroma should detect A, got {}", result.key);
    }

    #[test]
    fn test_valence_major_fast_bright() {
        let major_key = KeyResult { key: "C", mode: "major", confidence: 0.5 };
        let v = valence(&major_key, 140.0, 3500.0);
        assert!(v > 0.5, "Major + fast + bright should have high valence, got {}", v);
    }

    #[test]
    fn test_valence_minor_slow_dark() {
        let minor_key = KeyResult { key: "A", mode: "minor", confidence: 0.5 };
        let v = valence(&minor_key, 70.0, 800.0);
        assert!(v < 0.5, "Minor + slow + dark should have low valence, got {}", v);
    }

    #[test]
    fn test_acousticness_tonal_vs_noisy() {
        let acoustic = acousticness(0.01, 1500.0, 2.0); // tonal, low HF, calm
        let electronic = acousticness(0.5, 7000.0, 6.0); // flat, high HF, busy
        assert!(acoustic > electronic,
            "Tonal signal ({}) should be more acoustic than noisy signal ({})",
            acoustic, electronic);
    }

    #[test]
    fn test_acousticness_range() {
        let low = acousticness(1.0, 11025.0, 10.0);
        let high = acousticness(0.0, 500.0, 0.0);
        assert!(low >= 0.0 && low <= 1.0);
        assert!(high >= 0.0 && high <= 1.0);
    }

    #[test]
    fn test_format_key() {
        let result = KeyResult { key: "F#", mode: "minor", confidence: 0.7 };
        assert_eq!(format_key(&result), "F# minor");
    }

    // ---- LUFS tests ----

    #[test]
    fn test_lufs_silence() {
        let y = Array1::<Float>::zeros(22050);
        let lufs = loudness_lufs(y.view(), 22050);
        assert_eq!(lufs, -70.0, "Silence should be -70 LUFS");
    }

    #[test]
    fn test_lufs_loud_vs_quiet() {
        let loud = Array1::from_shape_fn(22050, |i| {
            (2.0 * std::f32::consts::PI * 1000.0 * i as Float / 22050.0).sin()
        });
        let quiet = loud.mapv(|v| v * 0.1);

        let lufs_loud = loudness_lufs(loud.view(), 22050);
        let lufs_quiet = loudness_lufs(quiet.view(), 22050);

        assert!(lufs_loud > lufs_quiet,
            "Loud LUFS ({}) should be > quiet LUFS ({})", lufs_loud, lufs_quiet);
        // 10x amplitude = 20 dB difference
        let diff = lufs_loud - lufs_quiet;
        assert!((diff - 20.0).abs() < 2.0,
            "LUFS difference {} should be ~20 dB for 10x amplitude ratio", diff);
    }

    #[test]
    fn test_lufs_range() {
        // Unit amplitude 1kHz sine should be around -3 LUFS
        // (K-weighting boosts HF slightly, so slightly higher than pure RMS)
        let y = Array1::from_shape_fn(44100, |i| {
            (2.0 * std::f32::consts::PI * 1000.0 * i as Float / 22050.0).sin()
        });
        let lufs = loudness_lufs(y.view(), 22050);
        assert!(lufs > -10.0 && lufs < 5.0,
            "Unit sine LUFS {} should be in reasonable range", lufs);
    }
}
