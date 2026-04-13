# Acknowledgments

## librosa

canora is a Rust reimplementation of [librosa](https://librosa.org/), the Python library for music and audio analysis.

canora's API design, default parameter values, and algorithm choices are directly inspired by librosa. While all code in canora is original (written from scratch in Rust), the following aspects are derived from librosa's design:

- Function names and signatures (e.g., `stft`, `melspectrogram`, `mfcc`, `beat_track`)
- Default parameter values (e.g., `n_fft=2048`, `hop_length=512`, `n_mels=128`)
- Algorithm implementations based on the same academic references used by librosa
- Test cases validated against librosa's output for numerical correctness

**librosa** is developed by Brian McFee and contributors, and is licensed under the ISC License:

> Copyright (c) 2013--2023, librosa development team.
>
> Permission to use, copy, modify, and/or distribute this software for any
> purpose with or without fee is hereby granted, provided that the above
> copyright notice and this permission notice appear in all copies.

See: https://github.com/librosa/librosa

## Academic References

The algorithms implemented in canora are based on published research, including:

- **STFT**: Allen, J.B. "Short term spectral analysis, synthesis, and modification by discrete Fourier transform." IEEE Transactions on Acoustics, Speech, and Signal Processing, 1977.
- **Mel scale**: Stevens, S.S., Volkmann, J., & Newman, E.B. "A scale for the measurement of the psychological magnitude pitch." JASA, 1937.
- **MFCC**: Davis, S., & Mermelstein, P. "Comparison of parametric representations for monosyllabic word recognition." IEEE Transactions on ASSP, 1980.
- **YIN**: De Cheveigné, A., & Kawahara, H. "YIN, a fundamental frequency estimator for speech and music." JASA, 2002.
- **pYIN**: Mauch, M., & Dixon, S. "pYIN: A fundamental frequency estimator using probabilistic threshold distributions." ICASSP, 2014.
- **Beat tracking**: Ellis, D.P.W. "Beat tracking by dynamic programming." JNMR, 2007.
- **CQT**: Brown, J.C. "Calculation of a constant Q spectral transform." JASA, 1991.
- **HPSS**: Fitzgerald, D. "Harmonic/percussive separation using median filtering." DAFx, 2010.
- **PCEN**: Wang, Y., et al. "Trainable frontend for robust and far-field keyword spotting." ICASSP, 2017.

## Rust Dependencies

canora is built on excellent Rust crates:

- [rustfft](https://github.com/ejmahler/RustFFT) / [realfft](https://github.com/HEnquist/realfft) — FFT computation
- [ndarray](https://github.com/rust-ndarray/ndarray) — N-dimensional arrays
- [symphonia](https://github.com/pdeljanov/Symphonia) — Audio decoding
- [rubato](https://github.com/HEnquist/rubato) — Sample rate conversion
- [rayon](https://github.com/rayon-rs/rayon) — Data parallelism
- [pyo3](https://github.com/PyO3/pyo3) / [rust-numpy](https://github.com/PyO3/rust-numpy) — Python bindings
