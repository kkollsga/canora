//! Local extrema detection.
//!
//! Replaces scipy.signal.argrelmax and scipy.signal.argrelmin.

use ndarray::ArrayView1;

use crate::types::Float;

/// Find indices of local maxima.
///
/// A sample `x[i]` is a local maximum if it is strictly greater than
/// all samples within `order` positions on either side.
pub fn argrelmax(x: ArrayView1<Float>, order: usize) -> Vec<usize> {
    let n = x.len();
    let mut maxima = Vec::new();

    for i in order..n.saturating_sub(order) {
        let mut is_max = true;
        for j in 1..=order {
            if i < j || i + j >= n {
                is_max = false;
                break;
            }
            if x[i] <= x[i - j] || x[i] <= x[i + j] {
                is_max = false;
                break;
            }
        }
        if is_max {
            maxima.push(i);
        }
    }

    maxima
}

/// Find indices of local minima.
pub fn argrelmin(x: ArrayView1<Float>, order: usize) -> Vec<usize> {
    let n = x.len();
    let mut minima = Vec::new();

    for i in order..n.saturating_sub(order) {
        let mut is_min = true;
        for j in 1..=order {
            if i < j || i + j >= n {
                is_min = false;
                break;
            }
            if x[i] >= x[i - j] || x[i] >= x[i + j] {
                is_min = false;
                break;
            }
        }
        if is_min {
            minima.push(i);
        }
    }

    minima
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_argrelmax_basic() {
        let x = array![1.0, 3.0, 2.0, 4.0, 1.0];
        let maxima = argrelmax(x.view(), 1);
        assert_eq!(maxima, vec![1, 3]);
    }

    #[test]
    fn test_argrelmin_basic() {
        let x = array![3.0, 1.0, 2.0, 0.0, 4.0];
        let minima = argrelmin(x.view(), 1);
        assert_eq!(minima, vec![1, 3]);
    }

    #[test]
    fn test_argrelmax_order2() {
        // With order=2, need to be greater than 2 neighbors on each side
        let x = array![1.0, 2.0, 5.0, 2.0, 1.0, 3.0, 1.0];
        let maxima = argrelmax(x.view(), 2);
        assert_eq!(maxima, vec![2]); // index 5 is not included: not > x[3]=2 with order=2
    }

    #[test]
    fn test_argrelmax_plateau() {
        // Plateaus are NOT local maxima (not strictly greater)
        let x = array![1.0, 3.0, 3.0, 1.0];
        let maxima = argrelmax(x.view(), 1);
        assert!(maxima.is_empty());
    }

    #[test]
    fn test_argrelmax_empty() {
        let x = array![1.0, 2.0];
        let maxima = argrelmax(x.view(), 1);
        assert!(maxima.is_empty());
    }

    #[test]
    fn test_argrelmax_sine() {
        let n = 1000;
        let x = ndarray::Array1::from_shape_fn(n, |i| {
            (i as f32 * 2.0 * std::f32::consts::PI / 100.0).sin()
        });
        let maxima = argrelmax(x.view(), 1);
        // Sine with period 100 has ~10 peaks in 1000 samples
        assert!(maxima.len() >= 8 && maxima.len() <= 11);
    }
}
