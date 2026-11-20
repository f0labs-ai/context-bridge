//! Passage vectors: rows of little-endian `f32` beside a JSON list of their ids.
//!
//! `vectors.f32` holds `ids.len() * dim` values, row after row; `vectors.json`
//! is `{"dim": …, "ids": […]}` with the ids in row order.

use crate::error::EvalError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The JSON half of a vector file pair.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorIndex {
    pub dim: usize,
    pub ids: Vec<String>,
}

/// A set of vectors of one length, with each row's length precomputed.
#[derive(Debug)]
pub struct Vectors {
    dim: usize,
    ids: Vec<String>,
    data: Vec<f32>,
    norms: Vec<f64>,
}

impl Vectors {
    /// Check the shape and the values. A vector of length zero has no direction and is refused.
    pub fn new(index: VectorIndex, data: Vec<f32>) -> Result<Self, EvalError> {
        let VectorIndex { dim, ids } = index;
        if dim == 0 {
            return Err(EvalError::invalid("vectors: dim is 0"));
        }
        if ids.len().checked_mul(dim) != Some(data.len()) {
            return Err(EvalError::invalid(format!(
                "vectors: {} ids of dim {dim} need {} values, the file holds {}",
                ids.len(),
                ids.len().saturating_mul(dim),
                data.len()
            )));
        }
        let mut norms = Vec::with_capacity(ids.len());
        for (id, row) in ids.iter().zip(data.chunks_exact(dim)) {
            if row.iter().any(|x| !x.is_finite()) {
                return Err(EvalError::invalid(format!(
                    "vectors: the vector of {id:?} holds a value that is not a number"
                )));
            }
            let norm = dot(row, row).sqrt();
            if norm == 0.0 {
                return Err(EvalError::invalid(format!(
                    "vectors: the vector of {id:?} is all zeros"
                )));
            }
            norms.push(norm);
        }
        Ok(Self {
            dim,
            ids,
            data,
            norms,
        })
    }

    /// Read the little-endian bytes of a `.f32` file.
    pub fn from_le_bytes(index: VectorIndex, bytes: &[u8]) -> Result<Self, EvalError> {
        let (values, rest) = bytes.as_chunks::<4>();
        if !rest.is_empty() {
            return Err(EvalError::invalid(format!(
                "vectors: {} bytes is not a whole number of f32 values",
                bytes.len()
            )));
        }
        Self::new(index, values.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    pub fn row(&self, row: usize) -> &[f32] {
        &self.data[row * self.dim..(row + 1) * self.dim]
    }

    /// `cos(a, b) = a·b / (|a| |b|)` of two rows.
    pub fn cosine(&self, a: usize, b: usize) -> f64 {
        dot(self.row(a), self.row(b)) / (self.norms[a] * self.norms[b])
    }

    /// Row of each id. A repeated id is a contract error.
    pub fn rows_by_id(&self) -> Result<HashMap<&str, usize>, EvalError> {
        let mut rows = HashMap::with_capacity(self.ids.len());
        for (row, id) in self.ids.iter().enumerate() {
            if rows.insert(id.as_str(), row).is_some() {
                return Err(EvalError::invalid(format!("vectors: id {id:?} appears twice")));
            }
        }
        Ok(rows)
    }
}

/// The dot product of two equal-length `f32` slices, summed in `f64`.
///
/// Eight running sums take every eighth product each and are added at the end
/// in a fixed order, so the result is the same on every machine and the loop
/// is not held back by one long chain of additions.
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    const LANES: usize = 8;
    let mut sums = [0.0f64; LANES];
    let (chunks_a, rest_a) = a.as_chunks::<LANES>();
    let (chunks_b, rest_b) = b.as_chunks::<LANES>();
    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        for lane in 0..LANES {
            sums[lane] += f64::from(ca[lane]) * f64::from(cb[lane]);
        }
    }
    let mut total = 0.0;
    for sum in sums {
        total += sum;
    }
    for (x, y) in rest_a.iter().zip(rest_b) {
        total += f64::from(*x) * f64::from(*y);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(dim: usize, ids: &[&str]) -> VectorIndex {
        VectorIndex {
            dim,
            ids: ids.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn the_dot_product_matches_a_sum_done_by_hand() {
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        // Ten terms cross the eight-lane boundary: sum of i*i for i in 1..=10 = 385.
        let v: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        assert_eq!(dot(&v, &v), 385.0);
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_is_one_for_the_same_direction_and_zero_at_right_angles() {
        let v = Vectors::new(
            index(2, &["x", "x2", "y", "d"]),
            vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 1.0, 1.0],
        )
        .unwrap();
        assert_eq!(v.cosine(0, 1), 1.0);
        assert_eq!(v.cosine(0, 2), 0.0);
        // (1,0)·(1,1) / (1 * sqrt 2) = 0.7071067811865475
        assert!((v.cosine(0, 3) - 0.5f64.sqrt()).abs() < 1e-15);
        assert_eq!(v.cosine(3, 0), v.cosine(0, 3));
    }

    #[test]
    fn bytes_are_read_little_endian() {
        let mut bytes = Vec::new();
        for x in [1.5f32, -2.0, 0.25, 8.0] {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let v = Vectors::from_le_bytes(index(2, &["a", "b"]), &bytes).unwrap();
        assert_eq!(v.row(0), [1.5, -2.0]);
        assert_eq!(v.row(1), [0.25, 8.0]);
        assert_eq!(v.rows_by_id().unwrap().get("b"), Some(&1));
    }

    #[test]
    fn a_wrong_shape_or_an_empty_vector_is_refused() {
        assert!(Vectors::new(index(2, &["a", "b"]), vec![1.0, 2.0, 3.0]).is_err());
        assert!(Vectors::new(index(0, &[]), vec![]).is_err());
        assert!(Vectors::new(index(2, &["a"]), vec![0.0, 0.0]).is_err());
        assert!(Vectors::new(index(2, &["a"]), vec![f32::NAN, 1.0]).is_err());
        assert!(Vectors::from_le_bytes(index(1, &["a"]), &[0, 0, 0]).is_err());
        let twice = Vectors::new(index(1, &["a", "a"]), vec![1.0, 1.0]).unwrap();
        assert!(twice.rows_by_id().is_err());
    }
}
