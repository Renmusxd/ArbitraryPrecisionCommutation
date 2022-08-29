use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rug::ops::CompleteRound;
use rug::Float;
use std::cmp::min;
use std::ops::{Index, IndexMut};

#[pyclass]
pub struct SparseBiDirectional {
    row_indexed: Vec<Vec<(usize, Float)>>,
    col_indexed: Vec<Vec<(usize, Float)>>,
    prec: u32,
    zero: Float,
}

#[pymethods]
impl SparseBiDirectional {
    #[new]
    fn new(matrix: Vec<(usize, usize, String)>, prec: u32, n: Option<usize>) -> PyResult<Self> {
        let max_row_col = n.unwrap_or_else(|| {
            matrix
                .iter()
                .flat_map(|(i, j, _)| [i, j].into_iter())
                .copied()
                .max()
                .map(|m| m + 1)
                .unwrap_or(1)
        });

        let hamiltonian = matrix
            .into_iter()
            .map(|(r, c, s)| (r, c, Float::parse(s)))
            .map(|(r, c, p)| (r, c, p.map(|p| Float::with_val(prec, p))));

        let mut hamiltonian_row_ordered = vec![vec![]; max_row_col];
        let mut hamiltonian_col_ordered = vec![vec![]; max_row_col];
        for (r, c, f) in hamiltonian {
            let f = f.map_err(|x| PyValueError::new_err(format!("Float error: {:?}", x)))?;
            hamiltonian_row_ordered[r].push((c, f.clone()));
            hamiltonian_col_ordered[c].push((r, f));
        }
        hamiltonian_row_ordered
            .iter_mut()
            .for_each(|v| v.sort_by_key(|(i, _)| *i));
        hamiltonian_col_ordered
            .iter_mut()
            .for_each(|v| v.sort_by_key(|(i, _)| *i));

        Ok(Self {
            row_indexed: hamiltonian_row_ordered,
            col_indexed: hamiltonian_col_ordered,
            prec,
            zero: Float::new(prec),
        })
    }

    fn __matmul__(&self, mat: &DenseMatrix) -> PyResult<DenseMatrix> {
        if self.col_indexed.len() != mat.n {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch {} v {}",
                self.col_indexed.len(),
                mat.n
            )));
        }

        let prec = min(self.prec, mat.prec);
        let newmat = DenseMatrix::new_from(mat.n, prec, |r, c| {
            let row = &self.row_indexed[r];
            Float::dot(
                row.iter()
                    .map(|(sparse_c, val)| (&mat[(*sparse_c, c)], val)),
            )
            .complete(prec)
        });
        Ok(newmat)
    }

    fn __rmatmul__(&self, mat: &DenseMatrix) -> PyResult<DenseMatrix> {
        if self.col_indexed.len() != mat.n {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch {} v {}",
                self.col_indexed.len(),
                mat.n
            )));
        }

        let prec = min(self.prec, mat.prec);
        let newmat = DenseMatrix::new_from(mat.n, min(self.prec, mat.prec), |r, c| {
            let col = &self.col_indexed[c];
            Float::dot(
                col.iter()
                    .map(|(sparse_r, val)| (&mat[(r, *sparse_r)], val)),
            )
            .complete(prec)
        });
        Ok(newmat)
    }

    fn __repr__(&self) -> String {
        format!(
            "[Sparse: {0}x{0} w/ prec={1}]",
            self.row_indexed.len(),
            self.prec
        )
    }
}

impl Index<(usize, usize)> for SparseBiDirectional {
    type Output = Float;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.row_indexed[index.0]
            .iter()
            .find(|(c, _)| index.1.eq(c))
            .map(|(_, v)| v)
            .unwrap_or(&self.zero)
    }
}

#[pyclass]
#[derive(Clone)]
struct FloatEntry {
    f: Float,
}

#[pymethods]
impl FloatEntry {
    #[new]
    fn new(data: String, prec: u32) -> PyResult<Self> {
        let f = Float::parse(data)
            .map_err(|err| PyValueError::new_err(format!("Float parse error {:?}", err)))?;
        Ok(Self {
            f: Float::with_val(prec, f),
        })
    }

    fn to_float(&self) -> f64 {
        self.f.to_f64()
    }

    fn __add__(&self, f: &FloatEntry) -> Self {
        Self {
            f: self.f.clone() + &f.f,
        }
    }

    fn __sub__(&self, f: &FloatEntry) -> Self {
        Self {
            f: self.f.clone() - &f.f,
        }
    }

    fn __mul__(&self, f: &FloatEntry) -> Self {
        Self {
            f: self.f.clone() * &f.f,
        }
    }
    fn __truediv__(&self, f: &FloatEntry) -> Self {
        Self {
            f: self.f.clone() / &f.f,
        }
    }

    fn __abs__(&self) -> Self {
        Self {
            f: self.f.clone().abs(),
        }
    }

    fn sqrt(&self) -> Self {
        Self {
            f: self.f.clone().sqrt(),
        }
    }

    fn __repr__(&self) -> String {
        format!("[{:?}]", self.f)
    }
}

#[pyclass]
struct DenseMatrix {
    n: usize,
    prec: u32,
    data: Vec<Float>,
}

#[pymethods]
impl DenseMatrix {
    #[new]
    fn new(n: usize, prec: u32, matrix: Option<Vec<String>>) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err(
                "N must be greater than 0".to_string(),
            ));
        }
        if let Some(matrix) = matrix {
            let data = matrix
                .into_iter()
                .map(|s| Float::parse(s))
                .map(|p| p.map(|p| Float::with_val(prec, p)))
                .try_fold(vec![], |mut acc, p| {
                    p.map(|p| {
                        acc.push(p);
                        acc
                    })
                })
                .map_err(|err| PyValueError::new_err(format!("Float parse error: {:?}", err)))?;

            Ok(Self { n, prec, data })
        } else {
            let data = vec![Float::new(prec); n * n];
            Ok(Self { n, prec, data })
        }
    }

    #[staticmethod]
    fn new_empty(n: usize, prec: u32) -> PyResult<Self> {
        Self::new(n, prec, None)
    }

    fn set_val(&mut self, r: usize, c: usize, data: String) -> PyResult<()> {
        self[(r, c)] = Float::with_val(
            self.prec,
            Float::parse(data)
                .map_err(|err| PyValueError::new_err(format!("Error parsing float: {:?}", err)))?,
        );
        Ok(())
    }

    #[staticmethod]
    fn new_from_numpy(data: PyReadonlyArray2<f64>, prec: u32) -> PyResult<Self> {
        let n = data.shape()[0];
        if n != data.shape()[1] {
            return Err(PyValueError::new_err(format!(
                "Matrix not square: {:?}",
                data.shape()
            )));
        }
        let data = data
            .as_slice()?
            .iter()
            .copied()
            .map(|f| Float::with_val(prec, f))
            .collect::<Vec<_>>();
        Ok(Self { n, data, prec })
    }

    fn trace(&self) -> FloatEntry {
        FloatEntry {
            f: Float::sum((0..self.n).map(|i| &self[(i, i)])).complete(self.prec),
        }
    }

    /// Performs Sqrt(Sum Aij^2)
    fn norm2(&self) -> FloatEntry {
        let data = self
            .data
            .iter()
            .map(|f| f.clone().square())
            .collect::<Vec<_>>();
        let f = Float::sum(data.iter()).complete(self.prec).sqrt();
        FloatEntry { f }
    }

    fn transpose(&self) -> Self {
        Self::new_from(self.n, self.prec, |r, c| self[(c, r)].clone())
    }

    fn to_numpy(&self, py: Python) -> Py<PyArray2<f64>> {
        let mut res = Array2::zeros((self.n, self.n));
        ndarray::Zip::indexed(&mut res).for_each(|(r, c), v| *v = self[(r, c)].to_f64());
        res.to_pyarray(py).to_owned()
    }

    fn __mul__(&self, other: &FloatEntry) -> Self {
        let prec = min(self.prec, other.f.prec());
        Self::new_from(self.n, prec, |r, c| {
            (&self[(r, c)] * &other.f).complete(prec)
        })
    }
    fn __rmul__(&self, other: &FloatEntry) -> Self {
        self.__mul__(other)
    }

    fn __sub__(&self, other: &Self) -> Self {
        let prec = min(self.prec, other.prec);
        Self::new_from(self.n, prec, |r, c| {
            (&self[(r, c)] - &other[(r, c)]).complete(prec)
        })
    }

    fn __truediv__(&self, val: &FloatEntry) -> Self {
        self.div_float(&val.f)
    }

    fn __repr__(&self) -> String {
        format!("[Dense: {0}x{0} w/ prec={1}]", self.n, self.prec)
    }
}

impl DenseMatrix {
    fn new_from<F>(n: usize, prec: u32, f: F) -> Self
    where
        F: Send + Sync + Fn(usize, usize) -> Float,
    {
        let mut v = Vec::new();
        v.resize(n * n, Float::new(prec));
        v.par_iter_mut().enumerate().for_each(|(i, v)| {
            let r = i % n;
            let c = i / n;
            *v = f(r, c);
        });
        Self { n, prec, data: v }
    }

    fn div_float(&self, f: &Float) -> Self {
        let prec = min(self.prec, f.prec());
        Self::new_from(self.n, prec, |r, c| (&self[(r, c)] / f).complete(prec))
    }
}

impl Index<(usize, usize)> for DenseMatrix {
    type Output = Float;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 + index.1 * self.n]
    }
}

impl IndexMut<(usize, usize)> for DenseMatrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 + index.1 * self.n]
    }
}

#[pymodule]
fn py_agp(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<SparseBiDirectional>()?;
    m.add_class::<DenseMatrix>()?;
    m.add_class::<FloatEntry>()?;
    Ok(())
}
