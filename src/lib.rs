use ndarray::Array2;
use numpy::{PyArray2, ToPyArray};
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
    fn new(matrix: Vec<(usize, usize, String)>, prec: u32) -> PyResult<Self> {
        let max_row_col = matrix
            .iter()
            .flat_map(|(i, j, _)| [i, j].into_iter())
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

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
            row.par_iter()
                .map(|(sparse_c, val)| &mat[(*sparse_c, c)] * val)
                .map(|m| m.complete(prec))
                .reduce(|| Float::new(prec), |a, b| a + b)
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
            col.par_iter()
                .map(|(sparse_r, val)| &mat[(r, *sparse_r)] * val)
                .map(|m| m.complete(prec))
                .reduce(|| Float::new(prec), |a, b| a + b)
        });
        Ok(newmat)
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

    fn __add__(&self, f: FloatEntry) -> Self {
        Self {
            f: self.f.clone() + f.f,
        }
    }

    fn __sub__(&self, f: FloatEntry) -> Self {
        Self {
            f: self.f.clone() - f.f,
        }
    }

    fn __mul__(&self, f: FloatEntry) -> Self {
        Self {
            f: self.f.clone() * f.f,
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
    fn new(n: usize, matrix: Vec<String>, prec: u32) -> PyResult<Self> {
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
    }

    fn trace(&self) -> FloatEntry {
        FloatEntry {
            f: (0..self.n)
                .into_par_iter()
                .map(|i| self[(i, i)].clone())
                .reduce(|| Float::new(self.prec), |a, b| a + b),
        }
    }

    /// Performs Sqrt(Sum Aij^2)
    fn norm2(&self) -> FloatEntry {
        let f = self
            .data
            .par_iter()
            .map(|f| f.clone().square())
            .reduce(|| Float::new(self.prec), |a, b| a + b)
            .sqrt();
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

    fn __mul__(&self, other: FloatEntry) -> Self {
        Self::new_from(self.n, min(self.prec, other.f.prec()), |r, c| {
            self[(r, c)].clone() * other.f.clone()
        })
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self::new_from(self.n, min(self.prec, other.prec), |r, c| {
            self[(r, c)].clone() - other[(r, c)].clone()
        })
    }

    fn __truediv__(&self, val: FloatEntry) -> Self {
        self.div_float(&val.f)
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
        Self::new_from(self.n, min(self.prec, f.prec()), |r, c| {
            &self[(r, c)] / f.clone()
        })
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
