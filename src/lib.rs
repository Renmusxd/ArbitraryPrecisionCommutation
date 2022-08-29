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
pub struct Matrix {
    data: MatrixType,
}

enum MatrixType {
    Sparse(SparseBiDirectional),
    Dense(DenseMatrix),
}

#[pymethods]
impl Matrix {
    #[staticmethod]
    fn new_dense(n: usize, prec: u32, matrix: Option<Vec<String>>) -> PyResult<Self> {
        let data = matrix
            .map(|matrix| DenseMatrix::new(n, matrix, prec))
            .unwrap_or_else(|| Ok(DenseMatrix::new_empty(n, prec)))?;

        Ok(Matrix {
            data: MatrixType::Dense(data),
        })
    }

    #[staticmethod]
    fn new_sparse(
        n: usize,
        prec: u32,
        matrix: Option<Vec<(usize, usize, String)>>,
    ) -> PyResult<Self> {
        let data = matrix
            .map(|matrix| SparseBiDirectional::new(n, matrix, prec))
            .unwrap_or_else(|| Ok(SparseBiDirectional::new_empty(n, prec)))?;

        Ok(Matrix {
            data: MatrixType::Sparse(data),
        })
    }

    fn set_value(&mut self, r: usize, c: usize, val: String) -> PyResult<()> {
        match &mut self.data {
            MatrixType::Sparse(sparse) => sparse.set_val(r, c, val),
            MatrixType::Dense(data) => data.set_val(r, c, val),
        }
    }

    fn n(&self) -> usize {
        match &self.data {
            MatrixType::Sparse(s) => s.row_indexed.len(),
            MatrixType::Dense(d) => d.n,
        }
    }
    fn prec(&self) -> u32 {
        match &self.data {
            MatrixType::Sparse(s) => s.prec,
            MatrixType::Dense(d) => d.prec,
        }
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        if self.n() != other.n() {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch {0}x{0} v {1}x{1}",
                self.n(),
                other.n()
            )));
        }
        let prec = min(self.prec(), other.prec());

        let res = match (&self.data, &other.data) {
            (MatrixType::Dense(da), MatrixType::Dense(db)) => Matrix {
                data: MatrixType::Dense(DenseMatrix::new_from(self.n(), prec, |r, c| {
                    (&da[(r, c)] + &db[(r, c)]).complete(prec)
                })),
            },
            (MatrixType::Sparse(s), MatrixType::Dense(d)) => {
                let mut d = DenseMatrix {
                    n: d.n,
                    prec,
                    data: d.data.clone(),
                };
                for (r, vs) in s.row_indexed.iter().enumerate() {
                    for (c, v) in vs.iter() {
                        d[(r, *c)] = (&d[(r, *c)] + v).complete(prec);
                    }
                }
                Matrix {
                    data: MatrixType::Dense(d),
                }
            }
            (MatrixType::Dense(_), MatrixType::Sparse(_)) => other.__add__(self)?,
            (MatrixType::Sparse(sa), MatrixType::Sparse(sb)) => {
                let s = sa.add(sb).map_err(|s| PyValueError::new_err(s))?;
                Matrix {
                    data: MatrixType::Sparse(s),
                }
            }
        };
        Ok(res)
    }

    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        if self.n() != other.n() {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch {0}x{0} v {1}x{1}",
                self.n(),
                other.n()
            )));
        }
        let prec = min(self.prec(), other.prec());

        let res = match (&self.data, &other.data) {
            (MatrixType::Dense(da), MatrixType::Dense(db)) => {
                let prec = min(da.prec, db.prec);
                Matrix {
                    data: MatrixType::Dense(DenseMatrix::new_from(self.n(), prec, |r, c| {
                        (&da[(r, c)] - &db[(r, c)]).complete(prec)
                    })),
                }
            }
            (MatrixType::Sparse(s), MatrixType::Dense(d)) => {
                let mut d = DenseMatrix {
                    n: d.n,
                    prec,
                    data: d.data.clone(),
                };
                for (r, vs) in s.row_indexed.iter().enumerate() {
                    for (c, v) in vs.iter() {
                        d[(r, *c)] = (v - &d[(r, *c)]).complete(prec);
                    }
                }
                Matrix {
                    data: MatrixType::Dense(d),
                }
            }
            (MatrixType::Dense(d), MatrixType::Sparse(s)) => {
                let mut d = DenseMatrix {
                    n: d.n,
                    prec,
                    data: d.data.clone(),
                };
                for (r, vs) in s.row_indexed.iter().enumerate() {
                    for (c, v) in vs.iter() {
                        d[(r, *c)] = (&d[(r, *c)] - v).complete(prec);
                    }
                }
                Matrix {
                    data: MatrixType::Dense(d),
                }
            }
            (MatrixType::Sparse(sa), MatrixType::Sparse(sb)) => {
                let s = sa.sub(sb).map_err(|s| PyValueError::new_err(s))?;
                Matrix {
                    data: MatrixType::Sparse(s),
                }
            }
        };
        Ok(res)
    }

    fn __mul__(&self, f: &FloatEntry) -> Self {
        match &self.data {
            MatrixType::Dense(dense) => {
                let prec = min(dense.prec, f.f.prec());
                Matrix {
                    data: MatrixType::Dense(DenseMatrix::new_from(dense.n, prec, |r, c| {
                        (&dense[(r, c)] * &f.f).complete(prec)
                    })),
                }
            }
            MatrixType::Sparse(sparse) => Matrix {
                data: MatrixType::Sparse(sparse.mul(f)),
            },
        }
    }

    fn __truediv__(&self, f: &FloatEntry) -> Self {
        match &self.data {
            MatrixType::Dense(dense) => Matrix {
                data: MatrixType::Dense(dense.truediv(f)),
            },
            MatrixType::Sparse(sparse) => Matrix {
                data: MatrixType::Sparse(sparse.truediv(f)),
            },
        }
    }

    fn __rmul__(&self, f: &FloatEntry) -> Self {
        self.__mul__(f)
    }

    fn __matmul__(&self, mat: &Matrix) -> PyResult<Matrix> {
        let res = match (&self.data, &mat.data) {
            (MatrixType::Sparse(sparse), MatrixType::Dense(dense)) => Matrix {
                data: MatrixType::Dense(sparse.mult_dense(dense)?),
            },
            (MatrixType::Dense(dense), MatrixType::Sparse(sparse)) => Matrix {
                data: MatrixType::Dense(sparse.left_mult_dense(dense)?),
            },
            (MatrixType::Dense(a), MatrixType::Dense(b)) => Matrix {
                data: MatrixType::Dense(a.matmul(b).map_err(|s| PyValueError::new_err(s))?),
            },
            (MatrixType::Sparse(a), MatrixType::Sparse(b)) => Matrix {
                data: MatrixType::Sparse(a.matmul(b).map_err(|s| PyValueError::new_err(s))?),
            },
        };
        Ok(res)
    }
}

#[pyclass]
pub struct SparseBiDirectional {
    row_indexed: Vec<Vec<(usize, Float)>>,
    col_indexed: Vec<Vec<(usize, Float)>>,
    prec: u32,
    zero: Float,
}

impl SparseBiDirectional {
    fn add(&self, other: &Self) -> Result<Self, String> {
        if self.row_indexed.len() != other.row_indexed.len() {
            return Err(format!(
                "Size mismatch {0}x{0} v {1}x{1}",
                self.row_indexed.len(),
                other.row_indexed.len()
            ));
        }

        let prec = min(self.prec, other.prec);
        let row_indexed = Self::merge_operation(
            &self.row_indexed,
            &other.row_indexed,
            |a, b| Some((a + b).complete(prec)),
            |a| Some(a.clone()),
            |b| Some(b.clone()),
        );
        let col_indexed = Self::merge_operation(
            &self.col_indexed,
            &other.col_indexed,
            |a, b| Some((a + b).complete(prec)),
            |a| Some(a.clone()),
            |b| Some(b.clone()),
        );
        Ok(Self {
            row_indexed,
            col_indexed,
            prec,
            zero: Float::new(prec),
        })
    }

    fn sub(&self, other: &Self) -> Result<Self, String> {
        if self.row_indexed.len() != other.row_indexed.len() {
            return Err(format!(
                "Size mismatch {0}x{0} v {1}x{1}",
                self.row_indexed.len(),
                other.row_indexed.len()
            ));
        }

        let prec = min(self.prec, other.prec);
        let row_indexed = Self::merge_operation(
            &self.row_indexed,
            &other.row_indexed,
            |a, b| Some((a - b).complete(prec)),
            |a| Some(a.clone()),
            |b| Some(-b.clone()),
        );
        let col_indexed = Self::merge_operation(
            &self.col_indexed,
            &other.col_indexed,
            |a, b| Some((a - b).complete(prec)),
            |a| Some(a.clone()),
            |b| Some(-b.clone()),
        );
        Ok(Self {
            row_indexed,
            col_indexed,
            prec,
            zero: Float::new(prec),
        })
    }

    fn merge_operation<F, G, H>(
        own: &[Vec<(usize, Float)>],
        other: &[Vec<(usize, Float)>],
        f: F,
        g: G,
        h: H,
    ) -> Vec<Vec<(usize, Float)>>
    where
        F: Fn(&Float, &Float) -> Option<Float> + Send + Sync,
        G: Fn(&Float) -> Option<Float> + Send + Sync,
        H: Fn(&Float) -> Option<Float> + Send + Sync,
    {
        own.par_iter()
            .zip(other.par_iter())
            .map(|(ra, rb)| {
                let mut r = vec![];
                let mut ai = 0;
                let mut bi = 0;
                while (ai < ra.len()) && (bi < rb.len()) {
                    let (ca, va) = &ra[ai];
                    let (cb, vb) = &rb[bi];
                    if ca == cb {
                        ai += 1;
                        bi += 1;
                        if let Some(f) = f(va, vb) {
                            r.push((*ca, f));
                        }
                    } else if ca < cb {
                        ai += 1;
                        if let Some(f) = g(va) {
                            r.push((*ca, f));
                        }
                    } else {
                        bi += 1;
                        if let Some(f) = h(vb) {
                            r.push((*cb, f));
                        }
                    }
                }
                for (ca, va) in (ai..ra.len()).map(|ai| &ra[ai]) {
                    if let Some(f) = g(va) {
                        r.push((*ca, f));
                    }
                }
                for (cb, vb) in (bi..rb.len()).map(|bi| &rb[bi]) {
                    if let Some(f) = h(vb) {
                        r.push((*cb, f));
                    }
                }
                r
            })
            .collect()
    }

    fn matmul(&self, other: &Self) -> Result<Self, String> {
        if self.col_indexed.len() != other.row_indexed.len() {
            return Err(format!(
                "Size mismatch {}x{} @ {}x{}",
                self.row_indexed.len(),
                self.col_indexed.len(),
                other.row_indexed.len(),
                other.col_indexed.len()
            ));
        }
        let prec = min(self.prec, other.prec);

        let mut row_indexed = vec![vec![]; self.row_indexed.len()];
        row_indexed.par_iter_mut().enumerate().for_each(|(r, row)| {
            (0..self.col_indexed.len()).for_each(|c| {
                // Check if (r,c) is nonzero
                let a_row = &self.row_indexed[r];
                let b_col = &other.col_indexed[c];

                let mut ai = 0;
                let mut bi = 0;
                let mut app = false;
                let mut net = Float::new(prec);
                while (ai < a_row.len()) && (bi < b_col.len()) {
                    let (ca, va) = &a_row[ai];
                    let (cb, vb) = &b_col[bi];
                    if ca == cb {
                        ai += 1;
                        bi += 1;
                        net += va * vb;
                        app = true;
                    } else if ca < cb {
                        ai += 1;
                    } else {
                        bi += 1;
                    }
                }
                if app {
                    row.push((c, net));
                }
            })
        });

        let mut col_indexed = vec![vec![]; other.col_indexed.len()];
        row_indexed.iter().enumerate().for_each(|(r, v)| {
            for (c, v) in v {
                col_indexed[*c].push((r, v.clone()));
            }
        });
        col_indexed
            .par_iter_mut()
            .for_each(|v| v.sort_by_key(|(i, _)| *i));

        Ok(Self {
            row_indexed,
            col_indexed,
            prec,
            zero: Float::new(prec),
        })
    }

    fn mul(&self, f: &FloatEntry) -> Self {
        let prec = min(self.prec, f.f.prec());
        Self {
            row_indexed: self
                .row_indexed
                .par_iter()
                .map(|v| {
                    v.iter()
                        .map(|(i, v)| (*i, (v * &f.f).complete(prec)))
                        .collect()
                })
                .collect(),
            col_indexed: self
                .col_indexed
                .par_iter()
                .map(|v| {
                    v.iter()
                        .map(|(i, v)| (*i, (v * &f.f).complete(prec)))
                        .collect()
                })
                .collect(),
            prec,
            zero: Float::new(prec),
        }
    }

    fn truediv(&self, f: &FloatEntry) -> Self {
        let prec = min(self.prec, f.f.prec());
        Self {
            row_indexed: self
                .row_indexed
                .par_iter()
                .map(|v| {
                    v.iter()
                        .map(|(i, v)| (*i, (v / &f.f).complete(prec)))
                        .collect()
                })
                .collect(),
            col_indexed: self
                .col_indexed
                .par_iter()
                .map(|v| {
                    v.iter()
                        .map(|(i, v)| (*i, (v / &f.f).complete(prec)))
                        .collect()
                })
                .collect(),
            prec,
            zero: Float::new(prec),
        }
    }

    fn set_val(&mut self, r: usize, c: usize, val: String) -> PyResult<()> {
        let f = Float::with_val(
            self.prec,
            Float::parse(val)
                .map_err(|s| PyValueError::new_err(format!("Parsing error {:?}", s)))?,
        );
        match self.row_indexed[r].binary_search_by_key(&c, |x| x.0) {
            Ok(i) => self.row_indexed[r][i].1 = f.clone(),
            Err(i) => self.row_indexed[r].insert(i, (c, f.clone())),
        }
        match self.col_indexed[r].binary_search_by_key(&r, |x| x.0) {
            Ok(i) => self.col_indexed[r][i].1 = f,
            Err(i) => self.col_indexed[r].insert(i, (r, f)),
        }
        Ok(())
    }
}

#[pymethods]
impl SparseBiDirectional {
    #[new]
    fn new(n: usize, matrix: Vec<(usize, usize, String)>, prec: u32) -> PyResult<Self> {
        let hamiltonian = matrix
            .into_iter()
            .map(|(r, c, s)| (r, c, Float::parse(s)))
            .map(|(r, c, p)| (r, c, p.map(|p| Float::with_val(prec, p))));

        let mut hamiltonian_row_ordered = vec![vec![]; n];
        let mut hamiltonian_col_ordered = vec![vec![]; n];
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

    #[staticmethod]
    fn new_empty(n: usize, prec: u32) -> Self {
        let row_indexed = vec![vec![]; n];
        let col_indexed = row_indexed.clone();
        Self {
            row_indexed,
            col_indexed,
            prec,
            zero: Float::new(prec),
        }
    }

    fn __matmul__(&self, mat: &DenseMatrix) -> PyResult<DenseMatrix> {
        self.mult_dense(mat)
    }

    fn __rmatmul__(&self, mat: &DenseMatrix) -> PyResult<DenseMatrix> {
        self.left_mult_dense(mat)
    }
}

impl SparseBiDirectional {
    fn mult_dense(&self, mat: &DenseMatrix) -> PyResult<DenseMatrix> {
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

    fn left_mult_dense(&self, mat: &DenseMatrix) -> PyResult<DenseMatrix> {
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

    fn __add__(&self, f: &FloatEntry) -> Self {
        Self {
            f: (&self.f + &f.f).complete(min(self.f.prec(), f.f.prec())),
        }
    }

    fn __sub__(&self, f: &FloatEntry) -> Self {
        Self {
            f: (&self.f - &f.f).complete(min(self.f.prec(), f.f.prec())),
        }
    }

    fn __mul__(&self, f: &FloatEntry) -> Self {
        Self {
            f: (&self.f * &f.f).complete(min(self.f.prec(), f.f.prec())),
        }
    }

    fn __truediv__(&self, f: &FloatEntry) -> Self {
        Self {
            f: (&self.f * &f.f).complete(min(self.f.prec(), f.f.prec())),
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

    #[staticmethod]
    fn new_empty(n: usize, prec: u32) -> Self {
        let data = vec![Float::new(prec); n * n];
        Self { n, prec, data }
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

    fn truediv(&self, f: &FloatEntry) -> Self {
        let prec = min(self.prec, f.f.prec());
        DenseMatrix::new_from(self.n, prec, |r, c| (&self[(r, c)] / &f.f).complete(prec))
    }

    fn div_float(&self, f: &Float) -> Self {
        let prec = min(self.prec, f.prec());
        Self::new_from(self.n, prec, |r, c| (&self[(r, c)] / f).complete(prec))
    }

    fn matmul(&self, mat: &DenseMatrix) -> Result<Self, String> {
        if self.n != mat.n {
            return Err(format!(
                "Dimension mismatch: {0}x{0} vs {1}x{1}",
                self.n, mat.n
            ));
        }
        let prec = min(self.prec, mat.prec);
        Ok(DenseMatrix::new_from(self.n, prec, |r, c| {
            (0..self.n)
                .map(|i| (&self[(r, i)] * &self[(i, c)]).complete(prec))
                .reduce(|a, b| a + b)
                .unwrap_or(Float::new(prec))
        }))
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
    m.add_class::<Matrix>()?;
    Ok(())
}
