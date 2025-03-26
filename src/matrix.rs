use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use num::Float;

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    values: Vec<Vec<T>>,
}

impl<T> Matrix<T>
where
    T: Float,
{
    const EPS: f64 = 1e-5;

    /// # Panics
    /// Zero rows
    pub fn new(values: Vec<Vec<T>>) -> Self {
        let row_len = values.first().unwrap().len();

        for row in &values {
            assert_eq!(row.len(), row_len);
        }

        Self { values }
    }

    pub fn column(values: &[T]) -> Self {
        let n = values.len();
        let mut v: Vec<Vec<T>> = Vec::with_capacity(values.len());
        v.resize(n, Vec::with_capacity(1));

        for i in 0..n {
            v[i].push(values[i]);
        }

        Self { values: v }
    }

    pub fn row(values: Vec<T>) -> Self {
        Self {
            values: vec![values],
        }
    }

    pub fn inversed(&self) -> Self {
        self.assert_square();

        let n = self.row_count();

        let e = Matrix::identity(n);

        let (l, u) = self.get_lu();

        let mut inversed_rows = Vec::with_capacity(n);

        for current_e in e.iter() {
            let r =
                Matrix::solve_lu_with(&l, &u, &Matrix::new(vec![current_e.clone()]).transposed());
            let t = r.transposed();
            inversed_rows.push(t.values.into_iter().next().unwrap());
        }

        Matrix::new(inversed_rows).transposed()
    }

    pub fn determinant(&self) -> T {
        self.assert_square();
        let n = self.row_count();

        let (_, u) = self.get_lu();
        let mut d = T::one();

        for i in 0..n {
            d = d * u[i][i];
        }

        d
    }

    pub fn transposed(&self) -> Self {
        let c = self.column_count();
        let r = self.row_count();

        let mut t = Matrix::zero_matrix(self.column_count(), self.row_count());

        for i in 0..c {
            for j in 0..r {
                t[i][j] = self[j][i];
            }
        }

        t
    }

    pub fn solve_tridiagonal(&self, d: &Self) -> Self {
        self.assert_square();
        assert_eq!(d.column_count(), 1);
        assert_eq!(d.row_count(), self.row_count());

        let n = d.row_count();
        let mut a = vec![T::zero(); n];
        let mut b = vec![T::zero(); n];
        let mut c = vec![T::zero(); n];

        for i in 0..n {
            b[i] = self[i][i];
            if i > 0 {
                a[i] = self[i][i - 1];
            }
            if i < n - 1 {
                c[i] = self[i][i + 1];
            }
        }

        let mut p = vec![T::zero(); n];
        let mut q = vec![T::zero(); n];
        let mut x = Matrix::new(vec![vec![T::zero(); n]]).transposed();

        p[0] = -c[0] / b[0];
        q[0] = d[0][0] / b[0];

        for i in 1..n - 1 {
            let denominator = b[i] + a[i] * p[i - 1];
            p[i] = -c[i] / denominator;
            q[i] = (d[i][0] - a[i] * q[i - 1]) / denominator;
        }

        x[n - 1][0] = (d[n - 1][0] - a[n - 1] * q[n - 2]) / (b[n - 1] + a[n - 1] * p[n - 2]);

        for i in (0..n - 1).rev() {
            x[i][0] = p[i] * x[i + 1][0] + q[i];
        }

        x
    }

    pub fn solve_lu(&self, b: &Self) -> Self {
        self.assert_square();
        assert_eq!(b.column_count(), 1);

        let (l, u) = self.get_lu();

        Matrix::solve_lu_with(&l, &u, b)
    }

    pub fn solve_lu_with(l: &Self, u: &Self, b: &Self) -> Self {
        l.assert_square();
        u.assert_square();

        assert_eq!(b.column_count(), 1);

        let z = Matrix::get_z(l, b);
        Matrix::get_x(u, &z)
    }

    /// # Panics
    /// Matrix is not square
    fn assert_square(&self) {
        assert_eq!(self.row_count(), self.column_count());
    }

    pub(crate) fn get_lu(&self) -> (Self, Self) {
        self.assert_square();

        let n = self.row_count();

        let mut u = self.clone();
        let mut l = Matrix::identity(self.row_count());

        // let mut swaps = Vec::with_capacity(self.row_count());

        for k in 0..n {
            // TODO: float optimization
            // let mut max_in_column = T::min_value();
            // let mut max_in_column_row_index = 0;

            // for i in k + 1..self.row_count() {
            //     if max_in_column < u[i][k] {
            //         max_in_column = u[i][k].abs();
            //         max_in_column_row_index = i;
            //     }
            // }

            // // swap kth row with the row, where leftmost nonzero element is the biggest in all rows
            // swaps.push(max_in_column_row_index);
            // u.swap(k, max_in_column_row_index);
            // l.swap(k, max_in_column_row_index);

            for i in k + 1..n {
                l[i][k] = u[i][k] / u[k][k];
                for j in k..n {
                    u[i][j] = u[i][j] - l[i][k] * u[k][j];
                }
            }
        }

        (l, u)
    }

    /// Get z from Lz = b
    fn get_z(l: &Self, b: &Self) -> Self {
        assert_eq!(b.column_count(), 1);
        assert_eq!(l.row_count(), l.column_count());

        let n = l.row_count();

        let mut z = Matrix::like(b);

        for i in 0..n {
            let mut s = T::zero();
            for j in 0..i {
                s = s + l[i][j] * z[j][0];
            }

            z[i][0] = b[i][0] - s;
        }

        z
    }

    /// Get x from Ux = z
    fn get_x(u: &Self, z: &Self) -> Self {
        assert_eq!(z.column_count(), 1);
        assert_eq!(u.row_count(), u.column_count());

        let n = u.row_count();

        let mut x = Matrix::like(z);

        for i in 0..n {
            let i = n - 1 - i;
            let mut s = T::zero();
            for j in i + 1..n {
                s = s + u[i][j] * x[j][0];
            }

            x[i][0] = (z[i][0] - s) / u[i][i];
        }

        x
    }

    pub fn row_count(&self) -> usize {
        self.values.len()
    }

    pub fn column_count(&self) -> usize {
        self.values.first().unwrap().len()
    }

    pub fn zero_matrix(row_count: usize, colum_count: usize) -> Self {
        Self {
            values: (0..row_count)
                .map(|_| (0..colum_count).map(|_| T::zero()).collect())
                .collect(),
        }
    }

    pub fn identity(n: usize) -> Self {
        Self {
            values: (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| if i == j { T::one() } else { T::zero() })
                        .collect()
                })
                .collect(),
        }
    }

    /// Returns a zero matrix with the same dimensions as the input one
    pub fn like(other: &Self) -> Self {
        Matrix::zero_matrix(other.row_count(), other.column_count())
    }
}

impl<T> Deref for Matrix<T> {
    type Target = Vec<Vec<T>>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<T> DerefMut for Matrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl<T> PartialEq for Matrix<T>
where
    T: Float + From<f64>,
{
    fn eq(&self, other: &Self) -> bool {
        for (r1, r2) in self.iter().zip(other.iter()) {
            for (e1, e2) in r1.iter().zip(r2.iter()) {
                if (*e1 - *e2).abs() > Matrix::<T>::EPS.into() {
                    return false;
                }
            }
        }

        true
    }
}

impl<T> Eq for Matrix<T> where T: Float + From<f64> {}

impl<T> Display for Matrix<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut col_widths = Vec::new();
        for row in &self.values {
            for (col, val) in row.iter().enumerate() {
                let width = format!("{:.3}", val).len();
                if col >= col_widths.len() {
                    col_widths.push(width);
                } else {
                    col_widths[col] = col_widths[col].max(width);
                }
            }
        }

        for row in &self.values {
            for (col, val) in row.iter().enumerate() {
                write!(f, "{:>width$.3} ", val, width = col_widths[col])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
