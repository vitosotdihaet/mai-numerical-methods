use std::{
    cell::LazyCell,
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use num::Float;

#[derive(Debug, Clone)]
pub(crate) struct Matrix<T> {
    values: Vec<Vec<T>>,
}

impl<T> Matrix<T>
where
    T: Float + Default + Debug + From<f64>,
{
    const EPS: LazyCell<T> = LazyCell::new(|| 1e-5.into());

    /// # Panics
    /// Zero rows
    pub fn new(values: Vec<Vec<T>>) -> Self {
        let row_len = values.first().unwrap().len();

        for row in &values {
            assert_eq!(row.len(), row_len);
        }

        Self { values }
    }

    // pub fn decompose_lower(&self, b: Self) -> Self {
    //     let mut z = Matrix::zero_matrix(self.row_count(), 1);

    //     z[0][0] = b[0][0];
    //     for i in self.row_count() {
    //         z[i][0] = b[i][0] -
    //     }

    //     z
    // }

    /// # Panics
    /// Matrix is not square
    pub fn get_lu(&self) -> (Self, Self) {
        assert_eq!(self.row_count(), self.column_count());

        let n = self.row_count();

        let mut u = self.clone();
        let mut l = Matrix::identity_matrix(self.row_count());

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
    pub fn get_z(l: &Self, b: &Self) -> Self {
        assert_eq!(b.column_count(), 1);
        assert_eq!(l.row_count(), l.column_count());

        let n = l.row_count();

        let mut z = Matrix::like(&b);

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
    pub fn get_x(u: &Self, z: &Self) -> Self {
        assert_eq!(z.column_count(), 1);
        assert_eq!(u.row_count(), u.column_count());

        let n = u.row_count();

        let mut x = Matrix::like(&z);

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

    pub fn identity_matrix(n: usize) -> Self {
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

    /// Return a zero matrix with the same dimensions as the input one
    pub fn like(other: &Self) -> Self {
        Matrix::zero_matrix(other.row_count(), other.column_count())
    }
}

impl<T> Deref for Matrix<T>
where
    T: Float + Default + Debug,
{
    type Target = Vec<Vec<T>>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<T> DerefMut for Matrix<T>
where
    T: Float + Default + Debug,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl<T> PartialEq for Matrix<T>
where
    T: Float + Default + Debug + From<f64>,
{
    fn eq(&self, other: &Self) -> bool {
        for (r1, r2) in self.iter().zip(other.iter()) {
            for (e1, e2) in r1.iter().zip(r2.iter()) {
                if (*e1 - *e2).abs() > *Matrix::EPS {
                    return false;
                }
            }
        }

        true
    }
}

impl<T> Eq for Matrix<T> where T: Float + Default + Debug + From<f64> {}

impl<T> Display for Matrix<T>
where
    T: Float + Default + Display,
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
