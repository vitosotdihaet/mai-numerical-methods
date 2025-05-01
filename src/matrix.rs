use std::{
    collections::HashMap,
    f64::consts::PI,
    fmt::{Debug, Display},
    ops::{Add, Deref, DerefMut, Mul, Sub},
};

use crate::error::Error as MatrixError;

use num::{Complex, Float};

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

    #[inline(always)]
    pub fn row(values: Vec<T>) -> Self {
        Self {
            values: vec![values],
        }
    }

    #[inline(always)]
    pub fn zero(row_count: usize, colum_count: usize) -> Self {
        Self {
            values: (0..row_count)
                .map(|_| (0..colum_count).map(|_| T::zero()).collect())
                .collect(),
        }
    }

    #[inline(always)]
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
    #[inline(always)]
    pub fn like(other: &Self) -> Self {
        Matrix::zero(other.row_count(), other.column_count())
    }

    pub fn inversed(&self) -> Self {
        self.assert_square();

        let n = self.row_count();

        let e = Matrix::identity(n);

        let (l, u) = self.get_lu();

        let mut inversed_rows = Vec::with_capacity(n);

        for current_e in e.iter() {
            let r = Matrix::solve_lu_with(&l, &u, &Matrix::column(current_e));
            inversed_rows.push(r.transposed().values.into_iter().next().unwrap());
        }

        Matrix::new(inversed_rows).transposed()
    }

    pub fn is_symmetrical(&self) -> bool {
        self.assert_square();
        let n = self.row_count();
        for i in 0..n {
            for j in 0..i {
                if self[i][j] != self[j][i] {
                    return false;
                }
            }
        }
        true
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

        let mut t = Matrix::zero(self.column_count(), self.row_count());

        for i in 0..c {
            for j in 0..r {
                t[i][j] = self[j][i];
            }
        }

        t
    }

    #[inline(always)]
    pub fn row_count(&self) -> usize {
        self.values.len()
    }

    #[inline(always)]
    pub fn column_count(&self) -> usize {
        self.values.first().unwrap().len()
    }

    pub fn norm2(&self) -> T {
        let mut norm = T::zero();
        for row in self.iter() {
            for &a in row {
                norm = norm + a * a
            }
        }
        norm.sqrt()
    }

    pub fn norm1(&self) -> T {
        let mut norm = T::zero();
        for row in self.iter() {
            for &a in row {
                norm = norm + a
            }
        }
        norm
    }

    pub fn eq_lossy(&self, other: &Self, accuracy: T) -> bool {
        for (r1, r2) in self.iter().zip(other.iter()) {
            for (e1, e2) in r1.iter().zip(r2.iter()) {
                if (*e1 - *e2).abs() > accuracy {
                    return false;
                }
            }
        }

        true
    }

    /// # Panics
    /// Matrix is not square
    #[inline(always)]
    fn assert_square(&self) {
        assert_eq!(self.row_count(), self.column_count());
    }

    /// # Panics
    /// Matrix is not a column
    #[inline(always)]
    fn assert_column(&self) {
        assert_eq!(self.column_count(), 1);
    }

    // /// # Panics
    // /// Matrix is not a row
    // #[inline(always)]
    // fn assert_row(&self) {
    //     assert_eq!(self.row_count(), 1);
    // }

    /// # Panics
    /// Matrix is not a row
    #[inline(always)]
    fn assert_symmetrical(&self) {
        assert!(self.is_symmetrical());
    }

    pub fn solve_tridiagonal(&self, d: &Self) -> Self {
        self.assert_square();
        d.assert_column();
        assert_eq!(d.row_count(), self.row_count());

        let n = d.row_count();
        let mut a = vec![T::zero(); n];
        let mut b = vec![T::zero(); n];
        let mut c = vec![T::zero(); n];

        for i in 0..n {
            b[i] = self[i][i];

            if i != 0 {
                a[i] = self[i][i - 1];
            }

            if i != n - 1 {
                c[i] = self[i][i + 1];
            } else {
                assert!(a[i] != T::zero());
            }

            if i == 0 {
                assert!(c[i] != T::zero());
            }

            if i > 0 && i < n - 1 {
                assert!(b[i] >= a[i] + c[i]);
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
        b.assert_column();

        let (l, u) = self.get_lu();

        Matrix::solve_lu_with(&l, &u, b)
    }

    pub fn solve_lu_with(l: &Self, u: &Self, b: &Self) -> Self {
        l.assert_square();
        u.assert_square();
        b.assert_column();

        let z = Matrix::get_z(l, b);
        Matrix::get_x(u, &z)
    }

    pub fn get_lu(&self) -> (Self, Self) {
        self.assert_square();

        let n = self.row_count();

        let mut u = self.clone();
        let mut l = Matrix::identity(self.row_count());

        for k in 0..n {
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
        b.assert_column();
        l.assert_square();

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
        z.assert_column();
        u.assert_square();

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

    /// Returns alpha and beta, where x^(k) = beta + alpha * x^(k - 1)
    fn setup_jacobian_seidel(&self, b: &Self) -> (Self, Self) {
        let n = self.row_count();
        let mut alpha = Matrix::like(self);
        let mut beta = Matrix::like(b);

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    alpha[i][j] = -self[i][j] / self[i][i];
                } else {
                    alpha[i][j] = T::zero();
                    beta[i][0] = b[i][0] / self[i][j];
                }
            }
        }

        (alpha, beta)
    }

    pub fn solve_jacobian(&self, b: &Self, accuracy: T) -> Result<Self, MatrixError> {
        self.assert_square();
        b.assert_column();
        let n = self.row_count();
        assert_eq!(n, b.row_count());

        if !self.jacobian_converges() {
            return Err(MatrixError::MethodDoesNotConverge);
        }

        let (alpha, beta) = self.setup_jacobian_seidel(b);

        let mut prev_x = Matrix::zero(n, 1);
        for i in 0..n {
            prev_x[i][0] = beta[i][0];
        }
        let mut x = &beta + &(&alpha * &prev_x);

        let an = alpha.norm2();
        let iter_count = (accuracy.log(an) + (T::from(1.).unwrap() - an).log(an)
            - (&x - &prev_x).norm2().log(an))
        .to_f64()
        .unwrap() as usize;

        if an >= T::from(1.).unwrap() {
            for _ in 1..1000 {
                prev_x = x;
                x = &beta + &(&alpha * &prev_x);
                if (&prev_x - &x).norm2() < accuracy {
                    break;
                }
            }
        } else {
            for _ in 1..iter_count {
                x = &beta + &(&alpha * &x);
            }
        }

        Ok(x)
    }

    pub fn solve_seidel(&self, b: &Self, accuracy: T) -> Result<Self, MatrixError> {
        self.assert_square();
        b.assert_column();
        let n = self.row_count();
        assert_eq!(n, b.row_count());

        if !self.jacobian_converges() {
            return Err(MatrixError::MethodDoesNotConverge);
        }

        let (mut alpha, mut beta) = self.setup_jacobian_seidel(b);
        let mut big_b = Matrix::like(&alpha);
        let mut big_c = Matrix::like(&alpha);

        for i in 0..n {
            for j in 0..n {
                if j >= i {
                    big_c[i][j] = alpha[i][j];
                } else {
                    big_b[i][j] = alpha[i][j];
                }
            }
        }

        let e = Matrix::identity(n);
        let t = (&e - &big_b).inversed();
        alpha = &t * &big_c;
        beta = &t * &beta;

        let mut prev_x = Matrix::zero(n, 1);
        for i in 0..n {
            prev_x[i][0] = beta[i][0];
        }
        let mut x = &beta + &(&alpha * &prev_x);

        let an = alpha.norm2();
        let iter_count = (accuracy.log(an) + (T::from(1.).unwrap() - an).log(an)
            - (&x - &prev_x).norm2().log(an))
        .to_f64()
        .unwrap() as usize;

        if an >= T::from(1.).unwrap() {
            for _ in 1..1000 {
                prev_x = x;
                x = &beta + &(&alpha * &prev_x);
                if (&prev_x - &x).norm2() < accuracy {
                    break;
                }
            }
        } else {
            for _ in 1..iter_count {
                x = &beta + &(&alpha * &x);
            }
        }

        Ok(x)
    }

    fn jacobian_converges(&self) -> bool {
        let n = self.row_count();
        for i in 0..n {
            let mut s = T::zero();
            for j in 0..n {
                if i != j {
                    s = s + self[i][j].abs();
                }
            }
            if self[i][i].abs() <= s {
                return false;
            }
        }
        true
    }

    pub fn evs_from_symmetrical_matrix(&self, accuracy: T) -> Vec<(T, Self)> {
        self.assert_symmetrical();
        let n = self.row_count();
        let mut evs = Vec::with_capacity(n);

        let mut a_k = self.clone();
        let mut u = Matrix::identity(n);

        loop {
            let mut maxi = -T::infinity();
            let (mut maxi_i, mut maxi_j) = (0, 0);

            for i in 0..n {
                for j in i + 1..n {
                    let a = a_k[i][j].abs();
                    if a > maxi {
                        maxi = a;
                        maxi_i = i;
                        maxi_j = j;
                    }
                }
            }

            let a_ii = a_k[maxi_i][maxi_i];
            let a_jj = a_k[maxi_j][maxi_j];
            let phi = if (a_ii - a_jj).abs() < T::from(Matrix::<T>::EPS).unwrap() {
                T::from(PI / 4.).unwrap()
            } else {
                (T::from(2.).unwrap() * a_k[maxi_i][maxi_j] / (a_ii - a_jj)).atan()
                    / T::from(2.).unwrap()
            };

            let (sin_phi, cos_phi) = phi.sin_cos();
            let mut u_k = Matrix::identity(n);
            u_k[maxi_i][maxi_j] = -sin_phi;
            u_k[maxi_j][maxi_i] = sin_phi;
            u_k[maxi_i][maxi_i] = cos_phi;
            u_k[maxi_j][maxi_j] = cos_phi;

            u = &u * &u_k;

            a_k = &(&u_k.transposed() * &a_k) * &u_k;
            let mut s = T::zero();
            for i in 0..n {
                for j in 0..i {
                    s = s + a_k[i][j] * a_k[i][j];
                }
            }

            if s.sqrt() <= accuracy {
                break;
            }
        }

        for i in 0..n {
            let mut col = Vec::with_capacity(n);
            for j in 0..n {
                col.push(u[j][i]);
            }
            evs.push((a_k[i][i], Matrix::column(&col)));
        }

        evs
    }

    pub fn eigen_values(&self, tolerance: T, max_iterations: u64) -> Vec<Complex<T>> {
        self.assert_square();
        let n = self.row_count();
        let mut a = self.clone();
        let mut converged = vec![false; n];

        for _ in 0..max_iterations {
            let mut all_converged = true;
            let mut i = 0;

            while i < n {
                if converged[i] {
                    i += 1;
                    continue;
                }

                if i < n - 1 && a[i + 1][i].abs() > tolerance {
                    let c = a[i + 1][i];
                    let subdiag_norm = if i < n - 2 {
                        (c * c + a[i + 2][i + 1].powi(2)).sqrt()
                    } else {
                        c.abs()
                    };

                    if subdiag_norm <= tolerance {
                        converged[i] = true;
                        converged[i + 1] = true;
                        i += 2;
                    } else {
                        all_converged = false;
                        i += 1;
                    }
                } else {
                    let sum_squares: T = (i + 1..n)
                        .map(|j| a[j][i] * a[j][i])
                        .fold(T::zero(), |acc, x| acc + x);
                    let norm = sum_squares.sqrt();

                    if norm <= tolerance {
                        converged[i] = true;
                    } else {
                        all_converged = false;
                    }
                    i += 1;
                }
            }

            if converged.iter().all(|&c| c) {
                break;
            }

            let (q, r) = a.get_qr();
            a = &r * &q;
        }

        let mut eigenvalues = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            if i < n - 1 && a[i + 1][i].abs() > tolerance {
                let a_ii = a[i][i];
                let a_ij = a[i][i + 1];
                let a_ji = a[i + 1][i];
                let a_jj = a[i + 1][i + 1];

                let trace = a_ii + a_jj;
                let determinant = a_ii * a_jj - a_ij * a_ji;
                let discriminant = trace.powi(2) - T::from(4.0).unwrap() * determinant;

                if discriminant < T::zero() {
                    let real = trace / T::from(2.0).unwrap();
                    let imag = (-discriminant).sqrt() / T::from(2.0).unwrap();
                    eigenvalues.push(Complex::new(real, imag));
                    eigenvalues.push(Complex::new(real, -imag));
                } else {
                    let sqrt_d = discriminant.sqrt();
                    eigenvalues.push(Complex::new(
                        (trace + sqrt_d) / T::from(2.0).unwrap(),
                        T::zero(),
                    ));
                    eigenvalues.push(Complex::new(
                        (trace - sqrt_d) / T::from(2.0).unwrap(),
                        T::zero(),
                    ));
                }
                i += 2;
            } else {
                eigenvalues.push(Complex::new(a[i][i], T::zero()));
                i += 1;
            }
        }

        eigenvalues
    }

    pub fn get_qr(&self) -> (Self, Self) {
        let n = self.row_count();
        let mut r = self.clone();
        let mut q = Matrix::identity(n);

        for i in 0..n - 1 {
            let mut v = Matrix::zero(n, 1);
            let mut n2 = T::zero();
            for j in i..n {
                n2 = n2 + r[j][i] * r[j][i];
            }
            v[i][0] = r[i][i] + r[i][i].signum() * n2.sqrt();
            for j in i + 1..n {
                v[j][0] = r[j][i];
            }
            let vt = v.transposed();
            let h = Matrix::identity(n)
                - (&(&v * &vt) * (T::one() / (&vt * &v)[0][0])) * T::from(2.).unwrap();
            q = &q * &h;
            r = &h * &r;
        }

        (q, r)
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

impl<T> Add for Matrix<T>
where
    T: Float,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        let n = self.row_count();
        assert_eq!(n, rhs.row_count());
        let m = self.column_count();
        assert_eq!(m, rhs.column_count());

        for i in 0..n {
            for j in 0..m {
                self[i][j] = self[i][j] + rhs[i][j];
            }
        }

        self
    }
}

impl<T> Add for &Matrix<T>
where
    T: Float,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let n = self.row_count();
        assert_eq!(n, rhs.row_count());
        let m = self.column_count();
        assert_eq!(m, rhs.column_count());

        let mut res = Matrix::like(self);
        for i in 0..n {
            for j in 0..m {
                res[i][j] = self[i][j] + rhs[i][j];
            }
        }

        res
    }
}

impl<T> Sub for &Matrix<T>
where
    T: Float,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let n = self.row_count();
        assert_eq!(n, rhs.row_count());
        let m = self.column_count();
        assert_eq!(m, rhs.column_count());

        let mut res = Matrix::like(self);
        for i in 0..n {
            for j in 0..m {
                res[i][j] = self[i][j] - rhs[i][j];
            }
        }

        res
    }
}

impl<T> Sub for Matrix<T>
where
    T: Float,
{
    type Output = Matrix<T>;

    fn sub(mut self, rhs: Self) -> Self::Output {
        let n = self.row_count();
        assert_eq!(n, rhs.row_count());
        let m = self.column_count();
        assert_eq!(m, rhs.column_count());

        for i in 0..n {
            for j in 0..m {
                self[i][j] = self[i][j] - rhs[i][j];
            }
        }

        self
    }
}

impl<T> Mul for &Matrix<T>
where
    T: Float,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let p = self.row_count();
        let q = self.column_count();
        assert_eq!(q, rhs.row_count());
        let r = rhs.column_count();

        let mut out = Matrix::zero(p, r);

        for i in 0..p {
            for j in 0..r {
                for k in 0..q {
                    out[i][j] = out[i][j] + self[i][k] * rhs[k][j];
                }
            }
        }

        out
    }
}

impl<T> Mul<T> for Matrix<T>
where
    T: Float,
{
    type Output = Matrix<T>;

    fn mul(mut self, num: T) -> Self::Output {
        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                self[i][j] = self[i][j] * num;
            }
        }

        self
    }
}

impl<T> Mul<T> for &Matrix<T>
where
    T: Float,
{
    type Output = Matrix<T>;

    fn mul(self, num: T) -> Self::Output {
        let mut out = Matrix::like(self);
        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                out[i][j] = self[i][j] * num;
            }
        }

        out
    }
}
