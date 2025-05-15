use crate::error::Error;
use num::Float;

pub fn halves_method<T, F>(f: F, x_range: (T, T), accuracy: T) -> T
where
    T: Float,
    F: Fn(T) -> T,
{
    let (mut l, mut r) = x_range;

    loop {
        let m = (l + r) / T::from(2.).unwrap();
        if r - l < T::from(2.).unwrap() * accuracy {
            break m;
        }

        let fm_signum = f(m).signum();
        if f(l).signum() != fm_signum {
            r = m;
        } else if f(r).signum() != fm_signum {
            l = m;
        } else {
            break m;
        }
    }
}

pub fn newtons_method<T, F>(f: F, x_approximate: T, accuracy: T) -> T
where
    T: Float,
    F: Fn(T) -> T,
{
    let mut x_prev = x_approximate;
    let mut fx_prev = f(x_prev);
    let mut x = x_approximate + accuracy;
    let mut fx = f(x);
    loop {
        let t = x;
        x = x - fx * (x - x_prev) / (fx - fx_prev);
        x_prev = t;

        if (x_prev - x).abs() < accuracy {
            return x;
        }

        fx_prev = fx;
        fx = f(x);
    }
}

pub fn iterations_method<T, F>(phi: F, x_approximate: T, accuracy: T) -> Result<T, Error>
where
    T: Float,
    F: Fn(T) -> T,
{
    let mut x = x_approximate;
    loop {
        let t = x;
        x = phi(x);
        if x > T::one() {
            break Err(Error::MethodDoesNotConverge);
        }
        if (x - t).abs() < accuracy {
            break Ok(x);
        }
    }
}

pub mod systems {
    use num::Float;

    use crate::{error::Error, matrix::Matrix};

    /// derivatives is \[df1/dx1, df1/dx2, ..., df1/dxn, df2/dx1, df2/dx2, ..., df2/dxn, ...\]
    pub fn newtons_method<T, F>(
        fs: &[F],
        derivatives: &[F],
        x_approximates: &[T],
        accuracy: T,
    ) -> Vec<T>
    where
        T: Float,
        F: Fn(&[T]) -> T,
    {
        let n = fs.len();
        assert_eq!(n * n, derivatives.len());
        assert_eq!(n, x_approximates.len());

        let mut x_prev = Vec::from(x_approximates);

        let mut t: Vec<Vec<T>> = Vec::with_capacity(n);
        for i in 0..n {
            t.push(Vec::with_capacity(n));
            for j in 0..n {
                t[i].push(derivatives[i * n + j](&x_prev));
            }
        }
        let jacobi_inversed = Matrix::new(t).inversed();

        let mut t: Vec<T> = vec![T::zero(); n];
        loop {
            let mut x = Vec::with_capacity(n);
            for i in 0..n {
                t[i] = fs[i](&x_prev);
            }
            let fxk = Matrix::column(&t);
            let m = &jacobi_inversed * &fxk;
            let mut s = T::zero();
            for i in 0..n {
                x.push(x_prev[i] - m[i][0]);
                s = s + (x[i] - x_prev[i]).powi(2);
            }

            if s.sqrt() < accuracy {
                break x;
            }

            x_prev = x;
        }
    }

    pub fn iterations_method<T, F>(
        phis: &[F],
        phi_derivatives: &[F],
        x_approximates: &[T],
        accuracy: T,
    ) -> Result<Vec<T>, Error>
    where
        T: Float,
        F: Fn(&[T]) -> T,
    {
        let n = phis.len();
        assert_eq!(n * n, phi_derivatives.len());
        assert_eq!(n, x_approximates.len());

        let mut x_prev = Vec::from(x_approximates);

        loop {
            if phi_derivatives
                .iter()
                .fold(T::zero(), |acc, dphi| acc.max(dphi(&x_prev).abs()))
                >= T::one()
            {
                break Err(Error::MethodDoesNotConverge);
            }

            let mut x = Vec::with_capacity(n);
            let mut s = T::zero();
            for i in 0..n {
                x.push(phis[i](&x_prev));
                s = s + (x_prev[i] - x[i]).powi(2);
            }

            if s.sqrt() < accuracy {
                break Ok(x);
            }

            x_prev = x;
        }
    }
}
