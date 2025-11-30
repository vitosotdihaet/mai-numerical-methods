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

pub mod differential {
    use num::Float;

    pub fn eulers_method<T, F>(f: F, x_range: (T, T), mut y: Vec<T>, step: T) -> Vec<(T, Vec<T>)>
    where
        T: Float,
        F: Fn(T, &Vec<T>) -> Vec<T>,
    {
        let (l, r) = x_range;
        let steps = ((r - l) / step).to_usize().unwrap();
        let mut out = Vec::with_capacity(steps + 1);
        let mut x = l;
        out.push((x, y.clone()));
        for _ in 0..steps {
            let dy = f(x, &y);
            y = y
                .iter()
                .zip(&dy)
                .map(|(&yi, &dyi)| yi + dyi * step)
                .collect();
            x = x + step;
            out.push((x, y.clone()));
        }
        out
    }

    pub fn runge_kutta<T, F>(f: F, x_range: (T, T), mut y: Vec<T>, step: T) -> Vec<(T, Vec<T>)>
    where
        T: Float,
        F: Fn(T, &Vec<T>) -> Vec<T>,
    {
        let (l, r) = x_range;
        let steps = ((r - l) / step).to_usize().unwrap();
        let mut out = Vec::with_capacity(steps + 1);
        let mut x = l;
        out.push((x, y.clone()));
        for _ in 0..steps {
            let k1 = f(x, &y);
            let y2 = y
                .iter()
                .zip(&k1)
                .map(|(&yi, &k1i)| yi + k1i * (step / T::from(2.).unwrap()))
                .collect();
            let k2 = f(x + step / T::from(2.).unwrap(), &y2);
            let y3 = y
                .iter()
                .zip(&k2)
                .map(|(&yi, &k2i)| yi + k2i * (step / T::from(2.).unwrap()))
                .collect();
            let k3 = f(x + step / T::from(2.).unwrap(), &y3);
            let y4 = y
                .iter()
                .zip(&k3)
                .map(|(&yi, &k3i)| yi + k3i * step)
                .collect();
            let k4 = f(x + step, &y4);

            y = y
                .iter()
                .zip(&k1)
                .zip(&k2)
                .zip(&k3)
                .zip(&k4)
                .map(|((((&yi, &k1i), &k2i), &k3i), &k4i)| {
                    yi + (k1i + T::from(2.).unwrap() * k2i + T::from(2.).unwrap() * k3i + k4i)
                        * (step / T::from(6.).unwrap())
                })
                .collect();

            x = x + step;
            out.push((x, y.clone()));
        }
        out
    }

    pub fn adams_method<F, T>(f: F, x_range: (T, T), y0: Vec<T>, step: T) -> Vec<(T, Vec<T>)>
    where
        T: Float,
        F: Fn(T, &Vec<T>) -> Vec<T> + Copy,
    {
        let (l, r) = x_range;
        let steps = ((r - l) / step).to_usize().unwrap();
        let mut out = Vec::with_capacity(steps + 1);
        let x0 = l;

        if steps < 4 {
            return runge_kutta(f, x_range, y0, step);
        }

        let rk_end = x0 + step * T::from(3).unwrap();
        let rk_solution = runge_kutta(f, (x0, rk_end), y0, step);

        for (x, y) in rk_solution {
            out.push((x, y));
        }

        for i in 4..=steps {
            let x = x0 + step * T::from(i).unwrap();

            let y_n = &out[i - 1].1;
            let y_n1 = &out[i - 2].1;
            let y_n2 = &out[i - 3].1;
            let y_n3 = &out[i - 4].1;

            let f_n = f(out[i - 1].0, y_n);
            let f_n1 = f(out[i - 2].0, y_n1);
            let f_n2 = f(out[i - 3].0, y_n2);
            let f_n3 = f(out[i - 4].0, y_n3);

            let next_y: Vec<T> = y_n
                .iter()
                .zip(&f_n)
                .zip(&f_n1)
                .zip(&f_n2)
                .zip(&f_n3)
                .map(|((((&y_val, &f_val), &f_val1), &f_val2), &f_val3)| {
                    let c24 = T::from(24.0).unwrap();
                    let c55 = T::from(55.0).unwrap();
                    let c59 = T::from(59.0).unwrap();
                    let c37 = T::from(37.0).unwrap();
                    let c9 = T::from(9.0).unwrap();

                    y_val + (step / c24) * (c55 * f_val - c59 * f_val1 + c37 * f_val2 - c9 * f_val3)
                })
                .collect();

            out.push((x, next_y));
        }

        out
    }

    pub mod dirichlet {
        use num::Float;

        use crate::matrix::Matrix;

        pub fn pefd<T, F1, F2, F3>(
            phi0: F1,
            phi1: F2,
            psi: F3,
            a: T,
            t_step: T,
            max_t: T,
            x_step: T,
            max_x: T,
        ) -> Vec<Vec<T>>
        where
            T: Float,
            F1: Fn(T) -> T,
            F2: Fn(T) -> T,
            F3: Fn(T) -> T,
        {
            let zero = T::zero();
            let one = T::one();
            let two = T::from(2.0).unwrap();

            let nx = (max_x / x_step).ceil().to_usize().unwrap() + 1;
            let nt = (max_t / t_step).ceil().to_usize().unwrap() + 1;
            let mut xs = vec![zero; nx];
            for (i, x) in xs.iter_mut().enumerate() {
                *x = x_step * T::from(i).unwrap();
            }
            let mut ts = vec![zero; nt];
            for (i, t) in ts.iter_mut().enumerate() {
                *t = t_step * T::from(i).unwrap();
            }

            let sigma = (a * a) * t_step / (x_step * x_step);

            // grid: u[t][x]
            let mut u = vec![vec![zero; nx]; nt];

            // u(x,0) = psi(x)
            for (j, x) in xs.iter().enumerate() {
                u[0][j] = psi(*x);
            }

            u[0][0] = phi0(zero);
            u[0][nx - 1] = phi1(zero);

            for k in 0..(nt - 1) {
                let t_next = ts[k + 1];

                u[k + 1][0] = phi0(t_next);
                u[k + 1][nx - 1] = phi1(t_next);

                for j in 1..(nx - 1) {
                    u[k + 1][j] =
                        sigma * u[k][j + 1] + (one - two * sigma) * u[k][j] + sigma * u[k][j - 1];
                }
            }

            u
        }

        pub fn pifd<T, F1, F2, F3>(
            phi0: F1,
            phi1: F2,
            psi: F3,
            a: T,
            t_step: T,
            max_t: T,
            x_step: T,
            max_x: T,
        ) -> Vec<Vec<T>>
        where
            T: Float,
            F1: Fn(T) -> T,
            F2: Fn(T) -> T,
            F3: Fn(T) -> T,
        {
            let zero = T::zero();
            let one = T::one();
            let two = T::from(2.0).unwrap();

            let nx = (max_x / x_step).ceil().to_usize().unwrap() + 1;
            let nt = (max_t / t_step).ceil().to_usize().unwrap() + 1;
            let mut xs = vec![zero; nx];
            for (i, x) in xs.iter_mut().enumerate() {
                *x = x_step * T::from(i).unwrap();
            }
            let mut ts = vec![zero; nt];
            for (i, t) in ts.iter_mut().enumerate() {
                *t = t_step * T::from(i).unwrap();
            }

            let sigma = (a * a) * t_step / (x_step * x_step);

            // grid: u[t][x]
            let mut u = vec![vec![zero; nx]; 1];

            // u(x,0) = psi(x)
            for (j, x) in xs.iter().enumerate() {
                u[0][j] = psi(*x);
            }

            u[0][0] = phi0(zero);
            u[0][nx - 1] = phi1(zero);

            let aj = -sigma;
            let bj = one + two * sigma;
            let cj = -sigma;

            for k in 0..(nt - 1) {
                let t_next = ts[k + 1];

                let mut m = Matrix::with_capacity(nx);
                let mut d = Matrix::column(&u[k]);
                d[0][0] = d[0][0] + sigma * phi0(t_next);
                d[nx - 1][0] = d[nx - 1][0] + sigma * phi1(t_next);

                let mut row = vec![zero; nx];
                row[0] = bj;
                row[1] = cj;
                m.push(row);

                for j in 1..(nx - 1) {
                    let mut row = vec![zero; nx];
                    row[j - 1] = aj;
                    row[j] = bj;
                    row[j + 1] = cj;
                    m.push(row);
                }

                let mut row = vec![zero; nx];
                row[nx - 2] = aj;
                row[nx - 1] = bj;
                m.push(row);

                u.push(m.solve_tridiagonal(&d).transposed().swap_remove(0));
                u[k + 1][0] = phi0(t_next);
                u[k + 1][nx - 1] = phi1(t_next);
            }

            u
        }

        pub fn crank_nicolsons<T, F1, F2, F3>(
            phi0: F1,
            phi1: F2,
            psi: F3,
            a: T,
            t_step: T,
            max_t: T,
            x_step: T,
            max_x: T,
            theta: T,
        ) -> Vec<Vec<T>>
        where
            T: Float,
            F1: Fn(T) -> T,
            F2: Fn(T) -> T,
            F3: Fn(T) -> T,
        {
            let zero = T::zero();
            let one = T::one();
            let two = T::from(2.0).unwrap();

            let nx = (max_x / x_step).ceil().to_usize().unwrap() + 1;
            let nt = (max_t / t_step).ceil().to_usize().unwrap() + 1;
            let mut xs = vec![zero; nx];
            for (i, x) in xs.iter_mut().enumerate() {
                *x = x_step * T::from(i).unwrap();
            }
            let mut ts = vec![zero; nt];
            for (i, t) in ts.iter_mut().enumerate() {
                *t = t_step * T::from(i).unwrap();
            }

            let sigma = (a * a) * t_step / (x_step * x_step);

            // grid: u[t][x]
            let mut u = vec![vec![zero; nx]; nt];

            // u(x,0) = psi(x)
            for (j, x) in xs.iter().enumerate() {
                u[0][j] = psi(*x);
            }

            u[0][0] = phi0(zero);
            u[0][nx - 1] = phi1(zero);

            let aj = -sigma * theta;
            let bj = one + two * sigma * theta;
            let cj = -sigma * theta;

            for k in 0..(nt - 1) {
                let t_next = ts[k + 1];

                let mut m = Matrix::with_capacity(nx - 2);
                let mut d = Vec::with_capacity(nx - 2);

                for j in 1..(nx - 1) {
                    let explicit_part =
                        (one - theta) * sigma * (u[k][j - 1] - two * u[k][j] + u[k][j + 1]);
                    d.push(u[k][j] + explicit_part);
                }

                d[0] = d[0] + sigma * theta * phi0(t_next);
                d[nx - 3] = d[nx - 3] + sigma * theta * phi1(t_next);

                let mut first_row = vec![zero; nx - 2];
                first_row[0] = bj;
                first_row[1] = cj;
                m.push(first_row);

                for j in 1..(nx - 3) {
                    let mut row = vec![zero; nx - 2];
                    row[j - 1] = aj;
                    row[j] = bj;
                    row[j + 1] = cj;
                    m.push(row);
                }

                let mut last_row = vec![zero; nx - 2];
                last_row[nx - 4] = aj;
                last_row[nx - 3] = bj;
                m.push(last_row);

                let d_matrix = Matrix::column(&d);
                let sol = m.solve_tridiagonal(&d_matrix).transposed().swap_remove(0);

                u[k + 1][0] = phi0(t_next);
                u[k + 1][nx - 1] = phi1(t_next);
                for j in 1..(nx - 1) {
                    u[k + 1][j] = sol[j - 1];
                }
            }

            u
        }
    }

    pub mod robin {
        use num::Float;

        use crate::matrix::Matrix;

        pub fn pefd<T, F1, F2, F3>(
            phi0: F1,
            alpha: T,
            beta: T,
            phi1: F2,
            gamma: T,
            delta: T,
            psi: F3,
            a: T,
            t_step: T,
            max_t: T,
            x_step: T,
            max_x: T,
        ) -> Vec<Vec<T>>
        where
            T: Float,
            F1: Fn(T) -> T,
            F2: Fn(T) -> T,
            F3: Fn(T) -> T,
        {
            let zero = T::zero();
            let one = T::one();
            let two = T::from(2.0).unwrap();

            let nx = (max_x / x_step).ceil().to_usize().unwrap() + 1;
            let nt = (max_t / t_step).ceil().to_usize().unwrap() + 1;
            let mut xs = vec![zero; nx];
            for (i, x) in xs.iter_mut().enumerate() {
                *x = x_step * T::from(i).unwrap();
            }
            let mut ts = vec![zero; nt];
            for (i, t) in ts.iter_mut().enumerate() {
                *t = t_step * T::from(i).unwrap();
            }

            let sigma = (a * a) * t_step / (x_step * x_step);

            // grid: u[t][x]
            let mut u = vec![vec![zero; nx]; nt];

            // u(x,0) = psi(x)
            for (j, x) in xs.iter().enumerate() {
                u[0][j] = psi(*x);
            }

            u[0][0] = -(alpha / x_step) / (beta - alpha / x_step) * u[0][1]
                + phi0(zero) / (beta - alpha / x_step);
            u[0][nx - 1] = (gamma / x_step) / (delta + gamma / x_step) * u[0][nx - 2]
                + phi1(zero) / (delta + gamma / x_step);

            for k in 0..(nt - 1) {
                let t_next = ts[k + 1];

                for j in 1..(nx - 1) {
                    u[k + 1][j] =
                        sigma * u[k][j + 1] + (one - two * sigma) * u[k][j] + sigma * u[k][j - 1];
                }

                u[k + 1][0] = -(alpha / x_step) / (beta - alpha / x_step) * u[k + 1][1]
                    + phi0(t_next) / (beta - alpha / x_step);
                u[k + 1][nx - 1] = (gamma / x_step) / (delta + gamma / x_step) * u[k + 1][nx - 2]
                    + phi1(t_next) / (delta + gamma / x_step);
            }

            u
        }

        pub fn pifd<T, F1, F2, F3>(
            phi0: F1,
            alpha: T,
            beta: T,
            phi1: F2,
            gamma: T,
            delta: T,
            psi: F3,
            a: T,
            t_step: T,
            max_t: T,
            x_step: T,
            max_x: T,
        ) -> Vec<Vec<T>>
        where
            T: Float,
            F1: Fn(T) -> T,
            F2: Fn(T) -> T,
            F3: Fn(T) -> T,
        {
            let zero = T::zero();
            let one = T::one();
            let two = T::from(2.0).unwrap();

            let nx = (max_x / x_step).ceil().to_usize().unwrap() + 1;
            let nt = (max_t / t_step).ceil().to_usize().unwrap() + 1;
            let mut xs = vec![zero; nx];
            for (i, x) in xs.iter_mut().enumerate() {
                *x = x_step * T::from(i).unwrap();
            }
            let mut ts = vec![zero; nt];
            for (i, t) in ts.iter_mut().enumerate() {
                *t = t_step * T::from(i).unwrap();
            }

            let sigma = (a * a) * t_step / (x_step * x_step);

            // grid: u[t][x]
            let mut u = vec![vec![zero; nx]; 1];

            // u(x,0) = psi(x)
            for (j, x) in xs.iter().enumerate() {
                u[0][j] = psi(*x);
            }

            u[0][0] = -(alpha / x_step) / (beta - alpha / x_step) * u[0][1]
                + phi0(zero) / (beta - alpha / x_step);
            u[0][nx - 1] = (gamma / x_step) / (delta + gamma / x_step) * u[0][nx - 2]
                + phi1(zero) / (delta + gamma / x_step);

            let aj = -sigma;
            let bj = one + two * sigma;
            let cj = -sigma;

            for k in 0..(nt - 1) {
                let t_next = ts[k + 1];

                let mut m = Matrix::with_capacity(nx);
                let mut d = Matrix::column(&u[k]);

                d[0][0] = phi0(t_next);
                let mut row = vec![zero; nx];
                row[0] = beta - alpha / x_step;
                row[1] = alpha / x_step;
                m.push(row);

                for j in 1..(nx - 1) {
                    let mut row = vec![zero; nx];
                    row[j - 1] = aj;
                    row[j] = bj;
                    row[j + 1] = cj;
                    m.push(row);
                }

                d[nx - 1][0] = phi1(t_next);
                let mut row = vec![zero; nx];
                row[nx - 2] = -gamma / x_step;
                row[nx - 1] = delta + gamma / x_step;
                m.push(row);

                u.push(m.solve_tridiagonal(&d).transposed().swap_remove(0));
            }

            u
        }
    }
}
