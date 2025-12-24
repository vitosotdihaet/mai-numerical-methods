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

    pub mod parabolic {

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
                b: T,
                c: T,
                t_step_count: usize,
                max_t: T,
                x_step_count: usize,
                max_x: T,
            ) -> Vec<Vec<T>>
            where
                T: Float,
                F1: Fn(T) -> T,
                F2: Fn(T) -> T,
                F3: Fn(T) -> T,
            {
                let zero = T::zero();

                let x_step = max_x / T::from(x_step_count).unwrap();
                let t_step = max_t / T::from(t_step_count).unwrap();

                let nx = x_step_count + 1;
                let nt = t_step_count + 1;
                let mut xs = vec![zero; nx];
                for (i, x) in xs.iter_mut().enumerate() {
                    *x = x_step * T::from(i).unwrap();
                }
                let mut ts = vec![zero; nt];
                for (i, t) in ts.iter_mut().enumerate() {
                    *t = t_step * T::from(i).unwrap();
                }

                // grid: u[t][x]
                let mut u = vec![vec![zero; nx]; nt];

                // u(x,0) = psi(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][j] = psi(*x);
                }

                for k in 0..(nt - 1) {
                    let t_next = ts[k + 1];

                    for j in 1..(nx - 1) {
                        u[k + 1][j] = u[k][j]
                            + a * t_step * (u[k][j + 1] - u[k][j] - u[k][j] + u[k][j - 1])
                                / (x_step * x_step)
                            + b * t_step * (u[k][j + 1] - u[k][j - 1]) / (x_step + x_step)
                            + c * t_step * u[k][j];
                    }

                    u[k + 1][0] = (x_step * phi0(t_next) - u[k + 1][1]) / (beta * x_step - alpha);
                    u[k + 1][nx - 1] =
                        (x_step * phi1(t_next) + u[k + 1][nx - 2]) / (delta * x_step + gamma);
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
                b: T,
                c: T,
                t_step_count: usize,
                max_t: T,
                x_step_count: usize,
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

                let x_step = max_x / T::from(x_step_count).unwrap();
                let t_step = max_t / T::from(t_step_count).unwrap();

                let nx = x_step_count + 1;
                let nt = t_step_count + 1;
                let mut xs = vec![zero; nx];
                for (i, x) in xs.iter_mut().enumerate() {
                    *x = x_step * T::from(i).unwrap();
                }
                let mut ts = vec![zero; nt];
                for (i, t) in ts.iter_mut().enumerate() {
                    *t = t_step * T::from(i).unwrap();
                }

                // grid: u[t][x]
                let mut u = vec![vec![zero; nx]; 1];

                // u(x,0) = psi(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][j] = psi(*x);
                }

                let aj = -a * t_step / (x_step * x_step) + b * t_step / (x_step + x_step);
                let bj = (two * a * t_step) / (x_step * x_step) - c * t_step + one;
                let cj = -a * t_step / (x_step * x_step) - b * t_step / (x_step + x_step);

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

            pub fn crank_nicolsons<T, F1, F2, F3>(
                phi0: F1,
                alpha: T,
                beta: T,
                phi1: F2,
                gamma: T,
                delta: T,
                psi: F3,
                a: T,
                b: T,
                c: T,
                t_step_count: usize,
                max_t: T,
                x_step_count: usize,
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

                let x_step = max_x / T::from(x_step_count).unwrap();
                let t_step = max_t / T::from(t_step_count).unwrap();

                let nx = x_step_count + 1;
                let nt = t_step_count + 1;
                let mut xs = vec![zero; nx];
                for (i, x) in xs.iter_mut().enumerate() {
                    *x = x_step * T::from(i).unwrap();
                }
                let mut ts = vec![zero; nt];
                for (i, t) in ts.iter_mut().enumerate() {
                    *t = t_step * T::from(i).unwrap();
                }

                // grid: u[t][x]
                let mut u = vec![vec![zero; nx]; 1];

                // u(x,0) = psi(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][j] = psi(*x);
                }

                let aj = (-a * t_step / (x_step * x_step) + b * t_step / (x_step + x_step)) / two;
                let bj = ((two * a * t_step) / (x_step * x_step) - c * t_step) / two + one;
                let cj = (-a * t_step / (x_step * x_step) - b * t_step / (x_step + x_step)) / two;

                for k in 0..(nt - 1) {
                    let t_next = ts[k + 1];

                    let mut m = Matrix::with_capacity(nx);
                    let mut d = Matrix::column(&u[k]);

                    for j in 1..(nx - 1) {
                        d[j][0] = (a * t_step * (u[k][j + 1] - u[k][j] - u[k][j] + u[k][j - 1])
                            / (x_step * x_step)
                            + b * t_step * (u[k][j + 1] - u[k][j - 1]) / (x_step + x_step)
                            + c * t_step * u[k][j])
                            / two
                            + u[k][j];
                    }

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

    pub mod hyperbolic {
        pub mod dirichlet {
            use num::Float;

            use crate::matrix::Matrix;

            pub fn pefd<T, F1, F2, F3, F4, F5>(
                phi0: F1,
                phi1: F2,
                psi1: F3,
                psi2: F4,
                a: T,
                b: T,
                c: T,
                d: F5,
                alpha: T,
                beta: T,
                t_step_count: usize,
                max_t: T,
                x_step_count: usize,
                max_x: T,
            ) -> Vec<Vec<T>>
            where
                T: Float,
                F1: Fn(T) -> T,
                F2: Fn(T) -> T,
                F3: Fn(T) -> T,
                F4: Fn(T) -> T,
                F5: Fn(T, T) -> T,
            {
                let zero = T::zero();
                let one = T::one();
                let two = T::from(2.0).unwrap();

                let x_step = max_x / T::from(x_step_count).unwrap();
                let t_step = max_t / T::from(t_step_count).unwrap();

                let nx = x_step_count + 1;
                let nt = t_step_count + 1;
                let mut xs = vec![zero; nx];
                for (i, x) in xs.iter_mut().enumerate() {
                    *x = x_step * T::from(i).unwrap();
                }
                let mut ts = vec![zero; nt];
                for (i, t) in ts.iter_mut().enumerate() {
                    *t = t_step * T::from(i).unwrap();
                }

                // grid: u[t][x]
                let mut u = vec![vec![zero; nx]; nt];

                // u(x,t_0) = psi(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][j] = psi1(*x);
                }

                // u(x,t_1) = psi(x)
                for (j, x) in xs.iter().enumerate() {
                    u[1][j] = psi1(*x) + t_step * psi2(*x);
                }

                let u_kp1_jp1_arg = (t_step * t_step / alpha / x_step) * (a / x_step + b / two);
                let u_kp1_j_arg = two
                    + (t_step / alpha) * (-beta - two * a * t_step / x_step / x_step + c * t_step);
                let u_kp1_jm1_arg = (t_step * t_step / alpha / x_step) * (a / x_step - b / two);
                let u_k_j_arg = beta * t_step / alpha - one;
                let free_arg = |x: T, t: T| d(x, t) * t_step * t_step / alpha;

                for k in 0..(nt - 2) {
                    let t = ts[k + 2];

                    for j in 1..(nx - 1) {
                        let x = xs[j];
                        u[k + 2][j] = u[k + 1][j + 1] * u_kp1_jp1_arg
                            + u[k + 1][j] * u_kp1_j_arg
                            + u[k + 1][j - 1] * u_kp1_jm1_arg
                            + u[k][j] * u_k_j_arg
                            + free_arg(x, t);
                    }

                    u[k + 2][0] = -x_step * phi0(t) + u[k + 2][1];
                    u[k + 2][nx - 1] = x_step * phi1(t) + u[k + 2][nx - 2];
                }

                u
            }

            pub fn pifd<T, F1, F2, F3, F4, F5>(
                phi0: F1,
                phi1: F2,
                psi1: F3,
                psi2: F4,
                a: T,
                b: T,
                c: T,
                d: F5,
                alpha: T,
                beta: T,
                t_step_count: usize,
                max_t: T,
                x_step_count: usize,
                max_x: T,
            ) -> Vec<Vec<T>>
            where
                T: Float,
                F1: Fn(T) -> T,
                F2: Fn(T) -> T,
                F3: Fn(T) -> T,
                F4: Fn(T) -> T,
                F5: Fn(T, T) -> T,
            {
                let zero = T::zero();
                let one = T::one();
                let two = T::from(2.0).unwrap();

                let x_step = max_x / T::from(x_step_count).unwrap();
                let t_step = max_t / T::from(t_step_count).unwrap();

                let nx = x_step_count + 1;
                let nt = t_step_count + 1;
                let mut xs = vec![zero; nx];
                for (i, x) in xs.iter_mut().enumerate() {
                    *x = x_step * T::from(i).unwrap();
                }
                let mut ts = vec![zero; nt];
                for (i, t) in ts.iter_mut().enumerate() {
                    *t = t_step * T::from(i).unwrap();
                }

                // grid: u[t][x]
                let mut u = vec![vec![zero; nx]; 2];

                // u(x,t_0) = psi1(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][j] = psi1(*x);
                }

                // u(x,t_1)
                for (j, x) in xs.iter().enumerate() {
                    u[1][j] = psi1(*x) + t_step * psi2(*x);
                }

                // left side
                let u_k_j_arg = -two - beta * t_step / alpha;
                let u_km1_j_arg = one;
                let free_arg = |x: T, t: T| -d(x, t) * t_step * t_step / alpha;

                // right side
                let u_kp1_jp1_arg = a * t_step * t_step / alpha / x_step / x_step
                    + b * t_step * t_step / alpha / two / x_step;
                let u_kp1_j_arg = -two * a * t_step * t_step / alpha / x_step / x_step
                    - beta * t_step / alpha
                    + c * t_step * t_step / alpha
                    - one;
                let u_kp1_jm1_arg = a * t_step * t_step / alpha / x_step / x_step
                    - b * t_step * t_step / alpha / two / x_step;

                for k in 0..(nt - 2) {
                    let t = ts[k + 2];

                    let mut m = Matrix::with_capacity(nx);
                    let mut d = Matrix::zero(nx, 1);

                    d[0][0] = phi0(t);
                    let mut row = vec![zero; nx];
                    row[0] = -one / x_step;
                    row[1] = one / x_step;
                    m.push(row);

                    for j in 1..(nx - 1) {
                        let x = xs[j];
                        let mut row = vec![zero; nx];
                        row[j - 1] = u_kp1_jm1_arg;
                        row[j] = u_kp1_j_arg;
                        row[j + 1] = u_kp1_jp1_arg;
                        d[j][0] = u_km1_j_arg * u[k][j] + u_k_j_arg * u[k + 1][j] + free_arg(x, t);
                        m.push(row);
                    }

                    d[nx - 1][0] = phi1(t);
                    let mut row = vec![zero; nx];
                    row[nx - 2] = -one / x_step;
                    row[nx - 1] = one / x_step;
                    m.push(row);

                    u.push(m.solve_tridiagonal(&d).transposed().swap_remove(0));
                }

                u
            }
        }
    }

    pub mod elliptic {
        pub mod dirichlet {

            use num::Float;

            use crate::matrix::Matrix;

            pub fn iterative_method<T, F1, F2, F3, F4>(
                phi0: F1,
                phi1: F2,
                psi0: F3,
                psi1: F4,
                a: T,
                b: T,
                c: T,
                alpha: T,
                beta: T,
                x_step_count: usize,
                max_x: T,
                y_step_count: usize,
                max_y: T,
                eps: T,
                max_iter: usize,
            ) -> Vec<Matrix<T>>
            where
                T: Float,
                F1: Fn(T) -> T,
                F2: Fn(T) -> T,
                F3: Fn(T) -> T,
                F4: Fn(T) -> T,
            {
                let zero = T::zero();
                let two = T::from(2.0).unwrap();

                let x_step = max_x / T::from(x_step_count).unwrap();
                let y_step = max_y / T::from(y_step_count).unwrap();

                let nx = x_step_count + 1;
                let ny = y_step_count + 1;
                let mut xs = vec![zero; nx];
                for (i, x) in xs.iter_mut().enumerate() {
                    *x = x_step * T::from(i).unwrap();
                }
                let mut ys = vec![zero; ny];
                for (i, y) in ys.iter_mut().enumerate() {
                    *y = y_step * T::from(i).unwrap();
                }

                // grid: u[y][x]
                let mut u = vec![Matrix::zero(ny, nx)];

                // u(x, 0) = psi1(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][0][j] = psi0(*x);
                }

                // u(x, l2) = psi2(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][ny - 1][j] = psi1(*x);
                }

                // u(0, y) = phi1(x)
                for (i, y) in ys.iter().enumerate() {
                    u[0][i][0] = phi0(*y);
                }

                // u(l1, y) = phi2(x)
                for (i, y) in ys.iter().enumerate() {
                    u[0][i][nx - 1] = phi1(*y);
                }

                for i in 1..(ny - 1) {
                    for j in 1..(nx - 1) {
                        let left = u[0][i][0];
                        let right = u[0][i][nx - 1];
                        let bottom = u[0][0][j];
                        let top = u[0][ny - 1][j];

                        u[0][i][j] = (left
                            + (right - left) * T::from(j).unwrap() / T::from(nx - 1).unwrap()
                            + bottom
                            + (top - bottom) * T::from(i).unwrap() / T::from(ny - 1).unwrap())
                            / two;
                    }
                }

                let u_kp1_i_j_arg =
                    -two * alpha / x_step / x_step - two * beta / y_step / y_step + c;
                let u_ijp1_p_u_ijm1_arg = -alpha / x_step / x_step / u_kp1_i_j_arg;
                let u_ip1j_p_u_im1j_arg = -beta / y_step / y_step / u_kp1_i_j_arg;
                let u_ijp1_m_u_ijm1_arg = a / two / x_step / u_kp1_i_j_arg;
                let u_ip1j_m_u_im1j_arg = b / two / y_step / u_kp1_i_j_arg;

                for k in 0..max_iter {
                    let mut new_layer = Matrix::like(&u[k]);

                    // u(x, 0) = psi1(x)
                    for (j, x) in xs.iter().enumerate() {
                        new_layer[0][j] = psi0(*x);
                    }

                    // u(x, l2) = psi2(x)
                    for (j, x) in xs.iter().enumerate() {
                        new_layer[ny - 1][j] = psi1(*x);
                    }

                    // u(0, y) = phi1(x)
                    for (i, y) in ys.iter().enumerate() {
                        new_layer[i][0] = phi0(*y);
                    }

                    // u(l1, y) = phi2(x)
                    for (i, y) in ys.iter().enumerate() {
                        new_layer[i][nx - 1] = phi1(*y);
                    }

                    let mut max_diff = T::zero();
                    let cu = &u[k];
                    for i in 1..(ny - 1) {
                        for j in 1..(nx - 1) {
                            new_layer[i][j] = u_ijp1_p_u_ijm1_arg * (cu[i][j + 1] + cu[i][j - 1])
                                + u_ip1j_p_u_im1j_arg * (cu[i + 1][j] + cu[i - 1][j])
                                + u_ijp1_m_u_ijm1_arg * (cu[i][j + 1] - cu[i][j - 1])
                                + u_ip1j_m_u_im1j_arg * (cu[i + 1][j] - cu[i - 1][j]);
                            let diff = (new_layer[i][j] - cu[i][j]).abs();
                            if diff > max_diff {
                                max_diff = diff;
                            }
                        }
                    }

                    u.push(new_layer);

                    // see https://www.desmos.com/calculator/dyveh5rhpi for the example
                    if max_diff <= eps {
                        break;
                    }
                }

                u
            }

            pub fn relaxed_method<T, F1, F2, F3, F4>(
                phi0: F1,
                phi1: F2,
                psi0: F3,
                psi1: F4,
                a: T,
                b: T,
                c: T,
                alpha: T,
                beta: T,
                x_step_count: usize,
                max_x: T,
                y_step_count: usize,
                max_y: T,
                eps: T,
                max_iter: usize,
                relax_amount: T,
            ) -> Vec<Matrix<T>>
            where
                T: Float,
                F1: Fn(T) -> T,
                F2: Fn(T) -> T,
                F3: Fn(T) -> T,
                F4: Fn(T) -> T,
            {
                let zero = T::zero();
                let one = T::one();
                let two = T::from(2.0).unwrap();

                let x_step = max_x / T::from(x_step_count).unwrap();
                let y_step = max_y / T::from(y_step_count).unwrap();

                let nx = x_step_count + 1;
                let ny = y_step_count + 1;
                let mut xs = vec![zero; nx];
                for (i, x) in xs.iter_mut().enumerate() {
                    *x = x_step * T::from(i).unwrap();
                }
                let mut ys = vec![zero; ny];
                for (i, y) in ys.iter_mut().enumerate() {
                    *y = y_step * T::from(i).unwrap();
                }

                // grid: u[y][x]
                let mut u = vec![Matrix::zero(ny, nx)];

                // u(x, 0) = psi1(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][0][j] = psi0(*x);
                }

                // u(x, l2) = psi2(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][ny - 1][j] = psi1(*x);
                }

                // u(0, y) = phi1(x)
                for (i, y) in ys.iter().enumerate() {
                    u[0][i][0] = phi0(*y);
                }

                // u(l1, y) = phi2(x)
                for (i, y) in ys.iter().enumerate() {
                    u[0][i][nx - 1] = phi1(*y);
                }

                for i in 1..(ny - 1) {
                    for j in 1..(nx - 1) {
                        let left = u[0][i][0];
                        let right = u[0][i][nx - 1];
                        let bottom = u[0][0][j];
                        let top = u[0][ny - 1][j];

                        u[0][i][j] = (left
                            + (right - left) * T::from(j).unwrap() / T::from(nx - 1).unwrap()
                            + bottom
                            + (top - bottom) * T::from(i).unwrap() / T::from(ny - 1).unwrap())
                            / two;
                    }
                }

                let u_kp1_i_j_arg =
                    -two * alpha / x_step / x_step - two * beta / y_step / y_step + c;
                let u_ijp1_p_u_ijm1_arg = -alpha / x_step / x_step / u_kp1_i_j_arg;
                let u_ip1j_p_u_im1j_arg = -beta / y_step / y_step / u_kp1_i_j_arg;
                let u_ijp1_m_u_ijm1_arg = a / two / x_step / u_kp1_i_j_arg;
                let u_ip1j_m_u_im1j_arg = b / two / y_step / u_kp1_i_j_arg;

                for k in 0..max_iter {
                    let mut new_layer = Matrix::like(&u[k]);

                    // u(x, 0) = psi1(x)
                    for (j, x) in xs.iter().enumerate() {
                        new_layer[0][j] = psi0(*x);
                    }

                    // u(x, l2) = psi2(x)
                    for (j, x) in xs.iter().enumerate() {
                        new_layer[ny - 1][j] = psi1(*x);
                    }

                    // u(0, y) = phi1(x)
                    for (i, y) in ys.iter().enumerate() {
                        new_layer[i][0] = phi0(*y);
                    }

                    // u(l1, y) = phi2(x)
                    for (i, y) in ys.iter().enumerate() {
                        new_layer[i][nx - 1] = phi1(*y);
                    }

                    let mut max_diff = T::zero();
                    let cu = &u[k];
                    for i in 1..(ny - 1) {
                        for j in 1..(nx - 1) {
                            new_layer[i][j] = cu[i][j] * relax_amount
                                + (one - relax_amount)
                                    * (u_ijp1_p_u_ijm1_arg * (cu[i][j + 1] + cu[i][j - 1])
                                        + u_ip1j_p_u_im1j_arg * (cu[i + 1][j] + cu[i - 1][j])
                                        + u_ijp1_m_u_ijm1_arg * (cu[i][j + 1] - cu[i][j - 1])
                                        + u_ip1j_m_u_im1j_arg * (cu[i + 1][j] - cu[i - 1][j]));
                            let diff = (new_layer[i][j] - cu[i][j]).abs();
                            if diff > max_diff {
                                max_diff = diff;
                            }
                        }
                    }

                    u.push(new_layer);

                    // see https://www.desmos.com/calculator/dyveh5rhpi for the example
                    if max_diff <= eps {
                        break;
                    }
                }

                u
            }

            pub fn seidel<T, F1, F2, F3, F4>(
                phi0: F1,
                phi1: F2,
                psi0: F3,
                psi1: F4,
                a: T,
                b: T,
                c: T,
                alpha: T,
                beta: T,
                x_step_count: usize,
                max_x: T,
                y_step_count: usize,
                max_y: T,
                eps: T,
                max_iter: usize,
            ) -> Vec<Matrix<T>>
            where
                T: Float,
                F1: Fn(T) -> T,
                F2: Fn(T) -> T,
                F3: Fn(T) -> T,
                F4: Fn(T) -> T,
            {
                let zero = T::zero();
                let two = T::from(2.0).unwrap();

                let x_step = max_x / T::from(x_step_count).unwrap();
                let y_step = max_y / T::from(y_step_count).unwrap();

                let nx = x_step_count + 1;
                let ny = y_step_count + 1;
                let mut xs = vec![zero; nx];
                for (i, x) in xs.iter_mut().enumerate() {
                    *x = x_step * T::from(i).unwrap();
                }
                let mut ys = vec![zero; ny];
                for (i, y) in ys.iter_mut().enumerate() {
                    *y = y_step * T::from(i).unwrap();
                }

                // grid: u[y][x]
                let mut u = vec![Matrix::zero(ny, nx)];

                // u(x, 0) = psi1(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][0][j] = psi0(*x);
                }

                // u(x, l2) = psi2(x)
                for (j, x) in xs.iter().enumerate() {
                    u[0][ny - 1][j] = psi1(*x);
                }

                // u(0, y) = phi1(x)
                for (i, y) in ys.iter().enumerate() {
                    u[0][i][0] = phi0(*y);
                }

                // u(l1, y) = phi2(x)
                for (i, y) in ys.iter().enumerate() {
                    u[0][i][nx - 1] = phi1(*y);
                }

                for i in 1..(ny - 1) {
                    for j in 1..(nx - 1) {
                        let left = u[0][i][0];
                        let right = u[0][i][nx - 1];
                        let bottom = u[0][0][j];
                        let top = u[0][ny - 1][j];

                        u[0][i][j] = (left
                            + (right - left) * T::from(j).unwrap() / T::from(nx - 1).unwrap()
                            + bottom
                            + (top - bottom) * T::from(i).unwrap() / T::from(ny - 1).unwrap())
                            / two;
                    }
                }

                let u_kp1_i_j_arg =
                    -two * alpha / x_step / x_step - two * beta / y_step / y_step + c;
                let u_ijp1_p_u_ijm1_arg = -alpha / x_step / x_step / u_kp1_i_j_arg;
                let u_ip1j_p_u_im1j_arg = -beta / y_step / y_step / u_kp1_i_j_arg;
                let u_ijp1_m_u_ijm1_arg = a / two / x_step / u_kp1_i_j_arg;
                let u_ip1j_m_u_im1j_arg = b / two / y_step / u_kp1_i_j_arg;

                for k in 0..max_iter {
                    let mut max_diff = T::zero();
                    let cu = &u[k];
                    let mut new_layer = cu.clone();
                    for i in 1..(ny - 1) {
                        for j in 1..(nx - 1) {
                            let prev = cu[i][j];
                            new_layer[i][j] = u_ijp1_p_u_ijm1_arg
                                * (new_layer[i][j + 1] + new_layer[i][j - 1])
                                + u_ip1j_p_u_im1j_arg * (new_layer[i + 1][j] + new_layer[i - 1][j])
                                + u_ijp1_m_u_ijm1_arg * (new_layer[i][j + 1] - new_layer[i][j - 1])
                                + u_ip1j_m_u_im1j_arg * (new_layer[i + 1][j] - new_layer[i - 1][j]);
                            let diff = (new_layer[i][j] - prev).abs();
                            if diff > max_diff {
                                max_diff = diff;
                            }
                        }
                    }

                    u.push(new_layer);

                    // see https://www.desmos.com/calculator/dyveh5rhpi for the example
                    if max_diff <= eps {
                        break;
                    }
                }

                u
            }
        }
    }

    pub mod two_dimensional {
        pub mod parabolic {
            use num::Float;

            use crate::matrix::Matrix;

            pub fn variable_direction_method<T, F1, F2, F3, F4, F5, FF>(
                phi0: F1,
                phi1: F2,
                psi0: F3,
                psi1: F4,
                khi: F5,
                a: T,
                b: T,
                f: FF,
                x_step_count: usize,
                max_x: T,
                y_step_count: usize,
                max_y: T,
                t_step_count: usize,
                max_t: T,
            ) -> Vec<Matrix<T>>
            where
                T: Float,
                F1: Fn(T, T) -> T,
                F2: Fn(T, T) -> T,
                F3: Fn(T, T) -> T,
                F4: Fn(T, T) -> T,
                F5: Fn(T, T) -> T,
                FF: Fn(T, T, T) -> T,
            {
                let zero = T::zero();
                let one = T::one();
                let two = T::from(2.0).unwrap();

                let x_step = max_x / T::from(x_step_count).unwrap();
                let y_step = max_y / T::from(y_step_count).unwrap();
                let t_step = max_t / T::from(t_step_count).unwrap();

                let nx = x_step_count + 1;
                let ny = y_step_count + 1;
                let nt = t_step_count + 1;
                let mut xs = vec![zero; nx];
                for (i, x) in xs.iter_mut().enumerate() {
                    *x = x_step * T::from(i).unwrap();
                }
                let mut ys = vec![zero; ny];
                for (i, y) in ys.iter_mut().enumerate() {
                    *y = y_step * T::from(i).unwrap();
                }
                let mut ts = vec![zero; nt];
                for (i, t) in ts.iter_mut().enumerate() {
                    *t = t_step * T::from(i).unwrap();
                }

                let mut u = Vec::with_capacity(nt);
                u.push(Matrix::zero(ny, nx));

                for (i, y) in ys.iter().enumerate() {
                    for (j, x) in xs.iter().enumerate() {
                        u[0][i][j] = khi(*x, *y);
                    }
                }

                // temporal layer k + 1/2: implicit scheme for y
                let u_khalf_im1_j_arg = -b / y_step / y_step;
                let u_khalf_i_j_arg = two * b / y_step / y_step + two / t_step;
                let u_khalf_ip1_j_arg = -b / y_step / y_step;

                // temporal layer k + 1: implicit scheme for x
                let u_kn_i_jm1_arg = -a / x_step / x_step;
                let u_kn_i_j_arg = two * a / x_step / x_step + two / t_step;
                let u_kn_i_jp1_arg = -a / x_step / x_step;

                for k in 0..(nt - 1) {
                    // temporal layer k + 1/2: implicit scheme for y
                    let mut first_temporal_layer = Matrix::zero(ny, nx);
                    let t_half = (ts[k + 1] + ts[k]) / two;

                    // u_kn_i_0 = phi0(y, t)
                    for (i, &y) in ys.iter().enumerate() {
                        first_temporal_layer[i][0] = phi0(y, t_half);
                    }

                    for j in 1..(nx - 1) {
                        let x = xs[j];
                        let mut m = Matrix::with_capacity(ny);
                        let mut d = Matrix::zero(ny, 1);

                        d[0][0] = psi0(x, t_half);
                        let mut row = vec![zero; ny];
                        row[0] = one;
                        m.push(row);

                        for i in 1..(ny - 1) {
                            d[i][0] = two * u[k][i][j] / t_step
                                + a / x_step / x_step
                                    * (u[k][i][j + 1] - two * u[k][i][j] + u[k][i][j - 1])
                                + f(x, ys[i], t_half);
                            let mut row = vec![zero; ny];
                            row[i - 1] = u_khalf_im1_j_arg;
                            row[i] = u_khalf_i_j_arg;
                            row[i + 1] = u_khalf_ip1_j_arg;
                            m.push(row);
                        }

                        d[ny - 1][0] = psi1(x, t_half);
                        let mut row = vec![zero; ny];
                        row[ny - 2] = -one / y_step;
                        row[ny - 1] = one / y_step;
                        m.push(row);

                        let result = m.solve_tridiagonal(&d);
                        for i in 0..ny {
                            first_temporal_layer[i][j] = result[i][0];
                        }
                    }

                    for (i, &y) in ys.iter().enumerate() {
                        first_temporal_layer[i][nx - 1] =
                            phi1(y, t_half) * x_step + first_temporal_layer[i][nx - 2];
                    }

                    // temporal layer k + 1: implicit scheme for x
                    let mut second_temporal_layer = Matrix::with_capacity(ny);
                    let t_next = ts[k + 1];

                    let mut first_row = vec![zero; nx];
                    for (j, &x) in xs.iter().enumerate() {
                        first_row[j] = psi0(x, t_next);
                    }
                    second_temporal_layer.push(first_row);

                    for i in 1..(ny - 1) {
                        let y = ys[i];
                        let mut m = Matrix::with_capacity(nx);
                        let mut d = Matrix::zero(nx, 1);

                        d[0][0] = phi0(y, t_next);
                        let mut row = vec![zero; nx];
                        row[0] = one;
                        m.push(row);

                        for j in 1..(nx - 1) {
                            d[j][0] = two * first_temporal_layer[i][j] / t_step
                                + b / y_step / y_step
                                    * (first_temporal_layer[i + 1][j]
                                        - two * first_temporal_layer[i][j]
                                        + first_temporal_layer[i - 1][j])
                                + f(xs[j], y, t_next);
                            let mut row = vec![zero; nx];
                            row[j - 1] = u_kn_i_jm1_arg;
                            row[j] = u_kn_i_j_arg;
                            row[j + 1] = u_kn_i_jp1_arg;
                            m.push(row);
                        }

                        d[nx - 1][0] = phi1(y, t_next);
                        let mut row = vec![zero; nx];
                        row[nx - 2] = -one / x_step;
                        row[nx - 1] = one / x_step;
                        m.push(row);

                        second_temporal_layer.push(m.solve_tridiagonal(&d).transposed().remove(0));
                    }

                    let mut last_row = vec![zero; nx];
                    for (j, &x) in xs.iter().enumerate() {
                        last_row[j] = psi1(x, t_next) * y_step + second_temporal_layer[ny - 2][j];
                    }
                    second_temporal_layer.push(last_row);

                    u.push(second_temporal_layer);
                }

                u
            }

            pub fn fractional_steps_method<T, F1, F2, F3, F4, F5, FF>(
                phi0: F1,
                phi1: F2,
                psi0: F3,
                psi1: F4,
                khi: F5,
                a: T,
                b: T,
                f: FF,
                x_step_count: usize,
                max_x: T,
                y_step_count: usize,
                max_y: T,
                t_step_count: usize,
                max_t: T,
            ) -> Vec<Matrix<T>>
            where
                T: Float,
                F1: Fn(T, T) -> T,
                F2: Fn(T, T) -> T,
                F3: Fn(T, T) -> T,
                F4: Fn(T, T) -> T,
                F5: Fn(T, T) -> T,
                FF: Fn(T, T, T) -> T,
            {
                let zero = T::zero();
                let one = T::one();
                let two = T::from(2.0).unwrap();

                let x_step = max_x / T::from(x_step_count).unwrap();
                let y_step = max_y / T::from(y_step_count).unwrap();
                let t_step = max_t / T::from(t_step_count).unwrap();

                let nx = x_step_count + 1;
                let ny = y_step_count + 1;
                let nt = t_step_count + 1;
                let mut xs = vec![zero; nx];
                for (i, x) in xs.iter_mut().enumerate() {
                    *x = x_step * T::from(i).unwrap();
                }
                let mut ys = vec![zero; ny];
                for (i, y) in ys.iter_mut().enumerate() {
                    *y = y_step * T::from(i).unwrap();
                }
                let mut ts = vec![zero; nt];
                for (i, t) in ts.iter_mut().enumerate() {
                    *t = t_step * T::from(i).unwrap();
                }

                let mut u = Vec::with_capacity(nt);
                u.push(Matrix::zero(ny, nx));

                for (i, y) in ys.iter().enumerate() {
                    for (j, x) in xs.iter().enumerate() {
                        u[0][i][j] = khi(*x, *y);
                    }
                }

                // temporal layer k + 1/2: explicit scheme for y
                let u_khalf_im1_j_arg = -b / y_step / y_step;
                let u_khalf_i_j_arg = two * b / y_step / y_step + one / t_step;
                let u_khalf_ip1_j_arg = -b / y_step / y_step;

                // temporal layer k + 1: explicit scheme for x;
                let u_kn_i_jm1_arg = -a / x_step / x_step;
                let u_kn_i_j_arg = two * a / x_step / x_step + one / t_step;
                let u_kn_i_jp1_arg = -a / x_step / x_step;

                // temporal layer k + 1: explicit scheme for x
                for k in 0..(nt - 1) {
                    // temporal layer k + 1/2: explicit scheme for y
                    // tl[j][i] = u^{k+1/2}_{ij}
                    let mut first_temporal_layer = Matrix::with_capacity(nx);
                    let t_half = (ts[k + 1] + ts[k]) / two;

                    // u_kn_i_0 = phi0(y, t)
                    let mut first_row = vec![zero; ny];
                    for (i, &y) in ys.iter().enumerate() {
                        first_row[i] = phi0(y, t_half);
                    }
                    first_temporal_layer.push(first_row);

                    for j in 1..(nx - 1) {
                        let x = xs[j];
                        let mut m = Matrix::with_capacity(ny);
                        let mut d = Matrix::zero(ny, 1);

                        d[0][0] = psi0(x, t_half);
                        let mut row = vec![zero; ny];
                        row[0] = one;
                        m.push(row);

                        for i in 1..(ny - 1) {
                            d[i][0] = u[k][i][j] / t_step + f(x, ys[i], ts[k]) / two;
                            let mut row = vec![zero; ny];
                            row[i - 1] = u_khalf_im1_j_arg;
                            row[i] = u_khalf_i_j_arg;
                            row[i + 1] = u_khalf_ip1_j_arg;
                            m.push(row);
                        }

                        d[ny - 1][0] = psi1(x, t_half);
                        let mut row = vec![zero; ny];
                        row[ny - 2] = -one / y_step;
                        row[ny - 1] = one / y_step;
                        m.push(row);

                        first_temporal_layer
                            .push(m.solve_tridiagonal(&d).transposed().swap_remove(0));
                    }

                    // (u_kn_i_J - u_kn_i_Jm1)/x_step = phi1(y, t)
                    let mut last_row = vec![zero; ny];
                    for (i, &y) in ys.iter().enumerate() {
                        last_row[i] = phi1(y, t_half) * x_step + first_temporal_layer[nx - 2][i];
                    }
                    first_temporal_layer.push(last_row);

                    // temporal layer k + 1: explicit scheme for x;
                    // tl[i][j] = u^{k + 1}_{ij}
                    let mut second_temporal_layer = Matrix::with_capacity(ny);
                    let t_next = ts[k + 1];

                    let mut first_row = vec![zero; nx];
                    for (j, &x) in xs.iter().enumerate() {
                        first_row[j] = psi0(x, t_next);
                    }
                    second_temporal_layer.push(first_row);

                    for i in 1..(ny - 1) {
                        let y = ys[i];
                        let mut m = Matrix::with_capacity(nx);
                        let mut d = Matrix::zero(nx, 1);

                        d[0][0] = phi0(y, t_next);
                        let mut row = vec![zero; nx];
                        row[0] = one;
                        m.push(row);

                        for j in 1..(nx - 1) {
                            d[j][0] =
                                first_temporal_layer[j][i] / t_step + f(xs[j], y, t_next) / two;
                            let mut row = vec![zero; nx];
                            row[j - 1] = u_kn_i_jm1_arg;
                            row[j] = u_kn_i_j_arg;
                            row[j + 1] = u_kn_i_jp1_arg;
                            m.push(row);
                        }

                        d[nx - 1][0] = phi1(y, t_next);
                        let mut row = vec![zero; nx];
                        row[nx - 2] = -one / x_step;
                        row[nx - 1] = one / x_step;
                        m.push(row);

                        second_temporal_layer
                            .push(m.solve_tridiagonal(&d).transposed().swap_remove(0));
                    }

                    let mut last_row = vec![zero; nx];
                    for (j, &x) in xs.iter().enumerate() {
                        last_row[j] = psi1(x, t_next) * y_step + second_temporal_layer[ny - 2][j];
                    }
                    second_temporal_layer.push(last_row);

                    u.push(second_temporal_layer);
                }

                u
            }
        }
    }
}
