use num::Float;

use crate::{error::Error, matrix::Matrix};

pub fn lagranges<T, F>(f: F, xs: &[T]) -> impl Fn(T) -> T + '_
where
    T: Float,
    F: Fn(T) -> T,
{
    let n = xs.len();
    let mut ys = Vec::with_capacity(n);

    for x in xs {
        ys.push(f(*x));
    }

    move |x: T| {
        let mut y = T::zero();
        for i in 0..n {
            let mut p = T::one();
            for j in 0..n {
                if i != j {
                    p = p * (x - xs[i]) / (xs[i] - xs[j])
                }
            }
            y = y + ys[i] * p;
        }
        y
    }
}

pub fn newtons<T, F>(f: F, xs: &[T]) -> impl Fn(T) -> T + '_
where
    T: Float,
    F: Fn(T) -> T,
{
    let n = xs.len();
    let mut divided_differences: Vec<T> = xs.iter().map(|&x| f(x)).collect();

    for i in 1..n {
        for j in (i..n).rev() {
            divided_differences[j] =
                (divided_differences[j] - divided_differences[j - 1]) / (xs[j] - xs[j - i]);
        }
    }

    move |x: T| {
        let mut y = divided_differences[n - 1];
        for i in (0..n - 1).rev() {
            y = y * (x - xs[i]) + divided_differences[i];
        }
        y
    }
}

pub fn least_squares_method<T, F>(xs: &[T], ys: &[T], polynom_degree: usize) -> impl Fn(T) -> T
where
    T: Float,
    F: Fn(T) -> T,
{
    let n = xs.len();
    assert_eq!(n, ys.len());

    let mut a: Matrix<T> = Matrix::zero(polynom_degree, polynom_degree);
    let mut b: Matrix<T> = Matrix::zero(polynom_degree, 1);
    for k in 0..polynom_degree {
        let mut pows = Vec::with_capacity(n);
        for i in 0..polynom_degree {
            pows.push(
                xs.iter()
                    .map(|&x| x.powi((k + i) as i32))
                    .fold(T::zero(), |agg, x| agg + x),
            );
        }
        a[k] = pows;

        b[k][0] = ys
            .iter()
            .zip(xs)
            .map(|(&y, &x)| y * x.powi(k as i32))
            .fold(T::zero(), |agg, bi| agg + bi);
    }

    let coefs = a.solve_lu(&b);

    move |x: T| {
        coefs
            .iter()
            .enumerate()
            .map(|(i, c)| c[0] * x.powi(i as i32))
            .fold(T::zero(), |agg, y| agg + y)
    }
}

pub fn first_derivative<T>(xs: &[T], ys: &[T], x: T) -> Result<T, Error>
where
    T: Float,
{
    let n = xs.len();
    assert_eq!(n, ys.len());
    let mut range_i = None;
    for i in 0..n - 1 {
        if xs[i] < x && x <= xs[i + 1] {
            range_i = Some(i);
            break;
        }
    }

    let Some(i) = range_i else {
        return Err(Error::NoSolutions);
    };

    Ok((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))
}

pub fn second_derivative<T>(xs: &[T], ys: &[T], x: T) -> Result<T, Error>
where
    T: Float,
{
    let n = xs.len();
    assert_eq!(n, ys.len());
    let mut range_i = None;
    for i in 0..n - 2 {
        if xs[i] < x && x <= xs[i + 1] {
            range_i = Some(i);
            break;
        }
    }

    let Some(i) = range_i else {
        return Err(Error::NoSolutions);
    };

    Ok(((ys[i + 2] - ys[i + 1]) / (xs[i + 2] - xs[i + 1])
        - (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))
        / (xs[i + 2] - xs[i]))
}

pub fn integral_rectangle<T, F>(f: F, a: T, b: T, step: T) -> T
where
    T: Float,
    F: Fn(T) -> T,
{
    assert!(a < b);
    let mut x = a;
    let halfstep = step / T::from(2).unwrap();
    let mut integral = T::zero();
    while x < b {
        integral = integral + f(x + halfstep);
        x = x + step;
    }
    step * integral
}

pub fn integral_trapezoid<T, F>(f: F, a: T, b: T, step: T) -> T
where
    T: Float,
    F: Fn(T) -> T,
{
    assert!(a < b);
    let mut x = a;
    let mut integral = T::zero();
    while x < b {
        integral = integral + (f(x) + f(x + step));
        x = x + step
    }

    T::from(0.5).unwrap() * integral * step
}

pub fn integral_simpson<T, F>(f: F, a: T, b: T, mut step: T) -> T
where
    T: Float,
    F: Fn(T) -> T,
{
    assert!(a < b);
    // make a step smaller, so number of iterations is 2*n
    while ((b - a) / step).floor() % T::from(2.0).unwrap() != T::zero() {
        step = step * T::from(0.9).unwrap();
    }

    let mut result = T::zero();
    let mut x = a + step;

    while x < b {
        result = result + f(x - step) + T::from(4.0).unwrap() * f(x) + f(x + step);
        x = x + T::from(2.0).unwrap() * step;
    }

    result * step / T::from(3.0).unwrap()
}

pub fn integral_runge_romberg<T>(
    integral_1: T,
    integral_2: T,
    step_1: T,
    step_2: T,
    remainder_power: T,
) -> T
where
    T: Float,
{
    integral_1 + (integral_1 - integral_2) / ((step_2 / step_1).powf(remainder_power) - T::one())
}
