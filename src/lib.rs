pub mod equation;
pub mod error;
pub mod function_approximation;
pub mod matrix;

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn identity_matrix() {
        let m = Matrix::identity(3);
        assert_eq!(
            m,
            Matrix::new(vec![vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.],])
        );
    }

    #[test]
    fn multiplication() {
        assert_eq!(
            &Matrix::row(vec![1., 2., 3.]) * &Matrix::column(&[4., 5., 6.]),
            Matrix::row(vec![32.])
        );
    }

    #[test]
    fn addition() {
        assert_eq!(
            Matrix::new(vec![vec![1., 2.], vec![3., 4.]])
                + Matrix::new(vec![vec![3., 7.], vec![3., 5.]]),
            Matrix::new(vec![vec![4., 9.], vec![6., 9.]])
        )
    }

    #[test]
    fn symmetry() {
        assert!(Matrix::new(vec![vec![1., 2.], vec![2., 3.]]).is_symmetrical());
        assert!(!Matrix::new(vec![vec![1., 4.], vec![2., 3.]]).is_symmetrical());
    }
}

#[cfg(test)]
mod lab_tests {
    use num::{complex::ComplexFloat, Complex};

    use crate::matrix::Matrix;

    #[test]
    fn lab_1_3() {
        let a = Matrix::new(vec![
            vec![10., 1., 1.],
            vec![2., 10., 1.],
            vec![2., 2., 10.],
        ]);

        let b = Matrix::column(&[12., 13., 14.]);

        let accuracy = 1e-2;

        let answ = Matrix::column(&[1., 1., 1.]);
        assert!(a
            .solve_jacobian(&b, accuracy, 1000)
            .unwrap()
            .eq_lossy(&answ, accuracy));
        assert!(a
            .solve_seidel(&b, accuracy, 1000)
            .unwrap()
            .eq_lossy(&answ, accuracy));
    }

    #[test]
    fn lab_1_4() {
        let a = Matrix::new(vec![vec![4., 2., 1.], vec![2., 5., 3.], vec![1., 3., 6.]]);

        let accuracy = 0.3;

        let evs = a.evs_from_symmetrical_matrix(accuracy);

        for (val, vec) in evs {
            assert!((&a * &vec).eq_lossy(&(vec * val), accuracy));
        }
    }

    #[test]
    fn lab_1_5() {
        let a = Matrix::new(vec![vec![1., 3., 1.], vec![1., 1., 4.], vec![4., 3., 1.]]);

        let (q, r) = a.get_qr();

        assert!(q.eq_lossy(
            &Matrix::new(vec![
                vec![-0.24, 0.97, 0.11],
                vec![-0.24, 0.05, -0.97],
                vec![-0.94, -0.25, 0.22],
            ]),
            0.01,
        ));

        assert!(r.eq_lossy(
            &Matrix::new(vec![
                vec![-4.24, -3.77, -2.12],
                vec![0., 2.19, 0.91],
                vec![0., 0., -3.56],
            ]),
            0.01,
        ));
    }

    #[test]
    fn lab_1_5_complex() {
        let a = Matrix::new(vec![
            vec![-1., 0., -4.],
            vec![2., -5., 0.],
            vec![0., 2., 0.],
        ]);

        let accuracy = 1e-3;

        let vals = a.eigen_values(accuracy, 1_000);
        let answ = vec![
            Complex::new(-5.61697, 0.),
            Complex::new(-0.191517, 1.67685),
            Complex::new(-0.191517, -1.67685),
        ];

        'outer: for v in &vals {
            for a in &answ {
                if (*v - *a).abs() < accuracy {
                    continue 'outer;
                }
            }
            panic!("{v} was not found in {answ:?}");
        }
    }
}

#[cfg(test)]
mod labs {
    #[cfg(feature = "plot_tests")]
    use std::process::Command;

    use num::{complex::ComplexFloat, Complex};
    use plotters::{
        prelude::*,
        style::full_palette::{ORANGE, PURPLE},
    };

    use crate::{
        equation::{
            differential::{adams_method, eulers_method, runge_kutta},
            halves_method, iterations_method, newtons_method, systems,
        },
        function_approximation::{
            first_derivative, integral_rectangle, integral_runge_romberg, integral_simpson,
            integral_trapezoid, lagranges, least_squares_method, newtons, second_derivative,
        },
        matrix::Matrix,
    };

    #[cfg(feature = "plot_tests")]
    fn open_plot(path: &str) {
        #[cfg(target_os = "windows")]
        Command::new("cmd").args(["/C", path]).spawn().unwrap();

        #[cfg(target_os = "macos")]
        Command::new("open").arg(path).spawn().unwrap();

        #[cfg(target_os = "linux")]
        Command::new("eog").arg(path).spawn().unwrap();
    }

    #[cfg(not(feature = "plot_tests"))]
    fn open_plot(_: &str) {}

    #[test]
    fn lab_1_1() {
        let a = Matrix::new(vec![
            vec![-7., 3., -4., 7.],
            vec![8., -1., -7., 6.],
            vec![9., 9., 3., -6.],
            vec![-7., -9., -8., -5.],
        ]);

        let b = Matrix::column(&[-126., 29., 27., 34.]);

        let x = a.solve_lu(&b);
        assert!((&a * &x).eq_lossy(&b, Matrix::<f64>::EPS));

        let n = a.row_count();
        let inversed = a.inversed();
        assert_eq!(&a.clone() * &inversed, Matrix::identity(n));

        let determinant = a.determinant();
        assert!((determinant - 16500f64).abs() < 1e-9);
    }

    #[test]
    fn lab_1_2() {
        let a = Matrix::new(vec![
            vec![-7., -6., 0., 0., 0.],
            vec![6., 12., 0., 0., 0.],
            vec![0., -3., 5., 0., 0.],
            vec![0., 0., -9., 21., 8.],
            vec![0., 0., 0., -5., -6.],
        ]);

        let d = Matrix::column(&[-75., 126., 13., -40., -24.]);

        let x = a.solve_tridiagonal(&d);
        assert!((&a * &x).eq_lossy(&d, Matrix::<f64>::EPS));
    }

    #[test]
    fn lab_1_3() {
        let a = Matrix::new(vec![
            vec![28., 9., -3., -7.],
            vec![-5., 21., -5., -3.],
            vec![-8., 1., -16., 5.],
            vec![0., -2., 5., 8.],
        ]);

        let b = Matrix::column(&[-159., 63., -45., 24.]);

        let accuracy = 1e-5;

        let x_jacobian = a.solve_jacobian(&b, accuracy, 100000).unwrap();
        assert!((&a * &x_jacobian).eq_lossy(&b, accuracy));

        let x_seidel = a.solve_seidel(&b, accuracy, 1000).unwrap();
        assert!((&a * &x_seidel).eq_lossy(&b, accuracy));
    }

    #[test]
    fn lab_1_4() {
        let a = Matrix::new(vec![
            vec![-7., -6., 8.],
            vec![-6., 3., -7.],
            vec![8., -7., 4.],
        ]);

        let accuracy = 1e-3;

        let evs = a.evs_from_symmetrical_matrix(accuracy);

        for (val, vec) in evs {
            assert!((&a * &vec).eq_lossy(&(vec * val), accuracy));
        }
    }

    #[test]
    fn lab_1_5() {
        let a = Matrix::new(vec![
            vec![-1., 4., -4.],
            vec![2., -5., 0.],
            vec![-8., 2., 0.],
        ]);

        let accuracy = 1e-3;

        let vals = a.eigen_values(accuracy, 1_000_000);
        let answ = vec![
            Complex::new(-8.1264, 0.),
            Complex::new(5.40491, 0.),
            Complex::new(-3.27851, 0.),
        ];

        'outer: for v in &vals {
            for a in &answ {
                if (*v - *a).abs() < accuracy {
                    continue 'outer;
                }
            }
            panic!("{v} was not found in {answ:?}");
        }
    }

    #[test]
    fn lab_2_1() {
        let f = |x: f64| x.sin() - 2. * x * x + 0.5;
        let accuracy = 0.000001;

        let x_halves = halves_method(f, (0.5, 1.), accuracy);
        assert!(
            f(x_halves).abs() <= accuracy * 2.,
            "x = {x_halves}, f(x) = {} != 0",
            f(x_halves)
        );

        let x_newton = newtons_method(f, 0.5, accuracy);
        assert!(
            f(x_newton).abs() <= accuracy * 2.,
            "x = {x_newton}, f(x) = {} != 0",
            f(x_newton)
        );

        let phi = |x: f64| ((x.sin() + 0.5) / 2.).sqrt();
        let x_iterations =
            iterations_method(phi, 0.5, accuracy).expect("iterations method did not converge");
        assert!(
            f(x_iterations).abs() <= accuracy * 2.,
            "x = {x_iterations}, f(x) = {} != 0",
            f(x_iterations)
        );
    }

    #[test]
    fn lab_2_2() {
        let fs = vec![|xs: &[f64]| xs[0] - xs[1].cos() - 1., |xs: &[f64]| {
            xs[1] - xs[0].sin() - 1.
        }];
        let f_derivatives = vec![
            |_: &[f64]| 1.,
            |xs: &[f64]| xs[1].sin(),
            |xs: &[f64]| -xs[0].cos(),
            |_: &[f64]| 1.,
        ];
        let x_approximates = vec![0.5, 1.];
        let n = fs.len();
        let accuracy = 0.001;

        let x_newton = systems::newtons_method(&fs, &f_derivatives, &x_approximates, accuracy);
        for i in 0..n {
            assert!(
                fs[i](&x_newton).abs() < 2. * accuracy,
                "f_{i}(x) = {} != 0",
                fs[i](&x_newton)
            );
        }

        let phis = vec![|xs: &[f64]| xs[1].cos() + 1., |xs: &[f64]| xs[0].sin() + 1.];
        let phi_derivatives = vec![
            |_: &[f64]| 0.,
            |xs: &[f64]| -xs[1].sin(),
            |xs: &[f64]| xs[0].cos(),
            |_: &[f64]| 0.,
        ];

        let x_iterations =
            systems::iterations_method(&phis, &phi_derivatives, &x_approximates, accuracy)
                .expect("iterations method did not converge");

        for i in 0..n {
            assert!(
                fs[i](&x_iterations).abs() < 2. * accuracy,
                "f_{i}(x) = {} != 0",
                fs[i](&x_iterations)
            );
        }
    }

    #[test]
    fn lab_3_1() {
        let y = |x: f64| x.atan();
        let x_i = vec![-3., -1., 1., 3.];
        let x = -0.5;
        let y_lagranges = lagranges(y, &x_i);
        println!(
            "\nlagrange: y({x:.3}) = {:.3}, L({x:.3}) = {:.3}; the difference is {:.3}",
            y(x),
            y_lagranges(x),
            y(x) - y_lagranges(x)
        );

        let x_i = vec![-3., 0., 1., 3.];
        let y_newtons = newtons(y, &x_i);
        println!(
            "newton: y({x:.3}) = {:.3}, N({x:.3}) = {:.3}; the difference is {:.3}",
            y(x),
            y_newtons(x),
            y(x) - y_newtons(x)
        );
    }

    #[test]
    fn lab_3_2() {
        // free
    }

    #[test]
    fn lab_3_3() {
        let xs = vec![-5.0, -3.0, -1.0, 1.0, 3.0, 5.0];
        let ys = vec![-1.3734, -1.249, -0.7854, 0.7854, 1.249, 1.3734];

        let max_degree = 6;
        let fs: Vec<Box<dyn Fn(f64) -> f64>> = (1..=max_degree)
            .map(|degree| {
                let func = least_squares_method::<f64, Box<dyn Fn(f64) -> f64>>(&xs, &ys, degree);
                Box::new(func) as Box<dyn Fn(f64) -> f64>
            })
            .collect();

        let path = "plots/lab3.3.png";
        let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("lab 3.3 - least squares method", ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-5.5f64..5.5f64, -2.5f64..2.5f64)
            .unwrap();
        chart.configure_mesh().draw().unwrap();

        let colors = vec![BLUE, MAGENTA, ORANGE, CYAN, PURPLE, RED];
        let step = 0.01;
        println!("{}", fs.len());
        assert!(max_degree <= colors.len(), "add more colors!");
        for (f, c) in fs.iter().zip(colors) {
            chart
                .draw_series(LineSeries::new(
                    (-500..=500).map(|x| x as f64 * step).map(|x| (x, f(x))),
                    &c,
                ))
                .unwrap();
        }

        chart
            .draw_series(PointSeries::of_element(
                xs.into_iter().zip(ys),
                3f64,
                &RED,
                &|c, s, st| {
                    return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
                },
            ))
            .unwrap();

        open_plot(path);
    }

    #[test]
    fn lab_3_4() {
        let x = 1.;
        let xs = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let ys = vec![0.0, 0.97943, 1.8415, 2.4975, 2.9093];

        let dy = first_derivative(&xs, &ys, x).unwrap();
        println!("\ndy/dx({x}) = {dy}");
        let d2y = second_derivative(&xs, &ys, x).unwrap();
        println!("d2y/dx2({x}) = {d2y}");

        let n = xs.len();

        let degree = 3;
        let f = least_squares_method::<f64, Box<dyn Fn(f64) -> f64>>(&xs, &ys, degree);
        let df = least_squares_method::<f64, Box<dyn Fn(f64) -> f64>>(
            &xs[0..n - 1],
            &xs[0..n - 1]
                .iter()
                .map(|&x| first_derivative(&xs, &ys, x + 0.25).unwrap())
                .collect::<Vec<f64>>(),
            degree,
        );
        let d2f = least_squares_method::<f64, Box<dyn Fn(f64) -> f64>>(
            &xs[0..n - 2],
            &xs[0..n - 2]
                .iter()
                .map(|&x| second_derivative(&xs, &ys, x + 0.25).unwrap())
                .collect::<Vec<f64>>(),
            degree,
        );

        let step = 0.01;
        let path = "plots/lab3.4.png";
        let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("lab 3.4 - derivatives", ("sans-serif", 24))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-0.5f64..2.5f64, -1f64..3.5f64)
            .unwrap();
        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(
                (-0..=200).map(|x| x as f64 * step).map(|x| (x, f(x))),
                &RED,
            ))
            .unwrap();

        chart
            .draw_series(LineSeries::new(
                (-0..=200).map(|x| x as f64 * step).map(|x| (x, df(x))),
                &BLUE,
            ))
            .unwrap();

        chart
            .draw_series(LineSeries::new(
                (-0..=200).map(|x| x as f64 * step).map(|x| (x, d2f(x))),
                &GREEN,
            ))
            .unwrap();

        chart
            .draw_series(PointSeries::of_element(
                xs.into_iter().zip(ys),
                3f64,
                &RED,
                &|c, s, st| {
                    return EmptyElement::at(c) + Circle::new((0, 0), s, st.filled());
                },
            ))
            .unwrap();

        open_plot(path);
    }

    #[test]
    fn lab_3_5() {
        let f = |x: f64| x * x / (x * x + 16.);
        let (a, b) = (0., 2.);
        let (h1, h2) = (0.5, 0.25);

        let ip1 = integral_rectangle(f, a, b, h1);
        let ip2 = integral_rectangle(f, a, b, h2);
        let it1 = integral_trapezoid(f, a, b, h1);
        let it2 = integral_trapezoid(f, a, b, h2);
        let is1 = integral_simpson(f, a, b, h1);
        let is2 = integral_simpson(f, a, b, h2);

        let iprr = integral_runge_romberg(ip1, ip2, h1, h2, 2.);
        let itrr = integral_runge_romberg(it1, it2, h1, h2, 2.);
        let isrr = integral_runge_romberg(is1, is2, h1, h2, 2.);

        println!("\nstep      | rect      | trap      | simp");
        println!("{h1:.7} | {ip1:.7} | {it1:.7} | {is1:.7}");
        println!("{h2:.7} | {ip2:.7} | {it2:.7} | {is2:.7}");
        println!("runge     | {iprr:.7} | {itrr:.7} | {isrr:.7}");
    }

    #[test]
    fn lab_4_1() {
        let f = |x: f64, y: &Vec<f64>| vec![y[1], -y[1] * x.tan() - y[0] * x.cos().powi(2)];
        let y_true = |x: f64| x.sin().cos() + x.cos().sin();
        let x_range = (0.0, 1.0);
        let y0 = vec![y_true(x_range.0), 0.];
        let h = 0.1;

        println!();
        let mut e = 0.;
        let sol = eulers_method(f, x_range, y0.clone(), h);
        for (x, y_vec) in &sol[1..] {
            let y = y_vec[0];
            let exact = y_true(*x);
            e += (y - exact).abs();
        }
        println!("euler error: {e:.4}");

        let mut e = 0.;
        let sol = runge_kutta(f, x_range, y0.clone(), h);
        for (x, y_vec) in sol {
            let y = y_vec[0];
            let exact = (x.sin()).cos() + (x.cos()).sin();
            e += (y - exact).abs();
        }
        println!("runge-kutta error: {e:.4}");

        let mut e = 0.;
        let sol = adams_method(f, x_range, y0.clone(), h);
        for (x, y_vec) in sol {
            let y = y_vec[0];
            let exact = (x.sin()).cos() + (x.cos()).sin();
            e += (y - exact).abs();
        }
        println!("adams error: {e:.4}");
    }

    #[test]
    fn lab_4_2() {
        let ode =
            |x: f64, y: &Vec<f64>| vec![y[1], ((2.0 * x + 1.0) * y[1] - (x + 1.0) * y[0]) / x];
        let shoot = |a: f64| {
            let x_range = (1.0, 2.0);
            let h = 0.1;
            let y0 = vec![a, 3.0 * f64::exp(1.0)];
            let sol = runge_kutta(ode, x_range, y0, h);
            let last = &sol.last().unwrap().1;
            last[1] - 2.0 * last[0]
        };
        let a0 = f64::exp(1.0);
        let accuracy = 1e-3;

        let a = newtons_method(shoot, a0, accuracy);
        let x_range = (1.0, 2.0);
        let h = 0.1;
        let y0 = vec![a, 3.0 * f64::exp(1.0)];

        let sol = runge_kutta(ode, x_range, y0, h);
        for (x, y_vec) in sol {
            let y = y_vec[0];
            let exact = x.exp() * x * x;
            assert!(
                (y - exact).abs() < accuracy * 2.,
                "lab_4_2: at x={x:.2}, y_num={y:.5}, y_exact={exact:.5}"
            );
        }
    }
}
