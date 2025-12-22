pub mod equation;
pub mod error;
pub mod function_approximation;
pub mod matrix;

#[cfg(feature = "plot_tests")]
const GRAPH_WIDTH: u32 = 720;
#[cfg(feature = "plot_tests")]
const GRAPH_HEIGHT: u32 = 480;
#[cfg(feature = "plot_tests")]
const GRAPH_SIZE: (u32, u32) = (GRAPH_WIDTH, GRAPH_HEIGHT);

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
    use crate::GRAPH_SIZE;
    #[cfg(feature = "plot_tests")]
    use plotters::{
        prelude::*,
        style::full_palette::{ORANGE, PURPLE, TEAL},
    };
    #[cfg(feature = "plot_tests")]
    use std::process::Command;

    use num::{complex::ComplexFloat, Complex};

    #[cfg(feature = "plot_tests")]
    use crate::function_approximation::{
        first_derivative, least_squares_method, second_derivative,
    };

    use crate::{
        equation::{
            differential::{
                adams_method, elliptic, eulers_method,
                hyperbolic::dirichlet,
                parabolic::robin,
                runge_kutta,
                two_dimensional::parabolic::{fractional_steps_method, variable_direction_method},
            },
            halves_method, iterations_method, newtons_method, systems,
        },
        function_approximation::{
            integral_rectangle, integral_runge_romberg, integral_simpson, integral_trapezoid,
            lagranges, newtons,
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

    #[cfg(feature = "plot_tests")]
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
        let root = BitMapBackend::new(path, GRAPH_SIZE).into_drawing_area();
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

        root.present().unwrap();

        open_plot(path);
    }

    #[test]
    #[cfg(feature = "plot_tests")]
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
        let root = BitMapBackend::new(path, GRAPH_SIZE).into_drawing_area();
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

        root.present().unwrap();

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
            e += (y - exact).powi(2);
        }
        println!("euler error: {:.4}", e.sqrt());

        let mut e = 0.;
        let sol = runge_kutta(f, x_range, y0.clone(), h);
        for (x, y_vec) in sol {
            let y = y_vec[0];
            let exact = (x.sin()).cos() + (x.cos()).sin();
            e += (y - exact).powi(2);
        }
        println!("runge-kutta error: {:.4}", e.sqrt());

        let mut e = 0.;
        let sol = adams_method(f, x_range, y0.clone(), h);
        for (x, y_vec) in sol {
            let y = y_vec[0];
            let exact = (x.sin()).cos() + (x.cos()).sin();
            e += (y - exact).powi(2);
        }
        println!("adams error: {:.4}", e.sqrt());
    }

    #[test]
    fn lab_4_2() {
        let f = |x: f64, y: &Vec<f64>| vec![y[1], ((2.0 * x + 1.0) * y[1] - (x + 1.0) * y[0]) / x];
        let shoot = |a: f64| {
            let x_range = (1.0, 2.0);
            let h = 0.1;
            let y0 = vec![a, 3.0 * f64::exp(1.0)];
            let sol = runge_kutta(f, x_range, y0, h);
            let last = &sol.last().unwrap().1;
            last[1] - 2.0 * last[0]
        };
        let a0 = f64::exp(1.0);
        let accuracy = 1e-3;

        let a = newtons_method(shoot, a0, accuracy);
        let x_range = (1.0, 2.0);
        let h = 0.1;
        let y0 = vec![a, 3.0 * f64::exp(1.0)];

        let sol = runge_kutta(f, x_range, y0, h);
        for (x, y_vec) in sol {
            let y = y_vec[0];
            let exact = x.exp() * x * x;
            assert!(
                (y - exact).powi(2) < accuracy * 2.,
                "lab_4_2: at x={x:.2}, y_num={y:.5}, y_exact={exact:.5}"
            );
        }
    }

    fn lab_5_get_err<T>(
        x_step: T,
        t: T,
        solution: &Vec<T>,
        analytical: impl Fn(T, T) -> T,
    ) -> (T, T)
    where
        T: num::Float,
    {
        let nx = solution.len();
        let mut l2 = T::zero();
        let mut max_err = T::zero();
        for j in 0..nx {
            let x = T::from(j).unwrap() * x_step;

            let numerical = solution[j];
            let exact = analytical(x, t);

            let diff = (numerical - exact).abs();
            l2 = l2 + diff * diff;
            if diff > max_err {
                max_err = diff;
            }
        }
        l2 = (l2 / T::from(nx).unwrap()).sqrt();
        (l2, max_err)
    }

    #[test]
    fn lab_5() {
        println!("");
        let a = 1.5_f64;
        let b = 0.5_f64;
        let c = -1.0_f64;

        let max_x = std::f64::consts::PI;
        let max_t = 1.5_f64;

        let x_step_count = 100;
        let t_step_count = 100000;

        let ts = vec![max_t / 2., max_t];

        let analytical = |x: f64, t: f64| ((c - a) * t).exp() * (x + b * t).sin();

        let phi0 = |t: f64| ((c - a) * t).exp() * ((b * t).cos() + (b * t).sin());
        let phi1 = |t: f64| -((c - a) * t).exp() * ((b * t).cos() + (b * t).sin());
        let psi = |x: f64| x.sin();

        let t_step = max_t / t_step_count as f64;
        let x_step = max_x / x_step_count as f64;

        assert!(t_step / (x_step * x_step) < 1. / 2.);

        let mut solutions = Vec::with_capacity(5);
        let explicit_robin = robin::pefd(
            phi0,
            1.,
            1.,
            phi1,
            1.,
            1.,
            psi,
            a,
            b,
            c,
            t_step_count,
            max_t,
            x_step_count,
            max_x,
        );
        let implicit_robin = robin::pifd(
            phi0,
            1.,
            1.,
            phi1,
            1.,
            1.,
            psi,
            a,
            b,
            c,
            t_step_count,
            max_t,
            x_step_count,
            max_x,
        );
        let crank_nicolsons_robin = robin::crank_nicolsons(
            phi0,
            1.,
            1.,
            phi1,
            1.,
            1.,
            psi,
            a,
            b,
            c,
            t_step_count,
            max_t,
            x_step_count,
            max_x,
        );

        solutions.push(("EFDS Robin", explicit_robin));
        solutions.push(("IFDS Robin", implicit_robin));
        solutions.push(("CNS Robin", crank_nicolsons_robin));

        for (name, solution) in &solutions {
            for &t in &ts {
                let idx = (t / t_step).round() as usize;
                let slice = &solution[idx];
                let (l2, max) = lab_5_get_err(x_step, t, slice, analytical);
                println!("{name} t = {t:.3}: L2 error = {l2:.3}, max error = {max:.3}");
            }
        }

        #[cfg(feature = "plot_tests")]
        {
            let colors = vec![RED, BLUE, GREEN, TEAL, PURPLE, ORANGE];

            let render_t_step = 0.05f64;
            let ts = (0..(max_t / render_t_step) as usize)
                .map(|t| t as f64 * render_t_step)
                .collect::<Vec<_>>();
            let path = "plots/lab5.gif";
            let root = BitMapBackend::gif(path, GRAPH_SIZE, (5_000f64 * render_t_step) as u32)
                .unwrap()
                .into_drawing_area();
            root.fill(&WHITE).unwrap();

            for t in ts {
                let idx = (t / t_step).round() as usize;

                root.fill(&WHITE).unwrap();
                let mut chart = ChartBuilder::on(&root)
                    .caption(
                        format!("lab 5: finite-difference schemes, t={t:.3}"),
                        ("sans-serif", 24),
                    )
                    .margin(10)
                    .x_label_area_size(30)
                    .y_label_area_size(30)
                    .set_left_and_bottom_label_area_size(20)
                    .build_cartesian_2d(-0.25f64..max_x + 0.25, -0.25f64..1.0f64)
                    .unwrap();

                chart.configure_mesh().draw().unwrap();

                let color = RGBColor(100, 100, 100);
                chart
                    .draw_series(LineSeries::new(
                        (0..=(max_x * 10_000.) as usize)
                            .map(|j| j as f64 / 10_000.)
                            .map(|x| (x, analytical(x, t))),
                        &color,
                    ))
                    .unwrap()
                    .label("Analytical")
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y - 2), (x + 20, y + 2)], color)
                    });

                for ((name, solution), color) in solutions.iter().zip(&colors) {
                    let slice = &solution[idx];
                    chart
                        .draw_series(LineSeries::new(
                            (0..slice.len()).map(|j| (j as f64 * x_step, slice[j])),
                            &color,
                        ))
                        .unwrap()
                        .label(name.to_string())
                        .legend(move |(x, y)| {
                            PathElement::new(vec![(x, y - 2), (x + 20, y + 2)], color)
                        });
                }
                chart.configure_series_labels().draw().unwrap();
                root.present().unwrap();
            }

            open_plot(path);
        }
    }

    #[test]
    fn lab_6() {
        println!();
        let alpha = 1.0;
        let beta = 3.0;
        let a = 1.0;
        let b = 1.0;
        let c = -1.0;
        let d = |x: f64, t: f64| -x.cos() * (-t).exp();
        let phi0 = |t: f64| (-t).exp();
        let phi1 = |t: f64| -(-t).exp();
        let psi1 = |x: f64| x.sin();
        let psi2 = |x: f64| -x.sin();

        let max_x = std::f64::consts::PI;
        let max_t = 3.0;

        let x_step_count = 100;
        let t_step_count = 100000;

        let analytical = |x: f64, t: f64| x.sin() * (-t).exp();

        let ts = vec![max_t / 2., max_t];

        let t_step = max_t / t_step_count as f64;
        let x_step = max_x / x_step_count as f64;

        assert!(a * t_step / (x_step * x_step) < 1.);

        let mut solutions = Vec::with_capacity(2);
        let explicit = dirichlet::pefd(
            phi0,
            phi1,
            psi1,
            psi2,
            a,
            b,
            c,
            d,
            alpha,
            beta,
            t_step_count,
            max_t,
            x_step_count,
            max_x,
        );
        let implicit = dirichlet::pifd(
            phi0,
            phi1,
            psi1,
            psi2,
            a,
            b,
            c,
            d,
            alpha,
            beta,
            t_step_count,
            max_t,
            x_step_count,
            max_x,
        );

        solutions.push(("EFDS", explicit));
        solutions.push(("IFDS", implicit));

        #[cfg(feature = "plot_tests")]
        let mut l2_errors = Vec::new();

        for (name, solution) in &solutions {
            for &t in &ts {
                let idx = (t / t_step).round() as usize;
                let slice = &solution[idx];
                let (l2, max) = lab_5_get_err(x_step, t, slice, analytical);
                println!("{name} t = {t:.3}: L2 error = {l2:.3}, max error = {max:.3}");
            }

            #[cfg(feature = "plot_tests")]
            {
                let errors: Vec<f64> = (0..=t_step_count)
                    .map(|i| {
                        let t = i as f64 * t_step;
                        let slice = &solution[i];
                        let (l2, _) = lab_5_get_err(x_step, t, slice, analytical);
                        l2
                    })
                    .collect();
                l2_errors.push((name, errors));
            }
        }

        #[cfg(feature = "plot_tests")]
        {
            let colors = vec![RED, BLUE, GREEN, TEAL, PURPLE, ORANGE];
            let l2_error_path = "plots/lab6_l2_error.png";
            let root = BitMapBackend::new(l2_error_path, GRAPH_SIZE).into_drawing_area();
            root.fill(&WHITE).unwrap();

            let max_error = l2_errors
                .iter()
                .flat_map(|(_, errors)| errors.iter())
                .fold(0.0f64, |a, &b| a.max(b));

            let mut chart = ChartBuilder::on(&root)
                .caption("Lab 6: L2 error over time", ("sans-serif", 24))
                .margin(40)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .set_left_and_bottom_label_area_size(20)
                .build_cartesian_2d(
                    0.0 - max_t * 0.1..max_t * 1.1,
                    0.0 - max_error * 0.1..max_error * 1.1,
                )
                .unwrap();

            chart.configure_mesh().draw().unwrap();

            for ((name, errors), color) in l2_errors.iter().zip(&colors) {
                chart
                    .draw_series(LineSeries::new(
                        errors
                            .iter()
                            .enumerate()
                            .map(|(i, &err)| (i as f64 * t_step, err)),
                        &color,
                    ))
                    .unwrap()
                    .label(format!("{}", name))
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y - 2), (x + 20, y + 2)], color)
                    });
            }
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()
                .unwrap();
            root.present().unwrap();
            open_plot(l2_error_path);

            let render_t_step = 0.05f64;
            let ts = (0..(max_t / render_t_step) as usize)
                .map(|t| t as f64 * render_t_step)
                .collect::<Vec<_>>();
            let path = "plots/lab6.gif";
            let root = BitMapBackend::gif(path, GRAPH_SIZE, (5_000f64 * render_t_step) as u32)
                .unwrap()
                .into_drawing_area();
            root.fill(&WHITE).unwrap();

            for t in ts {
                let idx = (t / t_step).round() as usize;

                root.fill(&WHITE).unwrap();
                let mut chart = ChartBuilder::on(&root)
                    .caption(
                        format!("lab 6: finite-difference schemes, t={t:.3}"),
                        ("sans-serif", 24),
                    )
                    .margin(10)
                    .x_label_area_size(30)
                    .y_label_area_size(30)
                    .set_left_and_bottom_label_area_size(20)
                    .build_cartesian_2d(-0.25f64..max_x + 0.25, -0.25f64..1.0f64)
                    .unwrap();

                chart.configure_mesh().draw().unwrap();

                let color = RGBColor(100, 100, 100);
                chart
                    .draw_series(LineSeries::new(
                        (0..=(max_x * 10_000.) as usize)
                            .map(|j| j as f64 / 10_000.)
                            .map(|x| (x, analytical(x, t))),
                        &color,
                    ))
                    .unwrap()
                    .label("Analytical")
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y - 2), (x + 20, y + 2)], color)
                    });

                for ((name, solution), color) in solutions.iter().zip(&colors) {
                    let slice = &solution[idx];
                    chart
                        .draw_series(LineSeries::new(
                            (0..slice.len()).map(|j| (j as f64 * x_step, slice[j])),
                            &color,
                        ))
                        .unwrap()
                        .label(name.to_string())
                        .legend(move |(x, y)| {
                            PathElement::new(vec![(x, y - 2), (x + 20, y + 2)], color)
                        });
                }
                chart
                    .configure_series_labels()
                    .background_style(&WHITE.mix(0.8))
                    .border_style(&BLACK)
                    .draw()
                    .unwrap();
                root.present().unwrap();
            }

            open_plot(path);
        }
    }

    fn lab_7_get_err<T>(
        x_step: T,
        y_step: T,
        solution: &Matrix<T>,
        analytical: impl Fn(T, T) -> T,
    ) -> (T, T)
    where
        T: num::Float,
    {
        let ny = solution.len();
        let nx = solution[0].len();
        let mut l2 = T::zero();
        let mut max_err = T::zero();
        for i in 0..ny {
            let y = T::from(i).unwrap() * y_step;
            for j in 0..nx {
                let x = T::from(j).unwrap() * x_step;

                let numerical = solution[i][j];
                let exact = analytical(x, y);

                let diff = (numerical - exact).abs();
                l2 = l2 + diff * diff;
                if diff > max_err {
                    max_err = diff;
                }
            }
        }
        l2 = (l2 / T::from(nx).unwrap()).sqrt();
        (l2, max_err)
    }

    #[test]
    fn lab_7() {
        #[cfg(feature = "plot_tests")]
        const DRAW_ITERS: usize = 50;
        println!();
        let alpha = 1.0;
        let beta = 1.0;
        let a = -2.0;
        let b = -2.0;
        let c = 4.0;
        let phi0 = |y: f64| (-y).exp() * y.cos();
        let phi1 = |_y: f64| 0.;
        let psi0 = |x: f64| (-x).exp() * x.cos();
        let psi1 = |_x: f64| 0.;

        let x_step_count = 100;
        let y_step_count = 100;

        let max_x = std::f64::consts::PI / 2.;
        let max_y = std::f64::consts::PI / 2.;
        let eps = 1e-9;
        let relax_slope = 4.;
        let relax_amount = 1e4;
        let max_iter = 1000;
        let max_iter_seidel = 1000;

        let analytical = |x: f64, y: f64| (-x - y).exp() * x.cos() * y.cos();

        let iter_solution = elliptic::dirichlet::iterative_method(
            phi0,
            phi1,
            psi0,
            psi1,
            a,
            b,
            c,
            alpha,
            beta,
            x_step_count,
            max_x,
            y_step_count,
            max_y,
            eps,
            max_iter,
            relax_slope,
            relax_amount,
        );

        let seidel_solution = elliptic::dirichlet::seidel(
            phi0,
            phi1,
            psi0,
            psi1,
            a,
            b,
            c,
            alpha,
            beta,
            x_step_count,
            max_x,
            y_step_count,
            max_y,
            eps,
            max_iter_seidel,
            relax_slope,
            relax_amount,
        );

        let solutions = vec![("Iterative", iter_solution), ("Seidel's", seidel_solution)];

        let x_step = max_x / x_step_count as f64;
        let y_step = max_y / y_step_count as f64;

        for (name, solution) in &solutions {
            let (l2, max_err) = lab_7_get_err(x_step, y_step, solution, analytical);
            println!("{name}: l2 error = {l2}, max error = {max_err}");
        }

        #[cfg(feature = "plot_tests")]
        {
            use crate::{GRAPH_HEIGHT, GRAPH_WIDTH};
            let graph_width: u32 = GRAPH_WIDTH;
            let graph_height: u32 = GRAPH_HEIGHT * 2;
            let path = "plots/lab7.gif";

            let root = BitMapBackend::gif(path, (graph_width, graph_height), 60)
                .unwrap()
                .into_drawing_area();

            let errors: Vec<_> = solutions
                .iter()
                .map(|(_, s)| {
                    let mut max_error = std::f64::NEG_INFINITY;
                    let mut min_error = std::f64::INFINITY;
                    for yi in 0..y_step_count {
                        for xi in 0..x_step_count {
                            let x = xi as f64 * max_x / (x_step_count - 1) as f64;
                            let y = yi as f64 * max_y / (y_step_count - 1) as f64;
                            let error = s[yi][xi] - analytical(x, y);
                            if error > max_error {
                                max_error = error;
                            }
                            if error < min_error {
                                min_error = error;
                            }
                        }
                    }
                    (min_error, max_error)
                })
                .collect();

            let f_analytical = |x: f64, y: f64| analytical(x, y);

            let x_points = (0..=50).map(|i| i as f64 * max_x / 50.0);
            let y_points = (0..=50).map(|i| i as f64 * max_y / 50.0);

            for pitch in 0..DRAW_ITERS {
                if (pitch + 1) % 10 == 0 {
                    println!("drawn at iter = {}", pitch + 1);
                }
                root.fill(&WHITE).unwrap();

                let (subplot_analytical, mut subplot_solutions) =
                    root.split_vertically(graph_height / (1 + solutions.len()) as u32);

                let analytical_min = 0.;
                let analytical_max = 1.;
                let mut chart_analytical = ChartBuilder::on(&subplot_analytical)
                    .caption("Analytical Solution", ("sans-serif", 20))
                    .build_cartesian_3d(0.0..max_x, analytical_min..analytical_max, 0.0..max_y)
                    .unwrap();

                chart_analytical.with_projection(|mut p| {
                    p.pitch = (pitch as f64 / 10.0).sin() / 3. + 0.333;
                    p.scale = 0.7;
                    p.yaw = pitch as f64 / 20.;
                    p.into_matrix()
                });

                chart_analytical.configure_axes().draw().unwrap();

                chart_analytical
                    .draw_series(
                        SurfaceSeries::xoz(x_points.clone(), y_points.clone(), f_analytical)
                            .style_func(&|&v| {
                                let c = VulcanoHSL::get_color(v / 5.0);
                                RGBAColor(c.rgb().0, c.rgb().1, c.rgb().2, 0.5).into()
                            }),
                    )
                    .unwrap();

                for (i, (name, solution)) in solutions.iter().enumerate() {
                    let current_subplot = if i == solutions.len() - 1 {
                        subplot_solutions.clone()
                    } else {
                        let (upper, lower) = subplot_solutions
                            .split_vertically(graph_height / (1 + solutions.len()) as u32);
                        subplot_solutions = lower;
                        upper
                    };

                    let areas = current_subplot.split_evenly((1, 2));

                    let f = |x: f64, y: f64| {
                        let xi = ((x / max_x) * (x_step_count - 1) as f64).round() as usize;
                        let yi = ((y / max_y) * (y_step_count - 1) as f64).round() as usize;
                        solution[yi][xi]
                    };
                    let f_error = |x: f64, y: f64| {
                        let xi = ((x / max_x) * (x_step_count - 1) as f64).round() as usize;
                        let yi = ((y / max_y) * (y_step_count - 1) as f64).round() as usize;
                        solution[yi][xi] - analytical(x, y)
                    };
                    let (min_num, max_num) = (
                        solution
                            .iter()
                            .flatten()
                            .fold(f64::INFINITY, |a, &b| a.min(b)),
                        solution
                            .iter()
                            .flatten()
                            .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                    );
                    let (min_err, max_err) = errors[i];

                    let mut chart_solution = ChartBuilder::on(&areas[0])
                        .caption(format!("Numerical Solution: {name}"), ("sans-serif", 20))
                        .build_cartesian_3d(0.0..max_x, min_num..max_num, 0.0..max_y)
                        .unwrap();

                    chart_solution.with_projection(|mut p| {
                        p.pitch = (pitch as f64 / 10.0).sin() / 3. + 0.333;
                        p.scale = 0.7;
                        p.yaw = pitch as f64 / 20.;
                        p.into_matrix()
                    });

                    chart_solution.configure_axes().draw().unwrap();

                    chart_solution
                        .draw_series(
                            SurfaceSeries::xoz(x_points.clone(), y_points.clone(), f).style_func(
                                &|&v| {
                                    let c = VulcanoHSL::get_color(v / 5.0);
                                    RGBAColor(c.rgb().0, c.rgb().1, c.rgb().2, 0.5).into()
                                },
                            ),
                        )
                        .unwrap();

                    let mut chart_error = ChartBuilder::on(&areas[1])
                        .caption(
                            format!("Error (max = {:.2e})", min_err.abs().max(max_err)),
                            ("sans-serif", 20),
                        )
                        .build_cartesian_3d(0.0..max_x, min_err..max_err, 0.0..max_y)
                        .unwrap();

                    chart_error.with_projection(|mut p| {
                        p.pitch = (pitch as f64 / 10.0).sin() / 3. + 0.333;
                        p.scale = 0.7;
                        p.yaw = pitch as f64 / 20.;
                        p.into_matrix()
                    });

                    chart_error.configure_axes().draw().unwrap();

                    let x_points = (0..=100).map(|i| i as f64 * max_x / 100.0);
                    let y_points = (0..=100).map(|i| i as f64 * max_y / 100.0);

                    chart_error
                        .draw_series(
                            SurfaceSeries::xoz(x_points.clone(), y_points.clone(), f_error)
                                .style_func(&|&v| {
                                    let normalized = (v - min_err) / (max_err - min_err);
                                    let c = VulcanoHSL::get_color(normalized);
                                    RGBAColor(c.rgb().0, c.rgb().1, c.rgb().2, 0.5).into()
                                }),
                        )
                        .unwrap();
                }

                root.present().unwrap();
            }

            root.present().unwrap();
            open_plot(path);
        }
    }

    #[test]
    fn lab_8() {
        #[cfg(feature = "plot_tests")]
        const DRAW_ITERS: usize = 40;
        println!();
        struct TestSuite {
            a: f64,
            b: f64,
            mu: f64,
        }

        let max_x = std::f64::consts::PI;
        let max_y = std::f64::consts::PI;
        let max_t = std::f64::consts::PI;

        let x_step_count = 50;
        let y_step_count = 50;
        let t_step_count = 100;

        let test_suite = vec![TestSuite {
            a: 1.,
            b: 1.,
            mu: 1.,
        }];

        let ts = vec![max_t / 4., max_t / 2., max_t];
        let test_suite_length = test_suite.len();
        for (i, test) in test_suite.into_iter().enumerate() {
            let TestSuite { a, b, mu } = test;
            let f = |x: f64, y: f64, t: f64| {
                x.sin() * y.sin() * (mu * (mu * t).cos() + (a + b) * (mu * t).sin())
            };

            let phi0 = |_y: f64, _t: f64| 0.;
            let phi1 = |y: f64, t: f64| -y.sin() * (mu * t).sin();
            let psi0 = |_x: f64, _t: f64| 0.;
            let psi1 = |x: f64, t: f64| -x.sin() * (mu * t).sin();
            let khi = |_x: f64, _y: f64| 0.;

            let analytical = |x: f64, y: f64, t: f64| x.sin() * y.sin() * (mu * t).sin();

            let now = std::time::Instant::now();
            let vdm = variable_direction_method(
                phi0,
                phi1,
                psi0,
                psi1,
                khi,
                a,
                b,
                f,
                x_step_count,
                max_x,
                y_step_count,
                max_y,
                t_step_count,
                max_t,
            );
            println!("variable direction method took {:?} with parameters:\n\tx_steps = {x_step_count}\n\ty_steps = {y_step_count}\n\tt_steps = {t_step_count}\n\tmax_t = {max_t:.5}", now.elapsed());
            let now = std::time::Instant::now();
            let fsm = fractional_steps_method(
                phi0,
                phi1,
                psi0,
                psi1,
                khi,
                a,
                b,
                f,
                x_step_count,
                max_x,
                y_step_count,
                max_y,
                t_step_count,
                max_t,
            );
            println!("fractional steps method took {:?} with parameters:\n\tx_steps = {x_step_count}\n\ty_steps = {y_step_count}\n\tt_steps = {t_step_count}\n\tmax_t = {max_t:.5}", now.elapsed());

            let solutions = vec![("VDM", vdm), ("FSM", fsm)];

            for (name, solution) in &solutions {
                for &t in &ts {
                    let iter = ((t / max_t) * (t_step_count as f64)) as usize;
                    let mut max_error = std::f64::NEG_INFINITY;
                    for yi in 0..y_step_count {
                        for xi in 0..x_step_count {
                            let x = xi as f64 * max_x / (x_step_count - 1) as f64;
                            let y = yi as f64 * max_y / (y_step_count - 1) as f64;
                            let error = (solution[iter][yi][xi] - analytical(x, y, t)).abs();
                            if error > max_error {
                                max_error = error;
                            }
                        }
                    }

                    println!("{name} t = {t:.3}, iter = {iter}: max_error = {max_error:.3e}");
                }
            }

            if i != test_suite_length - 1 {
                continue;
            }

            #[cfg(feature = "plot_tests")]
            {
                use crate::{GRAPH_HEIGHT, GRAPH_WIDTH};
                let graph_width = GRAPH_WIDTH;
                let graph_height = GRAPH_HEIGHT * 2;
                let path = "plots/lab8.gif";

                let root = BitMapBackend::gif(path, (graph_width, graph_height), 50)
                    .unwrap()
                    .into_drawing_area();

                let x_points = (0..=50).map(|i| i as f64 * max_x / 50.0);
                let y_points = (0..=50).map(|i| i as f64 * max_y / 50.0);

                for iter in 0..=DRAW_ITERS {
                    if iter % 10 == 0 {
                        println!("drawing at iter = {}", iter);
                    }
                    root.fill(&WHITE).unwrap();

                    let t = iter as f64 / DRAW_ITERS as f64 * max_t;
                    let num_iter = ((t / max_t) * (t_step_count as f64)) as usize;

                    let analytical_min = -0.9;
                    let analytical_max = 0.9;
                    let f_analytical = |x: f64, y: f64| analytical(x, y, t);
                    let (subplot_analytical, mut subplot_solutions) =
                        root.split_vertically(graph_height / (1 + solutions.len()) as u32);

                    let mut chart_analytical = ChartBuilder::on(&subplot_analytical)
                        .caption(format!("Analytical at t = {t:.3}"), ("sans-serif", 20))
                        .build_cartesian_3d(0.0..max_x, analytical_min..analytical_max, 0.0..max_y)
                        .unwrap();

                    chart_analytical.with_projection(|mut p| {
                        p.pitch = (iter as f64 / 10.0).sin() / 3. + 0.333;
                        p.scale = 0.7;
                        p.yaw = iter as f64 / 20.;
                        p.into_matrix()
                    });

                    chart_analytical.configure_axes().draw().unwrap();

                    chart_analytical
                        .draw_series(
                            SurfaceSeries::xoz(x_points.clone(), y_points.clone(), f_analytical)
                                .style_func(&|&v| {
                                    let c = VulcanoHSL::get_color(v);
                                    RGBAColor(c.rgb().0, c.rgb().1, c.rgb().2, 0.5).into()
                                }),
                        )
                        .unwrap();

                    for (name, solution) in &solutions {
                        let current_subplot = if i == solutions.len() - 1 {
                            subplot_solutions.clone()
                        } else {
                            let (upper, lower) = subplot_solutions
                                .split_vertically(graph_height / (1 + solutions.len()) as u32);
                            subplot_solutions = lower;
                            upper
                        };

                        let areas = current_subplot.split_evenly((1, 2));

                        let mut max_error = std::f64::NEG_INFINITY;
                        let mut min_error = std::f64::INFINITY;
                        let mut plot_min_error: f64 = -2.1e-2;
                        let mut plot_max_error: f64 = 2.1e-2;
                        let min_num = analytical_min;
                        let max_num = analytical_max;
                        for yi in 0..y_step_count {
                            for xi in 0..x_step_count {
                                let x = xi as f64 * max_x / (x_step_count - 1) as f64;
                                let y = yi as f64 * max_y / (y_step_count - 1) as f64;
                                let error = solution[num_iter][yi][xi] - analytical(x, y, t);
                                max_error = max_error.max(error);
                                min_error = min_error.min(error);
                                plot_max_error = plot_max_error.max(error);
                                plot_min_error = plot_min_error.min(error);
                            }
                        }

                        let f = |x: f64, y: f64| {
                            let xi = ((x / max_x) * (x_step_count - 1) as f64).round() as usize;
                            let yi = ((y / max_y) * (y_step_count - 1) as f64).round() as usize;
                            solution[num_iter][yi][xi]
                        };
                        let f_error = |x: f64, y: f64| {
                            let xi = ((x / max_x) * (x_step_count - 1) as f64).round() as usize;
                            let yi = ((y / max_y) * (y_step_count - 1) as f64).round() as usize;
                            solution[num_iter][yi][xi] - analytical(x, y, t)
                        };

                        let mut chart_solution = ChartBuilder::on(&areas[0])
                            .caption(format!("{name} at iter = {num_iter}"), ("sans-serif", 20))
                            .build_cartesian_3d(0.0..max_x, min_num..max_num, 0.0..max_y)
                            .unwrap();

                        chart_solution.with_projection(|mut p| {
                            p.pitch = (iter as f64 / 10.0).sin() / 3. + 0.333;
                            p.scale = 0.7;
                            p.yaw = iter as f64 / 20.;
                            p.into_matrix()
                        });

                        chart_solution.configure_axes().draw().unwrap();

                        chart_solution
                            .draw_series(
                                SurfaceSeries::xoz(x_points.clone(), y_points.clone(), f)
                                    .style_func(&|&v| {
                                        let c = VulcanoHSL::get_color(v);
                                        RGBAColor(c.rgb().0, c.rgb().1, c.rgb().2, 0.5).into()
                                    }),
                            )
                            .unwrap();

                        let mut chart_error = ChartBuilder::on(&areas[1])
                            .caption(
                                format!("Error (max = {:.2e})", min_error.abs().max(max_error)),
                                ("sans-serif", 20),
                            )
                            .build_cartesian_3d(
                                0.0..max_x,
                                plot_min_error..plot_max_error,
                                0.0..max_y,
                            )
                            .unwrap();

                        chart_error.with_projection(|mut p| {
                            p.pitch = (iter as f64 / 10.0).sin() / 3. + 0.333;
                            p.scale = 0.7;
                            p.yaw = iter as f64 / 20.;
                            p.into_matrix()
                        });

                        chart_error.configure_axes().draw().unwrap();

                        chart_error
                            .draw_series(
                                SurfaceSeries::xoz(x_points.clone(), y_points.clone(), f_error)
                                    .style_func(&|&v| {
                                        let c =
                                        plotters::style::colors::colormaps::ViridisRGBA::get_color(
                                            1.0 - (v - plot_min_error) / (plot_max_error - plot_min_error),
                                        );
                                        RGBAColor(c.rgb().0, c.rgb().1, c.rgb().2, 0.3).into()
                                    }),
                            )
                            .unwrap();
                    }

                    root.present().unwrap();
                }

                root.present().unwrap();
                open_plot(path);
            }
        }
    }
}
