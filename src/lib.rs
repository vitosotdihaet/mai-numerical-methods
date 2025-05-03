pub mod equation;
pub mod error;
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
    use num::{complex::ComplexFloat, Complex};

    use crate::{
        equation::{halves_method, iterations_method, newtons_method, systems},
        matrix::Matrix,
    };

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
        let x_iterations = iterations_method(phi, 0.5, accuracy);
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
}
