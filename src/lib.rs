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
mod labs {
    use num::{complex::ComplexFloat, Complex};

    use crate::matrix::Matrix;

    /// Ax = b
    ///
    /// A = LU => LUx = b
    ///
    /// UX = z
    ///
    /// Lz = b
    #[test]
    fn lab_1_1() {
        let a = Matrix::new(vec![
            vec![-7., 3., -4., 7.],
            vec![8., -1., -7., 6.],
            vec![9., 9., 3., -6.],
            vec![-7., -9., -8., -5.],
        ]);

        let b = Matrix::column(&[-126., 29., 27., 34.]);

        let (l, u) = a.get_lu();

        // https://www.emathhelp.net/calculators/linear-algebra/lu-decomposition-calculator/?i=%5B%5B-7%2C3%2C-4%2C7%5D%2C%5B8%2C-1%2C-7%2C6%5D%2C%5B9%2C9%2C3%2C-6%5D%2C%5B-7%2C-9%2C-8%2C-5%5D%5D
        assert_eq!(
            l,
            Matrix::new(vec![
                vec![1., 0., 0., 0.],
                vec![-8. / 7., 1., 0., 0.],
                vec![-9. / 7., 90. / 17., 1., 0.],
                vec![1., -84. / 17., -208. / 201., 1.],
            ])
        );

        assert_eq!(
            u,
            Matrix::new(vec![
                vec![-7., 3., -4., 7.],
                vec![0., 17. / 7., -81. / 7., 14.],
                vec![0., 0., 1005. / 17., -1209. / 17.],
                vec![0., 0., 0., -1100. / 67.],
            ])
        );

        // https://matrixcalc.org/ru/slu.html#solve-using-Gaussian-elimination%28%7B%7B-7,3,-4,7,-126%7D,%7B8,-1,-7,6,29%7D,%7B9,9,3,-6,27%7D,%7B-7,-9,-8,-5,34%7D%7D%29
        let x = a.solve_lu(&b);
        assert_eq!(x, Matrix::column(&[8., -9., 2., -5.]));

        // https://math.semestr.ru/matrix/index.php
        let inversed = a.inversed();
        assert_eq!(
            inversed,
            Matrix::new(vec![
                vec![-3. / 55., 3. / 55., 1. / 165., -1. / 55.],
                vec![43. / 500., -2. / 125., 31. / 375., 1. / 500.],
                vec![-329. / 5500., -69. / 1375., -81. / 1375., -403. / 5500.],
                vec![19. / 1100., 9. / 275., -52. / 825., -67. / 1100.],
            ])
        );

        let n = a.row_count();
        assert_eq!(&a.clone() * &inversed, Matrix::identity(n));

        let determinant = a.determinant();
        assert!((determinant - 16500_f64).abs() < 1e-9);
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

        // https://matrixcalc.org/ru/slu.html#solve-using-Gaussian-elimination%28%7B%7B-7,-6,0,0,0,-75%7D,%7B6,12,0,0,0,126%7D,%7B0,-3,5,0,0,13%7D,%7B0,0,-9,21,8,-40%7D,%7B0,0,0,-5,-6,-24%7D%7D%29
        let x = a.solve_tridiagonal(&d);
        assert_eq!(x, Matrix::row(vec![3., 9., 8., 0., 4.]));
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

        // https://matrixcalc.org/ru/slu.html#solve-using-Gaussian-elimination%28%7B%7B28,9,-3,-7,-159%7D,%7B-5,21,-5,-3,63%7D,%7B-8,1,-16,5,-45%7D,%7B0,-2,5,8,24%7D%7D%29
        let answ = Matrix::column(&[-6., 3., 6., 0.]);

        let x_jacobian = a.solve_jacobian(&b, accuracy).unwrap();
        assert!(x_jacobian.eq_lossy(&answ, accuracy));

        let x_seidel = a.solve_seidel(&b, accuracy).unwrap();
        assert!(x_seidel.eq_lossy(&answ, accuracy));
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

    // #[test]
    // fn lab_2_1() {}
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
            .solve_jacobian(&b, accuracy)
            .unwrap()
            .eq_lossy(&answ, accuracy));
        assert!(a
            .solve_seidel(&b, accuracy)
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
    fn lab_1_5_extra() {
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
