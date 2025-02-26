pub mod matrix;

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn identity_matrix() {
        let m = Matrix::identity_matrix(3);
        assert_eq!(
            m,
            Matrix::new(vec![vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.],])
        );
    }
}

#[cfg(test)]
mod labs {
    use crate::matrix::Matrix;

    /// Ax = b
    ///
    /// A = LU => LUx = b
    ///
    /// UX = z
    ///
    /// Lz = b
    #[test]
    fn lab1() {
        let a = Matrix::new(vec![
            vec![-7., 3., -4., 7.],
            vec![8., -1., -7., 6.],
            vec![9., 9., 3., -6.],
            vec![-7., -9., -8., -5.],
        ]);

        let b = Matrix::new(vec![vec![-126.], vec![29.], vec![27.], vec![34.]]);

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

        let z = Matrix::get_z(&l, &b);
        let x = Matrix::get_x(&u, &z);

        // https://matrixcalc.org/ru/slu.html#solve-using-Gaussian-elimination%28%7B%7B-7,3,-4,7,-126%7D,%7B8,-1,-7,6,29%7D,%7B9,9,3,-6,27%7D,%7B-7,-9,-8,-5,34%7D%7D%29
        assert_eq!(
            x,
            Matrix::new(vec![vec![8.], vec![-9.], vec![2.], vec![-5.]])
        );
    }
}
