# 10 вариант

# 1.1
$$Ax = b$$

## Идея
Разложим исходную матрицу $A$ на матрицы $L$ и $U$, в $L$ на диагонали единицы, выше - нули, ниже - значения коэффициента, необходимые для построения ступенчатой матрицы $U$

0. Приведение к ступенчатому виду матрицу происходит за $O(n^3)$ - на каждом шаге $k$ мы вычитаем строку из $n - k + 1$ значений из $n - k + 1$ строк, всего шагов $k$

    После разложения, получим $LUx = b$

1. Решим $Lz = b$, что происходит очень просто из-за вида матрицы:
    $$
    \begin{pmatrix}
    1 & 0 & \cdots & 0 \\
    l_{2,1} &  1 & \cdots & 0 \\
    \cdots & \cdots & \cdots & \cdots \\
    l_{n,1} &  l_{n, 2} & \cdots & 1 \\
    \end{pmatrix}
    z =
    \begin{pmatrix}
    b_1 \\ b_2 \\ \cdots \\ b_n
    \end{pmatrix}
    $$

    1: $l_{1,1}z_1 = b_1 \Rightarrow z_1 = b_1$

    2: $l_{2,1}z_1 + l_{2,2}z_2 = b_2 \Rightarrow z_2 = b_2 - l_{2,1} z_1$

    $i$: $z_i = b_i - \sum\limits_{j = 1}^{i-1}l_{i,j}z_j,\ i = \overline{1,n}$, что происходит за $O(n^2)$

2. Решим $Ux = z$:
    $$
    \begin{pmatrix}
    u_{1,1} & u_{1,2} & \cdots & u_{1,n} \\
    0 &  u_{2,2} & \cdots & u_{2,n} \\
    \cdots & \cdots & \cdots & \cdots \\
    0 &  0 & \cdots & u_{n,n} \\
    \end{pmatrix}
    x =
    \begin{pmatrix}
    z_1 \\ z_2 \\ \cdots \\ z_n
    \end{pmatrix}
    $$

    $n$: $u_{n,n}x_n = z_n \Rightarrow x_n = \dfrac{z_n}{u_{n,n}}$
    
    $n - 1$: $u_{n - 1,n - 1}x_{n-1} + u_{n - 1, n}x_n = z_{n - 1} \Rightarrow x_{n - 1} = \dfrac{z_{n - 1} - u_{n-1,n}x_n}{u_{n-1,n-1}}$

    $i$: $x_i = \dfrac{z_i - \sum\limits_{j=i+1}^{n}u_{i,j}x_j}{u_{i,i}},\ i = \overline{n, 1}$ за $O(n^2)$

## Применение
Этот метод эффективен для решения множества уравнений с разными свободными значениями, потому что разложение $A$ на $L$ и $U$ происходит за $O(n^3)$ и таким образом как бы кешируется часть решение, а само решение, связанное со свободными значениями происходит за $O(n^2)$, что намного лучше, чем просто метод Гаусса за $O(n^3)$


# 1.2
$$
\begin{pmatrix}
b_1 & c_1 & 0 & \cdots & 0 \\
a_2 & b_2 & c_2 & \cdots & 0 \\
0 & a_3 & b_3 & \cdots & 0 \\
\cdots & \cdots & \cdots & \cdots & \cdots \\
0 & 0 & 0 & \cdots & c_n \\
\end{pmatrix}
x = d
$$

## Идея
Выразим $x_i$ как $x_i = P_ix_{i+1} + Q_i, i = \overline{1,n}$:

1: $b_1x_1 + c_1x_2 = d_1\ \Rightarrow\ x_1 = \dfrac{-c_1}{b_1} x_2 + \dfrac{d_1}{b_1}\ \Rightarrow\ P_1 = \dfrac{-c_1}{b_1}, Q_1 = \dfrac{d_1}{b_1}$

2: $a_2x_1 + b_2x_2 + c_2x_3 = d_2 \ \Rightarrow\ x_2 = \dfrac{-a_2}{b_2}x_1 - \dfrac{-c_2}{b_2}x_3 + d_2 = \dfrac{-a_2}{b_2}P_1x_2 + \dfrac{-a_2}{b_2}Q_1 - \dfrac{-c_2}{b_2}x_3 + d_2 \Rightarrow$
$$
x_2(1 + \dfrac{-a_2}{b_2}P_1) = \dfrac{-a_2}{b_2}Q_1 + \dfrac{-c_2}{b_2}x_3 + d_2\ \Big|\ \colon (1 + \dfrac{-a_2}{b_2}P_1) \\
x_2 = \dfrac{-c_2}{b_2 + a_2P_1}x_3 + \dfrac{d_2 - a_2 Q_1}{b_2 + a_2P_1}
$$

$i$: $x_i = \dfrac{-c_i}{b_i + a_iP_{i - 1}}x_{i+1} + \dfrac{d_i - a_i Q_{i - 1}}{b_i + a_iP_{i - 1}},\ i = \overline{1, n}$

Заметим, что:
- на последнем шаге $a_n = 0$, значит $P_1 = \dfrac{-c_1}{b_1}, Q_1 = \dfrac{d_1}{b_1}$ 
- на последнем шаге $c_n = 0$, значит $P_n = 0, Q_n = x_n$

При выражении $x_i$ через $x_{i-1}$ метод называется методом правой прогонки, иначе - левой

## Применение

Самый эффективный метод для трехдиагональных матриц - за $O(n)$