# 2.2. Solving Equations with Row Reduction

Not every system of linear equations has a solution. We define certain criteria for what different systems mean.

**Theorem 2.1.1 (Solutions to linear  equation).** Represent the system $A\vec x = \vec b$, involving $m$ linear equations in $n$ unknown, by the $m\times (n+1)$m matrix $[A|\vec b]$, which row reduces to $[\tilde A|\tilde b]$. Then,

1. If the row-reduced vector $\tilde b$ contains a pivotal 1, the system has no solutions.
2. If $\tilde b$ does not contain a pivotal 1, then solutions are uniquely determined by the values of the nonpivotal variables:
    1. If each column of $\tilde A$ contains a pivotal 1 (so there are no nonpivotal variables), the system has a unique solution.
    2. If at least one column of $\tilde A$ is nonpivotal, there are infinitely many solutions: exactly one for each value of the nonpivotal variables. 

**Examples**

The matrix $\begin{bmatrix}2&1&3&1\\1&-1&0&1\\1&1&2&1\end{bmatrix}$ row reduces to $\begin{bmatrix}1&0&1&0\\0&1&1&0\\0&0&0&1\end{bmatrix}$, which has no solutions since the pivotal 1 in the last row is in the $\tilde b$ column.

The matrix $\begin{bmatrix}2&1&3&1\\1&-1&0&1\\1&1&2&1/3\end{bmatrix}$ row reduces to $\begin{bmatrix}1&0&1&2/3\\0&1&1&-1/3\\0&0&0&0\end{bmatrix}$. This has infinite solutions because we can choose $z$ arbitrarily.