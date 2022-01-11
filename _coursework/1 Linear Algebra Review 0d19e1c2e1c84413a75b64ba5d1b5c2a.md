---
layout: page
title: 1. Linear Algebra Review
---
# 1. Basic Linear Algebra Review

Feb 8, 2021

## What is a matrix?

A matrix is a 2D array of numbers. A $m\times n$ matrix has $m$ rows and $n$ columns. It is a linear map between vector spaces. $L: \mathcal{V} \to \mathcal{W}$. 

- Any vector in the span of a basis can be written as a linear combination of the basis vectors.
- A vector can be thought as a $n\times 1$ matrix.
- A scalar can be thought of as a $1\times 1$ matrix.

## Common operations on matrices

### Matrix vector products

$$
A\vec x = \vec y.
$$

If we denote $A$ as the set of vectors $[a_1, \dots, a_n]$,

$$
y = \sum_{i=1}^nx_ia_i
$$

Note that if $A$ is $m\times n$, then $\vec x$ must be $n\times 1$.

### Transpose

$$
(A^T)_{ij} = A_{ji}
$$

### Inner product (over $\R$)

$$
\vec x^T\vec y \to \alpha
$$

$x$ and $y$ must be $n\times 1$.

### Outer product

$$
\vec x \vec y^T\to A
$$

### Matrix-matrix multiplication

$$
AB \to C
$$

This can be thought of a sum between a $m\times k$ matrix and $k\times n$ matrix.

$$
C = \sum_{i=1}^m\sum_{r=1}^ka_{ir}b_{rj}
$$


> 💡 Matrix-vector, inner product, and the outer product are all special cases of matrix-matrix multiplication!


Remember that matrix multiplication is linear and associative.

$$
A(\alpha_1B_1 + \alpha_2B_2) = \alpha_1AB_1+\alpha_2AB_2.
$$

## Blocked operations

A matrix can be thought of a combination of smaller matrices. Let $A_{ij}$ denote a block.

$$
\begin{bmatrix}
A_{11}& A_{12}\\
A_{21} & A_{22}

\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix}

= \begin{bmatrix}y_1\\y_2\end{bmatrix}
$$

 The solutions to this equation are

$$
y_1 = A_{11}x_1 + A_{12}x_2\\
y_2 = A_{21}x_1 + A_{22}x_2.
$$

## Linear Algebra Concepts

$$
\text{range}(A) = \text{"Col space of A"}\\
= \{Ax\mid x\in \R^n\}.
$$

From the range, we get the rank, which is the dimension of the image of $A$.

The null space (the kernel), is the set $\{z\mid Az=0\}$.

### Rank-nullity theorem

$$
\text{rank(A)} + \text{dim(null(A))} = n.
$$

**Example.** Suppose $A$ is equal to the outer product $uv^T$. The range of $A$ is 

$$
\{(uv^T)x\mid x\in \R^n\}.
$$

Since $v^Tx$ is a scalar, the rank of $A$ is 1. By the rank nullity theorem, the dimension of the null space is $n-1$.

**Proposition.** If for a matrix $m=n$ and is full rank, then by the rank nullity theorem, the dimension of the null space is 0. The matrix is called nonsingular (has an inverse).

## Major Course Topics

1. Solving linear systems. $x = A^{-1}b$.
2. Least squares problems. $\min_x \|Ax-b\|^2_2$
3. Eigenvalue problems. $Ax = \lambda x$
4. Numerical optimization. $y_i \approx c^T \sigma(W_2\sigma (W_1x_i+b_1)+b_2)$
    1. What types of solutions?
    2. Efficiency
    3. Trade off with speed and quality

This course will involve developing algorithms.

- They guarantee solutions (in exact arithmetic), or approximate answers
- Stable w/r/t perturbations
- Efficiency / computational complexity

**Theme:** Factorization $A = PLU$, $A = QR$, $A = QTQ^T$

## Goals

- Literate and comfortable with matrix comps + basic operations
- Know general building blocks and ideas
- Confidence to find a good solution for your problem
