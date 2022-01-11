---
layout: page
title: 4. Singular Value Decomposition
---
# 4. Singular Value Decomposition

The singular value decomposition makes use of the observation that matrices are essentially operations on a unit hypersphere. If $A$ has rank $r$, then exactly $r$ lengths of the hyperellipse, $\sigma_i$, are nonzero. If $m\geq n$, then at most $n$ will be nonzero.

**Definition 4.1.1 (Singular values).** Suppose we have the unit sphere $S$ and $A \in \R^{m\times n}$. The *singular values* of $A$, written $\sigma\_1, \dots, \sigma\_n$, are the lengths of the $n$ principal semiaxes of $AS$.

**Definition 4.1.2 (Left singular values).** The *left singular values* of $A$ are the unit vectors $\{u\_1, \dots, u\_n\}$ in the directions of the principal semiaxes of $AS$. The vector $\sigma\_i u\_i$ is the $i$th largest principal semiaxis of AS.

**Definition 4.1.3 (Right singular values).** The *right singular values* of $A$, $\{v\_1, \dots, v\_n\}$ are the preimages of the principal semiaxes of $AS$.

$$
Av_i = \sigma_iu_i.
$$

**Theorem 4.1.4 (Singular value decomposition).** The collection of vectors that make the principal semiaxes of $AS$ can be written as the matrix equation

$$
AV = U\Sigma\\A = U\Sigma V^T
$$

where $U\in \R^{m\times m}$ and $V\in \R^{n\times n}$. Both  $V$ and $U$ are orthonormal (unitary) and $\Sigma$ is diagonal along the first $n$ rows. The values of $\Sigma$ are the singular values. In this case, $A$ does not need to have full rank.

 

## Compact/Reduced/Thin SVD

The reduced SVD of $A$ is a similar transformation where $\hat \Sigma$ is $n\times n$ and $\hat U$ is $m\times n$. Here, $\hat U$ has less columns than  $U$ since it has rank $n$. 

$$
A = \hat U \hat \Sigma V^T.
$$

## SVD and Norms

Using SVD, the norm of the matrix is equal to its largest singular value.

$$
\|A\|_2 \|U\Sigma V^T\| = \|\Sigma \|_2 = \sigma_1.
$$

## Relationship with Eigendecomposition

**Theorem.** The nonzero singular values of $A$ are the square roots of the nonzero eigenvalues of $A^TA$ or $AA^T$:

$$
A^TA = V\Sigma U^TU\Sigma V^T = V\Sigma^2V^T.
$$

From this, we see that $\|A\|\_2 = \sqrt{\lambda\_{max}(A^TA)} = \sqrt{\sigma^2} = \sigma$. The values of $\Sigma^2$ are the eigenvalues.

**Theorem.** If $A = A^T$ ($A$ is symmetric), then the singular values of $A$ are the absolute values of the eigenvalues of $A$. 

$$
A = X\Lambda X^T = X|\Lambda|\text{sign}(\Lambda|) X^T
$$

where $X$ is orthonormal. There is no guarantee that the eigenvalues are non negative.

## Solving Linear Systems

SVD can be used to solve equations. Suppose

$$
Ax = b.
$$

By the SVD,

$$
U\Sigma V^Tx = b\\
\Sigma V^Tx = U^Tb\\
\Sigma y = U^Tb\\
y = \Sigma^{-1}U^Tb\\
x = V\Sigma^{-1}U^Tb.
$$

## Least Squares

Suppose w want to solve the equation

$$
\min_x\|Ax-b\|_2.
$$

By SVD, and since the 2 norm is invariant under orthogonal transformations,

$$
\begin{gather}
\|Ax-b\|_2\\
 = \|U\Sigma V^Tx - b\|_2\\
= \|\Sigma V^Tx - U^Tb\|_2.\\
= \|\begin{bmatrix}\Sigma\\0\end{bmatrix}V^Tx - \begin{bmatrix}U_1^Tb\\U_2^Tb\end{bmatrix}\|_2\\
\|\Sigma V^Tx - U_1^Tb\|_2\\
= \Sigma V^T x = U_1^Tb\\
x = V\Sigma^{-1}U_1^Tb.
\end{gather}
$$

## Rank/Range/Null Space

**Theorem.** The rank of $A$ is $r$, the number of nonzero singular values.

**Proof.** $U$ and $V^T$ have full rank since they are orthogonal. Therefore,

$$
\text{rank}(A) = \text{rank}(\Sigma).
$$

Since $\Sigma$ is diagonal, the range of $\Sigma$ is just the span of its singular values. Therefore, its rank is the number of nonzero singular values.

**Theorem.** The range of $A$ is the span of the vectors of $U$. The null space of $A$ is the span of the vectors of $V$.

**Proof.** 

$$
\text{range}(A) = \{Ax \mid x\in \R^n\}\\
 = \{U\Sigma V^T x \mid x\in \R^n\}.
$$

Since $V^T$ has the same number of pivotal columns as $A$, they have the same span.

$$
\{U\Sigma V^T x \mid x\in \R^n\} = \{U\Sigma y \mid y \in \R^n\}.
$$

Since $\Sigma$ is diagonal, each term in $\Sigma y$ is equal to $\sigma_iy_i$. Therefore, the span of $A$ is the span of $U$. 

Since $U$ has full rank, it has a trivial null space. Therefore, the null space of $A$ is equal to the null space of $V^T$.

## SVD as low rank approximation

**Theorem.** Another interpretation of the singular value decomposition is that $A$ is the sum of $r$ rank 1 matrices:

$$
A = \sum_{i=1}^r\sigma_iu_iv_i^T.
$$

**Demo.** An image can be compressed by representing its matrix as rank 1 matrices. For a matrix with high ranks, $\sigma_n$ is much greater than the other singular values. These can be discarded.

```julia
using PyPlot
using Images
using LinearAlgebra
ImgA = Float64(Gray.(load("cornell.jpg")))
PyPlot.imshow(A, cmap = "gray");
@show size(A)

function rank_k_approx(A, k)
	F = svd(A) #thin SVD by defeault
	Ak = F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.V[:, 1:k]'
	return Ak
end

PyPlot.imshow(rank_k_approx(A, 50), cmap="gray");
PyPlot.imshow(rank_k_approx(A, 500), cmap="gray");
PyPlot.imshow(rank_k_approx(A, 5000), cmap="gray");
```

---
