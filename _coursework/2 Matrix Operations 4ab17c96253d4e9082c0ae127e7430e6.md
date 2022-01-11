---
layout: page
title: 2. Matrix Operations
---

# 2. Matrix Operations

February 10, 202

## Basic Linear Algebra Subroutines (BLAS)

Basic linear algebra subroutines are a standard library interface for manipulating dense vectors and matrices. There are 3 levels of BLAS routines.

1. Vectors $O(n)$ flops. $O(n)$ data.
    1. $x \cdot y = z$
2. Matrix-vector $O(n^2)$ flops. $O(n^2)$ data.
    1. $Ax = B$
3. Matrix-matrix $O(n^3)$ flops. $O(n^2)$ data.
    1. $AB = C$

## Locality and memory

Matrices are usually stored as contiguous arrays, in **column major order**. The two costs of operations are

- arithmetic operations (**flops**)
- time spent fetching data (communication).

The cost of communication is much higher than the cost of a floating point operation.


> 💡 Communication takes about 100 cycles to fetch something.



Memories are optimized for

- spatial locality
    - It is faster to access data sequentially that is close to each other.
- temporal locality
    - This includes reusing data in cache.

## Examples of matrix vector products

Here is a basic implementation of matrix-vector multiplication.

```julia
function matvec_row(A, x)
	m, n = size(A)
	y = zeros(eltype(x), m)
	for i = 1:m
		for j = 1:n
			y[i] += A[i,j] * x[j]
		end
	end
	return y
end
```

This has some spatial locality in $y$ because it is reusing cache.

It has some temporal locality in $x$ because it is sequentially calling the pointer.

However, this is slower than iterating over the $n$ then $m$! This is because the column-wise implementation has spatial locality for both $i$ and $j$.

### Matrix multiplication

Notice that for BLAS3 operations, it takes $O(n^3)$ flops but uses $O(n^2)$ data. This creates a big opportunity for efficiency.

One way we can try to compute this is by the **inner product.** Suppose we want to compute $C = AB$. We can write $A$ as a $n\times 1$ matrix.

$$
C= \begin{bmatrix}
a_1^T\\\vdots\\a_m^T 
\end{bmatrix} \begin{bmatrix}b_1&\dots& b_n\end{bmatrix}\\
C_{ij} = a_i^T b_j
$$

We can also try computing an **outer** **product**.

$$
C= \begin{bmatrix}a_1&\dots& a_n\end{bmatrix}\begin{bmatrix}
b_1^T\\\vdots\\b_m^T 
\end{bmatrix} \\
C_{ij} = a_i b_j^T
$$

In practice, these can be made much faster by blocking.

## Blocking

Blocking is a way to increase temporal locality by representing matrices as smaller matrices. It allocates memory that can be allocated fast.

## Structured matrices

There are a couple important matrices that we should know.

1. Structured matrices
    1. Diagonal
    2. Tridiagonal
    3. Permutation
        1. One nonzero per row/col

---
