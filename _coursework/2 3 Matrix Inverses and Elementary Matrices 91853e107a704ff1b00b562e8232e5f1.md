# 2.3. Matrix Inverses and Elementary Matrices

**Proposition 2.3.1 (Solving equations with matrix inverse).** If $A$ has an inverse $A^{-1}$, then for any $\vec b$ the equation $A\vec x = \vec b$ has a unique solution, namely $\vec x = A^{-1}\vec b$. 

## Computing matrix inverses

If $A$ is a $n\times n$ matrix, and you construct the $n\times 2n$ augmented matrix $[A|I]$ and row reduce it, then either: 

1. The first $n$ columns row reduce to the identity, in which case the last $n$ columns of the row-reduced matrix are the inverse of $A$, or
2. The first $n$ columns do not row reduce to the identity, in which case $A$ does not have an inverse.

**Example.**  The matrix $A = \begin{bmatrix}2&1&3\\1&-1&1\\1&1&2\end{bmatrix}$ has inverse $A^{-1} = \begin{bmatrix}3&-1&-4\\1&-1&-1\\-2&1&3\end{bmatrix}$, because

$\begin{bmatrix}2&1&3&1&0&0\\1&-1&1&0&1&0\\1&1&2&0&0&1\end{bmatrix}$ row reduces to $\begin{bmatrix}1&0&0&3&-1&-4\\0&1&0&1&-1&-1\\0&0&1&-2&1&3\end{bmatrix}$.

## Elementary Matrices

Each one of the three row operations on a matrix $A$ can be represnted by multiplication on the left by an elementary matrix. All of these matricies are square. 

1. **Multiplication.** The *type 1 elementary matrix* $E_1(i, x)$ is the square matrix where all nondiagonal entries are $0$, and every entry on the diagonal is $1$ except for the $(i,i)$th entry, which is $x\neq 0$. 
2. **Addition.**  The *type 2 elementary matrix* $E_2(i,j,x)$, for $i\neq j$, is the square matrix with all diagonal entries $1$, and all other entries $0$, except for the $(i,j)$the, which is $x$. 
3. **Exchange.** The *type 3 elementary matrix* $E_3(i,j)$, $i\neq j$, is the square matrix where the entries $i$, $j$ and $j,i$ are $1$, as are all entries on the diagonal except $i,i$ and $j,j$, which are $0$. All others are $0$. 

Writing row operations as matrices lets us use properties of matrices to our advantage.

If $A$ is invertible, then $A^{-1}$ is equal to a product of elementary matrices such that 

$I = E_k\dots E_2E_1A$.

**Proposition 2.3.7.** Any square matrix $A$ can be approximated by a sequence of invertible matrices. 

**Proof.** Let $\tilde A = E_k\dots E_1A$, with $\tilde A$ upper triangular. If any diagonal entires of $\tilde A$  are $0$, exchange them for $1/n$. Denote this matrix as $\tilde A_n$. The matricies$A_n = E_1\dots E_k \tilde A_n$ are invertible, and converge to $A$ as $n\to \infty$.