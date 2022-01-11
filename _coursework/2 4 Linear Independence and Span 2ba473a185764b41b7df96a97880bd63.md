# 2.4. Linear Independence and Span

This chapter defines vocabulary used to give a precise way to describe linear equations. 

**Definition 2.4.1(Linear combination).** If $\vec v_1 , \dots, \vec v_k$ is a collection of vectors in $\R^n$, then a *linear combination* of the $\vec v_i$ is a vector $\vec w$ of the form

$$
\vec w = \sum_{i=1}^k a_i\vec v_i \text{\space\space\space\space\space for any scalars }a_i
$$

**Example.** The vector $\begin{bmatrix}3\\4\end{bmatrix}$ is a linear combination of $\vec e_i$ and $\vec e_2$:

$$
\begin{bmatrix}3\\4\end{bmatrix} = 3\begin{bmatrix}1\\0\end{bmatrix} + 4\begin{bmatrix}0\\1\end{bmatrix}.
$$

## Linear independence and span

*Linear independence* is a way to talk about uniqueness of solutions to linear equations; *span* is a way to talk about the existence of solutions. 

**Definition 2.4.2(Linear independence).** A set of vectors $\vec v_1,\dots \vec v_k$ is *linearly independent* if the *only* solution to 

$$
a_1\vec v_1 + a_2\vec v_2 + \dots + a_k\vec v_k = \vec 0 \text{ \space\space  is \space \space }a_1=a_2=\dots=a_k = 0.
$$

An equivalent definition is that vectors $\vec v_1, \dots, \vec v_k \in \R^n$ are linearly independent if and only if a vector $\vec w \in \R^n$ can be written as a linear combination of those vectors in at most one way: 

$$
\sum_{i=0}^k y_i \vec v_i = \sum_{i=0}^k x_i \vec v_i \text{ \space \space implies\space\space } y_1 =x_1, \dots,y_k = x_k.
$$

It is also equivalent to say that vectors are linearly independent if none of $\vec v_i$ is a linear combination of the others. If the columns of $[T]$ are linearly independent, then the transformation $T$ is injective. 

Geometrically, one vector is linearly independent if it isn't the zero vector. Two vectors are linearly independent if they don't both lie on the same line. Three vectors are linearly independent if they don't all lie in the same plane.

**Example.** The vectors $\vec e_1, \vec e_2 \in \R^2$ are linearly independent. There is only one way to write $\begin{bmatrix}3\\4\end{bmatrix}$ in terms of $\vec e_i$ and $\vec e_2$:

$$
3\begin{bmatrix}1\\0\end{bmatrix} + 4\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}3\\4\end{bmatrix}.
$$

However, $\begin{bmatrix}1\\0\end{bmatrix}, \begin{bmatrix}0\\1\end{bmatrix},\begin{bmatrix}3\\2\end{bmatrix}$ are not linearly independent because 

$$
\begin{bmatrix}3\\2\end{bmatrix} + 2\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}3\\4\end{bmatrix}.
$$

**Definition 2.4.3 (Span).** The *span* of $\vec v_1,\dots \vec v_k$ is the set of linear combinations $a_1\vec v_1 + \dots + a_k \vec v_k$. It is denoted $\text{Span}(\vec v_1, \dots , \vec v_k)$.

Span can also be used as a verb. $\vec e_1$ and $\vec e_2$ span $\R^2$ because every vector in the plane is a linear combination of the two standard basis vectors. Saying that the columns of $T: \R^n \to \R^m$ span $\R^m$ is equivalent to saying that $T$ is surjective.

**Exercise.** Given the vectors 

$$
\vec v_1 = \begin{bmatrix}1\\0\\0\\-1\end{bmatrix}, \vec v_2 = \begin{bmatrix}-2\\-1\\1\\0\end{bmatrix}, \vec v_3 = \begin{bmatrix}1\\1\\-1\\1\end{bmatrix}, \vec v_4 = \begin{bmatrix}0\\0\\1\\0\end{bmatrix},
$$

is $\vec v_4$ in $\text{Span}(\vec v_1, \vec v_2, \vec v_3)$?

No. Because $a_2(-1) + a_3(1) = 0$, $a_2 = a_3$. This contradicts the statement that $a_2 (1) + a_3(-1) = 1$. 

**Theorem 2.4.5 (Linear independence and span).** Let $\vec v_1, \dots, \vec v_k$ be vectors in $\R^n$; let $A$ be the $n\times k$ matrix $[\vec v_1, \dots, \vec v_k]$. Then

1. $\vec v_1, \dots, \vec v_k$ are linearly independent if and only if the row-reduced matrix $\tilde A$ has a pivotal 1 in every column.
2. $\vec v_1, \dots, \vec v_k$ span $\R^n$ if and only if $\tilde A$ has a pivotal $1$ in every row. 

**Proof.**

1. The vectors $\vec v_1 ,\dots \vec v_k$ are linearly independent if-and-only-if the solution to $A\vec x = \vec 0$ is $\vec x = 0$. Therefore, by Theorem 2.2.1, the system has a unique solution and each column has a pivotal 1. 
2. The vectors $\vec v_1, \dots, \vec v_k$ span $\R^n$ if and only if for any $\vec b \in \R^n$, the equation $A\vec x = \vec b$ has a solution. The row reduction of $[A|\vec b]$ is $[\tilde A, \tilde b]$. There exists $\tilde b$ containing a pivotal $1$ if and only if $\tilde A$ contains a row of $0$s. Thus the equation $A\vec x = \vec b$ has a solution for every $\vec b$ if and only if $\tilde A$ has a pivotal $1$ in every row. 

**Theorem 2.4.10.** In $\R^n$, $n+1$ vectors are never linearly independent, and $n-1$ vectors in $\R^n$ never span $\R^n$. 

**Proof.** When we use $n+1$ vectors in $\R^n$ to form a matrix $A$, the matrix is $n\times (n+1)$; when we row reduce $A$ to $\tilde A$, at least one column of $\tilde A$ contains no pivotal 1, since there can be at most $n$ pivotal 1's but the matrix has $n+1$ columns. So by part 1 of Theorem 2.4.5, the $n+1$ vectors are not linearly independent. 

When we form a matrix $A$ from $n-1$ vectors in $\R^n$, the matrix is $n\times (n-1)$. When we row reduce $A$, the resulting matrix must have at least one row with all $0$s, since we have $n$ rows but at most $n-1$ pivotal $1$s. So by part 2 of Theorem 2.4.5, $n-1$ vectors in $\R^n$ cannot span $\R^n$. 

## A set of vectors as a basis

Choosing a set of basis vectors is like choosing axes in a plane or in space. Their length and direction give us units for the axes.

 

**Definition 2.4.11 (Basis).** Let $V\subset\R^n$ be a subspace. An ordered set of vectors $\vec v_1, \dots, \vec v_k\in V$ is called a basis of $V$ if it satisfies any of the three equivalent conditions.

1. The set is a *maximally linearly independent set*: it is linearly independent, and if you add one more vector, the set will no longer be linearly independent.
2. The set is a *minimal spanning set*: it spans $V$, and if you drop one vector, it will no longer span $V$. 
3. The set is a *linearly independent set* spanning $V$.

In $\R^n$, the fundamental example of basis vectors is the *standard basis*. These vectors span $\R^n$ and are linearly independent.

## Orthonormal bases

**Definition 2.4..15 (Orthonormal set, orthonormal basis).** A set of vectors $\vec v_1, \dots \vec v_k$ is *orthonormal* if each vector in the set is orthogonal to every other vector in the set, and all vectors have length $1$: 

$$
\vec v_i \cdot \vec v_j = 0 \text{ for } i\neq j \text{ and }|\vec v_i| = 1 \text{ for all }i\leq k .
$$

An orthonormal set is an *orthonormal basis* of the subspace $V \subset \R^n$ that it spans. 

To check if orthogonal vectors form a basis, we only need to check their span, because they are automatically linearly independent. 

**Example.** Consider the infinite-dimensional space $C[0,\pi]$ of continuous functions on $[0, \pi]$ with the inner product

$$
<f,g> =\int_0^\pi f(x)g(x) dx.
$$

This "dot product" has the properties

$$
<af_1+bf_2, g> = a<f_1, g>+b<f_1,g>\\
<f,g> = <g, f>\\
<f,f> = 0 \text{ if and only if } f = 0.
$$

The proof is left as an exercise to the reader. 

We claim that the elements $\sin nx, n=1,2,\dots$ of $c[0,\pi]$ form an orthogonal set: if $n\neq m$ we have

$$
\int_0^\pi\sin nx\sin mxdx = 0.
$$

## Orthogonal matrices

A matrix whose columns form an orthonormal basis is called an *orthogonal matrix*. 

**Def 2.4.19 (Orthogonal matrix).** An $n\times n$ matrix is called *orthogonal* if it satisfies one of the following equivalent conditions:

1. $AA^T = A^TA = A$, i.e., $A^T = A^{-1}$.
2. The columns of $A$ form an orthonormal basis of $\R^n$.
3. The matrix $A$ preserves dot products: for any $\vec v$, $\vec w\in \R^n$, we have $A\vec v \cdot A \vec w = \vec v \cdot \vec w$. 

## Dimension

**Definition 2.4.21 (Dimension).** Every subspace $E \subset \R^n$ has a basis, and any two bases of $E$ have the same number of elements, called the dimension of $E$, and denoted $\dim E$.