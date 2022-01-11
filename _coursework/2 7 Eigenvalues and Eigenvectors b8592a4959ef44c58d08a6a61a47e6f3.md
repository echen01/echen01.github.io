# 2.7. Eigenvalues and Eigenvectors

**Example 2.7.1 (Fibonacci Numbers).** We can use a change of basis to create a formula for the Fibonacci numbers.

$$
a_n = \frac{5+\sqrt{5}}{10} \bigg(\frac{1+\sqrt{5}}{2}\bigg)^n + \frac{5-\sqrt{5}}{10}\bigg(\frac{1-\sqrt{5}}{2}\bigg)^n. 
$$

For any matrix $A$, the eigenvalues are the roots of the characteristic polynomial $\det (\lambda I - A) = 0$. This is not a good algorithm in practice because roots of high degree polynomials are hard to compute. In practice, QR decomposition is used. For symmetric matrices, the Jacobi algorithm is used. 

**Definition 2.7.2 (Eigenvector, eigenvalue, multiplicity).**  Let $V$ be a complex vector spce and $T: V\to V$ a linear transformation. A nonzero vector $v$ such that 

$$
Tv = \lambda v
$$

for some number $\lambda$ is called an *eigenvector* of $T$. The number $\lambda$ is the corresponding *eigenvalue.* The *multiplicity* of an eigenvalue $\lambda$ is the dimension of the *eigenspace* $\{v\mid Tv = \lambda v\}$. 

**Definition 2.7.3 (Eigenbasis).** A basis for a complex vector space $V$ is an *eigenbasis* of $V$ for a linear transformation $T$ if each element of the basis is an eigenvector of $T$.

## Diagonalization and eigenvectors

**Proposition 2.7.5 (Diagonalization and eigenvectors).** Let $A$ be an $n\times n$ matrix and $P = [\vec v_1 , \dots, \vec v_n]$ an invertible $n\times n$ matrix. 

1. The *eigenvalues* of $A$ and the *eigenvalues* of $P^{-1}AP$ coincide. 
2. If $(\vec v_1, \dots, \vec v_n)$ is an eigenbasis of $\Complex^n$ for $A$, with $A\vec v_i = \lambda_i \vec v_i$, then $P^{-1}AP$ is a diagonal matrix with diagonal entires $\lambda_1, \dots, \lambda_n$. 
3. Conversely, if $P^{-1}AP$ is diagonal with diagonal entries $\lambda_i$, the columns of $P$ are eigenvectors of $A$ with eigenvalues $\lambda_i$. 

<aside>
💡 $P^{-1}AP$ should remind you of the change of base formula.

</aside>

## Finding an eigenbasis

Suppose we have the vectors $\vec e_1, A\vec e_1, A^2 \vec e_1, \dots$ What is the first $A^k \vec e_1$ which is linearly dependent on the earlier vectors? This is just what row reduction is for!

$$
A^k \vec e_1 = a_0 \vec e_1 + a_1A\vec e_2 +a_2A^2\vec e_2 + \dots
$$

The roots of the polynomial $A^k - a_{k-1}A^{k-1} - \dots - a_0 = 0$ are eigenvalues of $A$. 

<aside>
💡 Most often, $k=n$, so $\vec e_1, A\vec e_1, A^2 \vec e_1, \dots,A^{k-1}\vec e_1$ are all linearly independent and the roots are all eigenvalues, which are then distinct.

</aside>

We can use the coefficients $a_0\dots a_k$ to define a polynomial 

$$
p(t) = a_1+a_1t+\dots+a_{k-1}t^{k-1}+t^k
$$

which satisfies $p(A)\vec e_1 = \vec 0$. By the fundamental theorem of algebra, $p$ has at least one root $\lambda$, so we can write 

$$
p(t)=(t-\lambda)q(t)
$$

for some polynomial $q$ of degree $k-1$.  Define $\vec v = q(A)\vec e_1$. Then,

$$
(A-\lambda I)\vec v = (A-\lambda I)(q(A)\vec e_1)=\Big((A-\lambda I)(q(A))\Big)\vec e_1 = p(A)\vec e_1 = \vec 0,
$$

so $A\vec v = \lambda \vec v$. For $\vec v$ to be an eigenvector with eigenvalue $\lambda$, we still need to check that $\vec v \neq\vec 0.$
$\vec v$ is not $\vec 0$ because it is equal to $q(A)\vec e_1$ and $q(A) = a_{k-1}A^{k-1}e_1+\dots+a_1$ is not $0$ since $k>0$. 

**In class example.** Find the eigenvectors and eigenvalues for $A = \begin{bmatrix}1&2\\-1&3\end{bmatrix}$. 

By row reduction, this is equal to 

$$
\begin{bmatrix}
1&0&-5\\0&1&4\end{bmatrix}.
$$

This tells us that $A^2\vec e_1 = -5\vec e_1 + 4A\vec e_1$, or $p(t) = t^2 -4t+5 = 0$. 

$$
\lambda_{1,2} = 2 \pm\sqrt{4-5} = 2\pm i.
$$

$$
p(t) = (t-(2+i))(t-(2-i)),
$$

so $(A-(2-i)I)\vec e_1$ is an eigenvector for the eigenvalue $2+i$. Let's check our work.

$$
(A-(2-i)I)\vec e_1 = \begin{bmatrix}1-(2-i)&2\\-1&3-(2-i)\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix}
$$

$$
= \begin{bmatrix}-1+i\\-1\end{bmatrix}.
$$

Is it true that 

$$
\begin{bmatrix}1&2\\-1&3\end{bmatrix}\begin{bmatrix}-1+i\\-1\end{bmatrix}\stackrel{?}{=}(2+i)\begin{bmatrix}-1+i\\-1\end{bmatrix}
$$

$$
= \begin{bmatrix}-3+i\\-2-i\end{bmatrix}.
$$

They do in fact agree. 

**Example.** Find the eigenbasis for $\begin{bmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{bmatrix}$. 

We first find $[\vec e_1|A\vec e_1|A^2\vec e_1|A^3 \vec e_1]$, which is

$$
\begin{bmatrix}1&1&2&5\\0&-1&-3&-9\\0&0&1&4\end{bmatrix}=\begin{bmatrix}1&0&0&0\\0&1&0&-3\\0&0&1&4\end{bmatrix}.
$$

This tells us that $A^3\vec e_1 = -3A\vec  e_1 + 4A^2\vec e_1$. Our polynomial becomes 

$$
t^3 -4t^2+3t = 0 = t(t^2-4t+3) = t(t-3)(t-1).
$$

For $\lambda_1 = 0$, we write $p(t) = t(t^2-4t+3) = tq_1(t)$ to find the eigenvector

 

$$
q_1(A)\vec e_1 = \begin{bmatrix}2\\-3\\1\end{bmatrix}-4\begin{bmatrix}1\\-1\\0\end{bmatrix}+3\begin{bmatrix}1\\0\\0\end{bmatrix} = \begin{bmatrix}1\\1\\1\end{bmatrix}. 
$$

For $\lambda_2=1$, we solve $p(t) = (t-1)(t^2-3t) = (t-1)q_2(t)$. 

$$
q_2(A)\vec e_1 = (A^2-3A)\vec e_1 
$$

$$
=\begin{bmatrix}2\\-3\\1\end{bmatrix}-3\begin{bmatrix}1\\-1\\0\end{bmatrix} = \begin{bmatrix}-1\\0\\1\end{bmatrix}.
$$

For  $\lambda_3 = 3$, $p(t) = (t-3)(t^2 - t) = (A^2 - A)\vec e_1$ $= \begin{bmatrix}1\\-2\\1\end{bmatrix}$. 

**Theorem 2.7.9 (Existence of eigenbasis).** Let $p_i$ be the lowest degree nonzero polynomial satisfying $p_i(A)\vec e_i = \vec 0$. Let $A$ be an $n\times n$ complex matrix. There exists an eigenbasis in $\Complex^n$ for $A$ if and only if all the roots of all the $p_i$ are multiplicity 1 (simple roots).