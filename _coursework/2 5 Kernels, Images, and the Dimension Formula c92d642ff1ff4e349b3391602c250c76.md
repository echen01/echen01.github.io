# 2.5. Kernels, Images, and the Dimension Formula

**Definition 1.1.5 (Subspace of** $\R^n$). A nonempty subset $V\subset \R^n$ is a *vector subspace* if it is closed under addition and closed under multiplication by scalars; that is, $V$ is a vector subspace when 

$$
\vec x, \vec y \in V \text{ and }a \in \R, \text{ then }\vec x + \vec y\in V \text{ and }a\vec x \in V. 
$$

---

Kernels are a way of speaking about the uniqueness of solutions of linear equations. Images are a way of speaking about existence of solutions of linear equations. 

**Definition 2.5.1 (Kernel and image).** Let $T: \R^n \to \R^m$ be a linear transformation.

1. The kernel of $T$, denoted $\ker T$, is the set of vectors $\vec x \in \R^n$ such that $T(\vec x) = \vec 0$.
2. The image of $T$, denoted $\text{img } T$, is the set of vectors $\vec w \in \R^m$ such that there is a vector $\vec v \in \R^n$ with $T(\vec v) = \vec w$. 

**Proposition 2.5.3.** Let $T: \R^n \to \R^m$ be a linear transformation. The system of linear equations $T(\vec x) = \vec b$ has

1. at most one solution for every $\vec b \in  \R^m$ if and only if $\ker T = \{\vec 0\}$.  ($T$ is injective).
2. at least one solution for every $\vec b \in \R^m$ if and only if $\text{img } T = \R^m$. ($T$ is surjective).

**Proof.** 

1. If the kernel of $T$ is not $\{\vec 0\}$, then there is more than one solution to $T(\vec x) = \vec 0$. We use proof by contradiction for the other direction. If there exists a $\vec b$ for which $T(\vec x) = \vec b$ has more than one solution, i.e., $T(\vec x_1) = T(\vec x_2) = \vec b$ and $\vec x_1 \neq \vec x_2$, then $T(\vec x_1 - \vec x_2) = T(\vec x_1) - T(\vec x_2) = \vec b - \vec b = \vec 0$. So $(\vec x_1 - \vec x_2)$ must be a nonzero element of the kernel and $\ker T\neq \{\vec 0\}$. 
2. This is true by definition.

## Finding bases for the image and kernel

**Theorem 2.4.5 (A basis for the image).** The pivotal columns of $[T]$ form a basis of $\text{img }  T$. 

**Example.** Consider the matrix $A$ that describes a linear transformation from $\R^5$ to $\R^4$. 

$$
A = \begin{bmatrix}1&2&4&-1&2\\-1&0&-2&-1&1\\2&0&4&2&1\\1&1&3&0&2\end{bmatrix}, \text{ which row reduces to }\tilde A = \begin{bmatrix}1&0&2&1&0\\0&1&1&-1&0\\0&0&0&0&1\\0&0&0&0&0\end{bmatrix}
$$

Because the pivotal $1$s are in the first, second, and fifth columns, these form the basis of $\text{img }A$. In other words, any vector in the image of $A$ can be uniquely expressed as a linear combination of those three vectors. 

## A basis for the kernel

Finding a basis for the kernel is much more complicated. A basis for the kernel is a set of vectors satisfying $A\vec w = \vec 0$. The basis vectors must be linearly independent. 

**Theorem 2.5.6 (A basis for the kernel).** Let $p$ be the number of nonpivotal columns of $A$, and $k_1,\dots,k_p$ their positions. For each nonpivotal column, form the vectors $\vec v_i$ satisfying $A\vec v_i = \vec 0$ and such that its $k_i$th entry is 1, and its $k_jth$ entries are all 0, for $j\neq i$. The vectors $\vec v_1, \dots, \vec v_p$ form a basis of $\ker A$. 

**Example.** Let's use the matrix $A = \begin{bmatrix}1&2&4&-1&2\\-1&0&-2&-1&1\\2&0&4&2&1\\1&1&3&0&2\end{bmatrix}$. Because it row reduces to $\begin{bmatrix}1&0&2&1&0\\0&1&1&-1&0\\0&0&0&0&1\\0&0&0&0&0\end{bmatrix}$, we know that $\vec v_3$ and $\vec v_4$ are non pivotal. Because there are non-pivotal, we can set two basis vectors where the third element of one is 1 and the fourth element of the other is 1.

$$
\vec v_1=\begin{bmatrix}-\\-\\1\\0\\-\end{bmatrix}, \vec v_2 = \begin{bmatrix}-\\-\\0\\1\\-\end{bmatrix}.
$$

Because the basis for the kernel is the set of vectors such that $A\vec x = \vec 0$, we add another row to $A$. 

$$
[A|\vec 0] = \begin{bmatrix}1&0&2&1&0&0\\0&1&1&-1&0&0\\0&0&0&0&1&0\\0&0&0&0&0&0\end{bmatrix}
$$

We get that 

$$
x_1 + 2x_3 + x_4 = 0\\
x_2+x_3-x_4 = 0\\
x_5 = 0.
$$

When $x_3=1$ and $x_4 = 0$, $\begin{bmatrix}-2\\-1\\1\\0\\0\end{bmatrix}$. When $x_3 = 0$ and $x_4 = 1$, $\begin{bmatrix}-1\\1\\0\\1\\0\end{bmatrix}$. 

## The Dimension Formula

**Theorem 2.5.8 (Dimension formula).** If $T: \R^n\to \R^m$ is a linear transformation of finite dimensional vector space, then $\dim (\ker T)+ \dim (\text{img } T )= n$, the dimension of the domain. 

**Definition 2.5.9 (Rank and nullity).** The dimension of the image of a linear transformation is called its *rank*. The dimension of the kernel of a linear transformation is called its *nullity*.

**Corollary 2.5.10 (Deducing existence from uniqueness).** Let $T:\R^n \to \R^n$ be a linear transformation. Then the equation $T(\vec x) = \vec b$ has a solution for every $\vec b \in \R^n$ if and only if the only solution to $T(\vec x) = \vec 0$ is $\vec x = \vec 0$. (i.e., if the kernel has dimension 0). 

## Interpolation and the dimension formula

The *Lagrange interpolation formula* is a formula for reconstituting a polynomial of degree at most $k$ from $k+1$ samples. We will use the dimension formula to show that such a formula must exist.   

<aside>
⚠️ Near the boundaries of a function, Lagrange interpolation is a terrible way to approximate a function.

</aside>

Let $P_k$ be the vector space of polynomials of degree $\leq k$. Given $k+1$ numbers $c_0, \dots, c_k$, can  we find a polynomial $p\in P_k$ such that 

$$
p(0) = c_0\\
\vdots\\
p(k) = c_k?
$$

Consider the linear transformation $T_k: P_k \to \R^{k+1}$ which takes a polynomial $p$ and returns the sampled values $p(0), \dots, p(k)$. 

$$
T_k(p) = \begin{bmatrix}p(0)\\\vdots\\p(k)\end{bmatrix}.
$$

The kernel of $T_k$ is the space of polynomials $p\in P_k$ that vanish at $0, 1,\dots, k$.  But a polynomial of degree $\leq k$ cannot have $k+1$ roots, unless it is the zero polynomial. So the kernel of $T_k$ is $\{0\}$. The dimension formula tells us that its image is all of $\R^{k+1}$:

$$
\dim \text{img }T_k + \dim \ker T_k = \dim P_k  = k+1.
$$

Because $\dim \ker T_k$ is $\{\vec 0\}$, $T_k$ is injective. Since the domain and codomain of $T_k$ are both of dimension $k+1$, by Corollary 2.5.10, $T_k$ is surjective. Thus $T_k$ is invertible. 

Consequently, there is a transformation $T_k^{-1}: \R^{k+1} \to P_k$ such that for the sampled values $c_0,\dots,c_k$, we have

$$
T_k^{-1}\begin{pmatrix}\begin{bmatrix}c_0\\\vdots\\c_k\end{bmatrix}\end{pmatrix} = \underbrace{a_0 + a_1x + \dots + a_kx^k}_{p(x)},
$$

where $p$ is the unique polynomial with $p(0) = c_0, p(1) = c_1, \dots, p(k) = c_k$. 

## Partial fractions

**Proposition 2.5.13 (Partial fractions).** Let 

$$
p(x) = (x-a_1)^{n_1}\dots (a-a_k)^{n_k},
$$

be a polynomial of degree $n = n_1 + \dots + n_k$ with the $a_i$ distinct and let $q$ be any polynomial of degree $< n$. Then the rational function $q/p$ can be written as a sum of simplier terms, called partial fractions: 

$$
\frac{q(x)}{p(x)} = \frac{q_1(x)}{(x-a_1)^{n_1}}+ \dots + \frac{q_k(x)}{(x-a_k)^{n_k}},
$$

with each $q_i$ a polynomial of degree $< n_i$. 

**Proof.** Proposition 2.5.13 assures that there will always be a solution to the resulting matrix from partial fractions. First, the matrix would be $n\times n$. This gives a linear transformation that has as its input a vector whose entries are the coefficients $q_1,\dots q_k$. There are $n$ coefficients, so each polynomial has degree $< n_i$. Its output is a vector giving the $n$ coefficients of $q$. We can think of this as the linear transformation $T: \R^n \to \R^n$. By corollary 2.5.10, proposition 2.5.13 is true if and only if the solution of $T(q_1,\dots, q_k) = \vec 0$ is $q_1,\dots q_k = \vec 0$.

**Example.** When $q(x) = 2x+3$ and $p(x) = x^2 - 1$, there exists polynomials $q_1$ and $q_2$ of degree less than 1 such that

$$
\frac{2x+3}{x^2-1}= \frac{A_0}{x+1}+\frac{B_0}{x-1}.
$$

We use this equation to create a system of equations. 

$$
\frac{2x+3}{x^2-1}= \frac{A_0(x-1)+B_0(x+1)}{(x+1)(x-1)}
$$

$$
= \frac{(A_0+B_0)x+B_0-A_0}{(x+1)(x-1)}
$$

We see that $A_0 + B_0 = 2$ and $B_0 - A_0 = 3$. We can use matrices to solve it. We get that $B_0 = 5/2$ and $A_0 = -1/2$. 

- **Practice.** If $q(x) = x^3 -1$ and $p(x)=(x+1)^2(x-1)^2$, then what is $q/p$?
    
    $$
    \frac{x^3-1}{(x+1)^2(x-1)^2} = \frac{A_1x + A_0}{(x+1)^2} + \frac{B_1x+B_0}{(x-1)^2}
    $$
    
    $$
    x^3-1 = (A_1x+A_0)(x^2-2x+1) + (B_1x+B_0)(x^2+2x+1)
    $$
    
    $$
    x^3-1 = A_1x^3-2A_1x^2+A_1x+A_0x^2-2A_0x+A_0+B_1x^3+2B_1x^2+B_1x+B_0x^2+2B_0x+B_0
    $$
    
    $$
    x^3-1 = (A_1+B_1)x^3+(-2A_1+A_0+2B_1+B_0)x^2+(A_1-2A_0+B_1+2B_0)x+(A_0+B_0)
    $$
    
    $$
    \begin{bmatrix}
    1&0&1&0 & 1\\
    -2&1&2&1&0\\
    1&-2&1&2&0\\
    0&1&0&1&-1
    
    \end{bmatrix} = 
    \begin{bmatrix}
    1&0&0&0 & 1/4\\
    0&1&0&0&-1/4\\
    0&0&1&0&3/4\\
    0&0&0&1&-3/4
    
    \end{bmatrix} .
    $$