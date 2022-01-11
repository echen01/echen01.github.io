# 2.6. Abstract Vector Spaces

A vector space is a set in which elements can be added and multiplied by scalars. $\text{Mat(n,m)}$ is an example of an abstract vector space. So is the space of polynomials. 

**Definition 2.6.1 (Vector space).**  A *vector space* $V$ is a set of vectors such that two vectors can be added to form another vector in $V$, and a vector can be multiplied by a scalar to form another vector in $V$. 

1. *Additive identity.* There exists a vector $0 \in V$ such that for any $v\in V$, we have $0+v = v$.
2. *Additive inverse.* For any $v\in V$, tehre exists a vector $-v \in V$ such that $v + (-v) = 0$. 
3. *Communative law for addition.* For all $v,w\in V$, we have $v+w = w+v$. 
4. *Associative law for addition.* For all $v_1, v_2, v_3 \in V$, $v_1 + (v_2+ v_3) = (v_1 + v_2) + v_3$. 
5. *Multiplicative identity*. For all $v\in V$, we have $1v = v$. 
6. *Associative law for multiplication.* 
7. *Distributive law for scalar addition.*
8. *Distributive law for vector addition.*

**Definition 2.6.12 (Concrete to abstract function).** Let $\{v\} = v_1,\dots, v_n$ be a finite, ordered collection of $n$ vectors in a vector space $V$. The *concrete to abstract* function $\Phi_{\{v\}}$ is the linear transformation $\Phi_{\{v\}}: \R^n \to V$ that translates $\R^n$ to $V$:

$$
\Phi_{\{v\}} (\vec a) = \Phi_{\{v\}}\begin{pmatrix}\begin{bmatrix}a_1\\\vdots \\ a_n\end{bmatrix}\end{pmatrix} = a_1v_1 + \dots + a_nv_n.
$$

$\vec v_1, \dots \vec v_n$ are linearly independent if and only if $\Phi_{\{v\}}$ is injective.

$\vec v_1, \dots, \vec v_n$ span $V$ if and only if $\Phi_{\{v\}}$ is surjective.

$\vec v_1, \dots, \vec v_n$ are a basis if and only if $\Phi_{\{v\}}$ is bijective. 

## Matrix with respect to a basis and change of basis

**Definition 2.6.16 (Matrix with respect to bases).** Let $V$ and $W$ be finite-dimensional vector spaces, with $\{v\} = v_1,\dots, v_n$ a basis of $B$ and $\{w\} = w_1, \dots, w_m$ a basis of $W$. Let $T: V \to W$ be a linear transformation. The matrix $[T]_{\{v\},\{w\}}$ of $T$ with respect to the bases $\{v\}$ and $\{w\}$ is the $m\times n$ matrix with entries $t_{i,j}$ where

$$
T(v_k) = \sum_{l=1}^m t_{l,k}w_l.
$$

Alternatively, it is the matrix of operation 

$$
[T]_{\{v\},\{w\}} = \Phi_{w}^{-1} \circ T \circ \Phi_{v}: \R^n \to \R^m.
$$

**Definition 2.6.17 (Change of basis matrix).** Let $V$ and $W$ be $n$-dimensional vector spaces. Given the bases $\{v\}, \{v'\}$ of $V$ and $\{w\}, \{w'\}$ of $W$, we can express the basis $[T]_{\{v\},\{w\}}$ in terms of $[T]_{\{v'\},\{w'\}}$:

$$
[T]_{\{v\}, \{w\}} = \Phi^{-1}_{\{w\}}\circ \Phi_{\{w'\}} \circ [T]_{\{v'\},\{w'\}}\circ \Phi^{-1}_{\{v'\}}\circ \Phi_{\{v\}}.
$$

**Bryan's Example.** Let $P^3$ be the vector space of polynomials of degree $\leq 3$. The basis is $(1, x, x^2, x^3)$. However, these vectors are not orthogonal. What does orthogonal even mean in this vector space? We can define the inner product of two vectors as something like $<p,q> = \int_0^1 p(x)q(x)dx$.

This is really important in fields like physics and chemistry. Hermite polynomials for example are a family of orthogonal polynomials. They are used in computer graphics! We will not discuss this now.