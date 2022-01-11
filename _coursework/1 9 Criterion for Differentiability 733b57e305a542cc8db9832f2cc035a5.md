# 1.9. Criterion for Differentiability

# Chapter 1.9 The MVT and Criteria for Differentiability

## The Mean Value Theorem

**Theorem 1.9.1.** Let $U \in \R^n$ be an open, let $f: U \to \R$ be differentiable, and let the segment $[a,b]$ joining $a$ to $b$ be contained in $U$. Then there exists $c_0 \in [a,b]$ such that

$$
f(b) -f(a) = [Df(c_0)](\vec{b-a}).
$$

## Differentiability and Pathological Functions

It is possible for all partial derivatives of $f$ to exist, and even all directional derivatives, and yet $f$ to not be differentiable. In this case, the Jacobian exists but does not represent the derivative. 

**Example:** 

$$
f\begin{pmatrix}x\\y\end{pmatrix}\begin{cases}
 \frac{x^2y}{x^2+y^2} & \text{if }\begin{pmatrix} x\\y\end{pmatrix}\neq \begin{pmatrix}0\\0\end{pmatrix}\\
0 &\text{if }\begin{pmatrix} x\\y\end{pmatrix}= \begin{pmatrix}0\\0\end{pmatrix}
\end{cases}
$$

This is indeed continuous at the origin. First we prove that all directional derivatives exist. 

Let $\vec v = \begin{bmatrix}x\\y\end{bmatrix}$. Recall from Chapter 7 that the directional derivative is equal to 

$$
\lim_{h\to 0}\frac{f(a+h\vec v)-f(a)}{h}
$$

For $f$, we get 

$$
\lim_{h\to 0}\frac{f\begin{pmatrix}hx\\hy\end{pmatrix} - f\begin{pmatrix}0\\0\end{pmatrix}}{h}
$$

$$
=\lim_{h\to 0}\frac{1}{h}\frac{h^3x^2y}{h^2(x^2+y^2)}
$$

$$
= \frac{x^2y}{x^2+y^2}
$$

However, the total derivative does not exist. The derivatives in different directions are not equal. For example, take the derivative in the direction of vector $\begin{bmatrix}1\\1\end{bmatrix}$. 

$$
\lim_{t\to 0}\frac{f\begin{pmatrix}\begin{pmatrix}0\\0\end{pmatrix}+t\begin{bmatrix}1\\1\end{bmatrix}\end{pmatrix}-f\begin{pmatrix}0\\0\end{pmatrix}}{t}= \lim_{t\to 0}\frac{t^3}{2t^3} = \frac{1}{2}
$$

This is not equal to the Jacobian matrix times the vector $\begin{bmatrix}1\\1\end{bmatrix}$. 

$$
\begin{bmatrix}D_1f\begin{pmatrix}0\\0\end{pmatrix},D_2f\begin{pmatrix}0\\0\end{pmatrix}\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix}0&0\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix}=0
$$

In this example, all the directional derivatives exists, but there is no tangent plane approximation. 

**Def 1.9.6. Continuously differentiable function.** A function is continuously differentiable on $U \subset \R^n$ if all its partial derivatives exist and are continuous on $U$. Such a function is known as a $C^1$ function. 

**Def 1.9.7. A $C^p$ function** on $U \subset \R^n$ is a function that is $p$ times continuously differentiable: all of its partial derivative up to order $p$ exist and are continuous on $U$. 

### Criterion for Differentiability

**Theorem 1.9.8 (Criterion for differentiability).** If $U$ is an open subset of $\R^n$, and $f: U \to \R^m$ is a $C^1$ mapping, then $f$ is differentiable on $U$, and its derivative is given by its Jacobian matrix.

This ensures that a function is also not pathological.