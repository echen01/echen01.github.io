---
layout: page
title: 2.9. Superconvergence
---

# 2.9. Superconvergence

If Newton's method slow converges, in base 2 Newton's method is giving you an extra bit at each iteration. That's terrible. Newton's method,when it converges, usually doubles the number of correct digits at each iteration. 

**Definition 2.9.2 (Superconvergence).** Set
$x\_i=\|a\_{i+1}-a\_i\|$. The sequence $a\_0, a\_1, \dots$ *supercoverges* if, when the $x\_i$ are written as base 2, then each number $x_i$ starts with $2^i - 1\approx 2^i$ zeros. 

**Theorem. 2.9.4 (Newton's method superconverges).** Set

$$
k = |\vec f(a_0)||[D\vec f(a_0)]^{-1}|^2 M < \frac{1}{2}\\
c = \frac{1-k}{1-2k}|[D\vec f(a_0)]^{-1}|\frac{M}{2}.
$$

If $\|\vec h\_n\| \leq \frac{1}{2c}$, then $\|\vec h\_{n+m}\| \leq \frac{1}{c} \cdot \Big(\frac{1}{2}\Big)^{2^m}$.  Since $\vec h\_n = \|a\_{n+1} - a\_n\|$, starting at step $n$ and using Newton's method for $m$ iterations causes the distance between $a_n$ and $a_{n+m}$ to shrink to practically nothing before our eyes; if $m=10$, 

$$
|\vec h_{n+m} |\leq \frac{1}{c}\cdot \Big(\frac{1}{2}\Big)^{1024}.
$$

## Kantorovich's theorem: a stronger version

**Definition 2.9.6 (The norm of a linear transformation).** Let $A: \R^n \to \R^m$ be a linear transformation. The *norm* $\|A\|$ of $A$ is 

$$
\|A\| = \sup|A\vec x|, \text{ when } \vec x\in \R^n \text{ and }|\vec x| = 1. 
$$

Note that it is always true that $\\|A\\|\leq\|A\|$. This is why using the norm rather than using the length makes Kantorovich's theorem stronger. 

**Theorem 2.9.8 (A stronger version of Kantorovich's theorem).** Kantorovich's theorem 2.8.13 still holds if you replace both lengths of matrices by norm of matrices: $\|[D\vec f(u\_1)] - [D\vec f(u\_2)]\|$ is replaced by $\\|[D\vec f(u\_1)] - [D\vec f(u\_2)]\\|$ and $\|[D\vec f(a\_0)]^{-1}\|^2$ by $\\|[D\vec f(a\_0)]^{-1}\\|^2$. 

**Proof.** Because the triangle inequality and Cauchy Schwartz also hold for norms of matrices, Kantorovich's theorem still holds. 

**Example 2.9.9 (Norm of a matrix is harder to compute).** The length
of a matrix is easy to compute. For $A =
\begin{bmatrix}1&1\\\0&1\end{bmatrix}$, it is $\sqrt{3}$. To compute the norm, let us parametricize $A\vec x$. 

$$
\begin{bmatrix}1&1\\0&1\end{bmatrix}\begin{bmatrix}\cos t\\\sin t\end{bmatrix} = \begin{bmatrix}
\cos t + \sin t\\\sin t
\end{bmatrix}.
$$

To compute the norm, we must calculate $\sup \sqrt{(\cos t + \sin t)^2 + \sin^2 t}$. 

We need to find the maximum of this function by seeing where the derivative vanishes. 

$$
\frac{d}{dt} (\cos t + \sin t)^2 + \sin^2 t = 0\\
= 2\cos 2t + \sin 2t = 0\\
2t = \arctan (-2).
$$

Because there are two values of $t$ that satisfy this equation, we choose the one that maximizes it. We find that 

$$
\cos2t_1 = -\frac{1}{\sqrt{5}} , \sin2t_1 = \frac{2}{\sqrt{5}}.
$$

After some computation, we get that 

$$
\Big|\begin{bmatrix}\cos t_1 + \sin t_1 \\\sin t_1\end{bmatrix}\Big|^2 = \frac{3+\sqrt{5}}{2}.
$$

Finally, we find that

$$
\|A\| = \Big \|\begin{bmatrix}1&1\\0&1\end{bmatrix}\Big\| = \sqrt{\frac{3+\sqrt{5}}{2}} = \sqrt{\frac{6+2\sqrt{5}}{4}} = \frac{1+\sqrt{5}}{4}.
$$

**Remark.** For $2\times 2$ matricies, there is a nifty formula for computing the norm. 

$$
\|A\| = \sqrt{\frac{|A|^2+\sqrt{|A|^4-4(\det A)^2}}{2}}
$$

**Example 2.9.10 (Using the norm in Newton's method).** Suppose we want to find a $2\times 2$ matrix $A$ such that $A^2 = \begin{bmatrix}8&1\\\\-1&10\end{bmatrix}$, and try to solve it by Newton's method. 

$$
F(A) = A^2 - \begin{bmatrix}8&1\\\\-1&10\end{bmatrix}.
$$

Let's start at $A_0 = \begin{bmatrix} 3 & 0\\\\0&3\end{bmatrix}$. We need to see whether the Kantorovish inequality is satisfied: 

$$
|F(A_0)|\cdot M\|[DF(A_0)]^{-1}\|^2 \leq \frac{1}{2}.
$$

First we compute the derivative. 

$$
[DF(A)]B = AB + BA.
$$

The following computation shows that $A \mapsto [DF(A)]$ is Lipschitz with respect to the norm, with Lipschitz ratio 2 on all of $\text{Mat}(2,2)$. 

$$
\begin{gather}
\|DF(A\_1) - DF(A\_2)]\| = \sup\_{\|B\| = 1} \Big \| \Big([DF(A\_1)] - [DF(A\_2)]\Big)B\Big\|\\\\
 =\sup\_{\|B\| = 1} \|A\_1B + BA\_1 - A\_2B - BA\_2\| = \sup\_{\|B\|= 1} |(A_1-A_2)B + B(A_1-A_2)|,\\\\
\leq \sup_{|B|=1}|(A_1-A_2)B| + |B(A_1-A_2)|\\\\
\leq \sup_{|B|=1} |A_1-A_2||B| + |B||A_1-A_2|\\\\
\leq \sup_{|B| =1}2|B||A_1-A_2| = 2|A_1-A_2|. 
\end{gather}
$$

Now we compute $\|F(A_0)\|$. 

$$
F(A_0) = \begin{bmatrix}9&0\\0&9\end{bmatrix} - \begin{bmatrix}8&1\\-1&10\end{bmatrix} = \begin{bmatrix}1&-1\\1&-1\end{bmatrix},
$$

so $\|F(A_0)\| = 2$. 

Last we compute $\\|DF(A\_0)]^{-1}\\|^2$. Because $A_0 = 3I$, 

$$
[DF(A_0)]B = 3B + 3B = 6B.
$$

This gives us that $[DF(A_0)]^{-1}B = \frac{B}{6}$. 

$$
\begin{gather}
\|[DF(A_0)]^{-1}\| = \sup_{|B| = 1} |B/6| = \sup_{|B| = 1} \frac{|B|}{6} = 1/6. \\\\
\|[DF(A_0)]^{-1}\|^2 = \frac{1}{36}.
\end{gather}
$$

Plugging these values into the inequality, we find that $2\times 2 \times 1/36 = 1/9 < 1/2$. We can compute the square root starting at $3I$.
