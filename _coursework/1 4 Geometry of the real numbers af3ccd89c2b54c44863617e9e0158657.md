---
layout: page
title: 1.4. Geometry of the real numbers
---

# 1.4. Geometry of the real numbers

# Geometry of $\R^n$

A **subspace** of $\R^n$ is a subset of $\R^n$ such that it is closed under addition and multiplication by scalars. That is, $V \subset \R^n$ is a vector subspace if $\vec x, \vec y \in V$ and $a\in\R$, then $\vec x + \vec  y \in V$ and $a\vec x \in V$. 

In $\R^2$, this includes

1. Lines through the origin.
2. The set of the zero vector $\{\vec 0\}$.
3.  $\R^2$ itself.

The **dot product** of two vectors $\vec x \cdot \vec y$ is $\sum_{i=1}^n x_i y_i$.

$$
\frac{\vec x \cdot \vec y}{|\vec x||\vec y|} = \cos \theta.
$$

The **cross product** in $\R^3$ of two vectors $\vec x \times \vec y$ is 

$$
\begin{bmatrix}
\det \begin{bmatrix}a_2 & b_2\\a_3&b_3\end{bmatrix}\\
-\det\begin{bmatrix}a_1&b_1\\a_3&b_3\end{bmatrix}\\
\det\begin{bmatrix}a_1&b_1\\a_2&b_2\end{bmatrix}
\end{bmatrix}
$$

The **Cauchy-Schwarz Inequality** proves that for two vectors, $\|\vec x \cdot \vec y\| \leq \|\vec x\|\|\vec y\|$.

It follows that for matrices, $\|AB\| \leq \|A\|\|B\|$. 

The **triangle inequality** proves that $\|a+b\| \leq \|a\| - \|b\|$ since,

$$
|a| = |a+b-b|\\
\leq |a+b|+|-b|\\
=|a+b|+|b|.
$$

Generalized, this is,

$$
\left |\sum_{i=0}^{\infty}x_i\right |\leq \sum_{i=0}^{\infty}|x_i|
$$

if $\sum_{i=0}^{\infty} x_i$ converges. This works because the absolute value function is continuous.
