---
title: 6. Floating Point
layout: page
---
# 6. Floating Point

February 19, 2021

Error in math comes from representation and arithmetic. Floating points is like binary scientific notation. 

**Definition (Normal floating point number).** A normal floating point number has the form

$$
(-1)^s \times (1 + 0.b_1\dots b_p)_2 \times 2^E.
$$

The IEEE standard for 64-bit has 1 bit for the $s$, 11 bits for $E$, and $52$ for the fraction. $E \in [-1022, 1023]$.

A 32-bit number has 8 bits for the exponent and 23 for the fraction.

## Exceptions

If a number if too large, then we have overflow.

If a number is too small, then the default behavior is to return a subnormal. 

If a number is nonsensical $0/0$,  then NaN is returned.

## Round-off Errors

To represent floating points, define $fl(x)$ such that $fl(x)$ is the floating representation of $x\in \R$. Define representation error as such:

**Relative representation error**

$$
\frac{(x-fl(x))}{x}.
$$

This is actually quite small. Let $\epsilon$ be machine epsilon. 

**Corollary.** $fl(x) = x(1 + \delta)$ and $|\delta | < \epsilon$. 

**Proof.** This is the maximum relative representation error of a number.

$$
\epsilon = \frac{1}{2}(\text{nextfloat}(1.0) - 1.0)\\
 = \frac{1}{2} (1 + 2^{-52} - 1) \approx 10^{-16}.
$$

### Roundoff error in arithmetic

Let $fl(x+y)$ be the closet floating point representation of $x+y$. Assuming no overflow or underflow, basic operations have relative error less than or equal to epsilon.

$$
\hat z = fl(x + y).
\\\hat z = z(1 + \delta)
$$

## Catastrophic Cancellation

If $\hat x = x(1 + \delta)$ and $\hat y = y(1 + \delta_2$ are floating point approximations to $x$ and $y$ that are very close, then $fl(\hat x - \hat y)$ may be a poor approximation to $x-y$ due to cancellation. 

**Example.** Consider the quadratic formula where $a = 1$. The problem occurs when the smallest root is close to machine precision.

$$
\begin{gather}
x = \frac{-b\pm \sqrt{b^2-4c}}{2}\\
x = \frac{-b \pm \sqrt{b^2 - 4c}(1 + \delta_1)}{2}(1 + \delta_2)\\
 = -b -b \delta_2 + \sqrt{b^2-4c}(1 +\delta_1 + \delta_2 + \delta_1\delta_2)\\
b \approx \frac{1}{\sqrt{\epsilon}}, c \approx 1\\
\end{gather}
$$

## Error Analysis

Suppose we want to compute the sum of $n$ numbers. The error is bounded by $n \epsilon$.

$$
 \|x - \tilde x\|/ \|x\| \leq  n \epsilon.
$$

From backwards stability, $\forall x$, $\tilde f(x) = f(\tilde x)$. $\|x - \hat x\| / \|x\| = O(\epsilon)$.
