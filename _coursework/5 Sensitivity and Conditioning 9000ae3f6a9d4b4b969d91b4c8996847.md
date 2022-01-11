---
layout: page
title: 5. Sensitivity and Conditioning
---

# 5. Sensitivity and Conditioning

February 16, 2021

## Norms and the Neumann Series

See [1.5. Limits and Continuity](/coursework/1-5-limits-and-continuity-3b8b88f1d1314946826f16c3ec7364e4).

Suppose $F$ is a square matrix and $\|F\| <1$ in some operator norm and consider the power series

$$
\sum_{j=0}^nF^j.
$$

By the sub multiplicative property, $\|F^j\| \leq \|F\|^j$. By the triangle inequality, the partial sums satisfy

$$
(I-F)\sum_{j=0}^nF^j= I - F^{n+1}.
$$

Therefore we have that

$$
\lim_{n\to \infty} \|(I-F)\sum_{j=0}^nF^j-I\| \leq \|F\|^{n+1}= 0.
$$

Therefore, $I-F$ is invertible and $(I-F)^{-1} = \sum_{j=0}^\infty F^j$.

## Notions of Error

The art of numerics is finding an approximation that works in a certain error bound. Some sources of error are

- measurement error
- roundoff error
- modeling error
- approximation error

First, we must formally explain what error is.

**Definition (Absolute error).** Suppose $\hat x$ is an approximation of $x$. The *absolute error* is

$$
e_{abs} = |\hat x - x|.
$$

Absolute error has the same dimensions of $x$.

**Definition (Relative error).** The *relative error* is a measure with a more natural sense of scale. It is a proportion.

$$
e_{rel} = \frac{e_{abs}}{|x|}.
$$

We can also approximate the relative error by replacing $\hat x$ with $x$.

$$
\hat e_{rel} = \frac{e_{abs}}{|\hat x|}.
$$

To see how good of an estimate $\hat e_{rel}$ is, we know that if $\hat e_{rel} <1$, then

$$
\frac{\hat e_{rel}}{1+\hat e_{rel}} \leq e_{rel} \leq \frac{\hat e_{rel}}{1-\hat e_{rel}}.
$$

If $\hat e_{rel}$ is much less than 1, it is a good estimate for $e_{rel}$.

**Definition (Mixed error).** When $x=0$, relative error makes no sense because it is undefined. Another interpretation we can create is mixed error.

$$
e_{mixed} = \frac{e_{abs}}{|x| + \tau}
$$

where $\tau$ is some natural scale factor associated with $x$.

### Error for vectors

Instead of using absolute value, we can replace everything with vector norms. We can also consider the component-wise errors.

$$
e_{abs, i} = |\hat x_i - x_i|\quad e_{rel, i} = \frac{e_{abs, i}}{|x_i|}.
$$

**Theorem.** The maximum component-wise relative error can be computed as a norm-wise error in a norm defined in terms of the solution vector.

$$
\max_ie_{rel, i} = \|\text{diag}(x)^{-1}(\hat x - x)\|
$$

## Forwards and backwards error and conditioning

Often, we will want to approximate a function $f$ by another function $\hat f$. 

**Definition (forward error).** The forward error of this function approximation is

$$
|\hat f(x) - f(x)|.
$$

This is the function output.

**Definition (backward error, backward stable).** When we have a slightly wrong input, 

$$
\hat f(x) = f(\hat x),
$$

then $\| \hat x - x\|$ is the backwards error. An algorithm that always has a small backwards error is *backward stable*. 

**Definition (condition number).** The condition number is the relative error divided by the relative perturbation. It is a tight constant relating relative output error to relative input error. For example, suppose we are evaluating the function $f$ with a perturbed input $\hat x = x+h$. Note that the relative error is $\|h\|/\|x\|$. 

Let $\kappa[f(x)]$ be the smallest constant such that

$$
\frac{|f(x+h) - f(x)|}{|f(x)|}\leq \kappa [f(x)] \frac{|h|}{|x|} + o(|h|).
$$

Suppose $f$ is differentiable. The relative error is 

$$
\frac{|f(x+e) - f(x)|}{|f(x)|}.
$$

Using a Taylor expansion, $f(x + h) \approx f(x) + f'(x)h$. Therefore, the condition number is

$$
\kappa [f(x)] = \lim_{h \to 0}\frac{|f(x+h)-f(x)|/f(x)|}{|(x+h)-x|/|x|} = \frac{|f'(x)||x|}{|f(x)|}.
$$

If $f$ is Lipschitz in a neighborhood of $x$, then

$$
\kappa[f(x)] = \frac{M_{f(x)}|x|}{|f(x)|}
$$

where $M\_f$ is the smallest constant such that $\|f(x + h) - f(x)\| \leq M\_f\|h\| + o(\|h\|)$.

If there is no linear bound on a function, it has an *infinite condition number*.

A problem is *well conditioned* if it has a small conditioned number. A problem is *ill-conditioned* if it has a large condition number. 


> 💡 A backwards stable algorithm applied to a well-conditioned problem has a small forward error.


## Perturbing matrix problems

**Theorem (Condition number for matrix multiplication).** Suppose that we want to compute some matrix operation $y = Ax$ but there is some error $\hat y = (A+E)x$. The absolute error is

$$
e_{abs} = \|\hat y - y\| = \|Ex\|.
$$

The relative error is

$$
\frac{\|\hat y - y\|}{\|y\|} = \frac{\|Ex\|}{\|y\|}.
$$

Assuming $A$ is invertble,

$$
\|Ex\| = \|EA^{-1}y\| \leq \|E\|\|A^{-1}\| \|y\|.
$$

Therefore,

$$
\frac{\|\hat y - y\|}{\|y\|} \leq \|A\|\|A^{-1}\|\frac{\|E\|}{\|A\|} = \|A^{-1}\|\|E\|\\
 = \|A\|\|A^{-1}\|\frac{\|E\|}{\|A\|} = \kappa(A) \frac{\|E\|}{\|A\|}.
$$

Here, we see that $\kappa (A) = \|A\|\|A^{-1}\|$. While not quite accurate, this is called the condition number of $A$. This is also equal to $\sigma_1(A) \cdot \sigma_1(A^{-1})$.

**Claim.** $\sigma_1(A^{-1}) = 1/ \sigma_1(A)$.

**Proof.** $A = U \Sigma V^T$, so $A^{-1}  = V\Sigma^{-1}U^T$. Since $V$ and $U$ are orthonormal, the magnitude is the largest singular value.

When we make perturbations to $x$, our condition number is $\|A\| \|x\| / \|Ax\|$. This is derived from our following computation.

$$
\hat y = A(x + h)\\
e_{abs} =\| Ax+Ah - Ax = Ah\|.\\
e_{rel} = \frac{\|Ah\|}{\|Ax\|}
$$

Diving by the perturbation,

$$
\frac{\|Ah\| / \|Ax\|}{\|x+h -x\|/\|x\|} =\frac{\|Ah\|\|x\|}{\|Ax\|\|h\|} =  \frac{\|A\|\\\|x\|}{\|Ax\|}.
$$

Notice that $\|A\| = \|J\|$. This agrees with the limit definition of A's derivative as $h\to 0$.

## Dimensions and scaling

It is common that the first step of analyzing an application is *nondimensionalization*. This is the idea of combining constants to obtain a small number of dimensionless constants. One example is the gravity constant $G$.

---
