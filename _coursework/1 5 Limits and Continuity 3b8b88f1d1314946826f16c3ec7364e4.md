---
layout: page
title: 1.5. Limits and Continuity
---
# 1.5. Limits and Continuity

**Definition 1.5.1 (Open ball).** For any $x\in \R^n$ and any $r>0$, the *open ball* of radius $r$ around $x$ is the subset 

$$
B_r(x) = \{y\in \R^n \text{ such that } |x-y|< r\}.
$$

Note that $\|x-y\|$ must be less than $r$ for the ball to be open; it cannot be = $r$. In dimension 1, a ball is an interval.

**Definition 1.5.2 (Open set of** $\R^n$). A subset $U\subset \R^n$ is *open* if for every point $x\in U$, there exists $r>0$ such that the open ball $B_r(x)$ is contained in $U$.

# Geometric Series of Matrices

Let $A$ be a square matrix. If $\|A\| < 1$, the series $S =  I+A+A^2+\dots$ converges to $(I-A)^{-1}$. 

**Proof.** 

$$
S_k = I+A+A^2+\dots+A^k\\
S_kA = A+A^2+\dots+A^k+A^{k+1}
$$

By substracting $S_kA$ from $S_k$, we get that $S_k-S_kA =S_k(I-A)= I-A^{k+1}$. By the Cauchy-Schwarz inequality, we know that $\|A^{k+1}\| \leq \|A\|^k\|A\|=\|A\|^{k+1}$. This tells us that the series of numbers $\sum_{k=1}^{\infty} \|A\|^k$ converges when $\|A\| < 1$, hence, $S$ converges. Thus, 

$$
S(I-A)=\lim_{k\to \infty}S_k(I-A) = \lim_{k\to\infty}(I-A^{k+1})
$$

$$
=I-\lim_{k\to\infty}A^{k+1} = I
$$

This equation shows that the inverse of $I-A$ is $S$. 

We now have to prove that the set of invertible $n\times n$ matrices is open. 

**Proof.** Suppose $B$ is invertible, and $\|H\| < 1/\|B^{-1}\|$. Then, $\|-B^{-1}H\| < 1$, so $I+B^{-1}H$ is invertible, and

$$
(I+B^{-1}H)^{-1}B^{-1} = (B(I+B^{-1}H))^{-1}=(B+H)^{-1}.
$$

Thus if $\|H\| < 1/\|B^{-1}\|$, the matrix $B+H$ is invertible, giving an explicit neighborhood of $B$ made up of invertible matrices. 

### Problem 1.5.23

A. Let $A$ be an $n \times n$ matrix. What does it mean to say that 

$$
\lim_{B\to A}(A-B)^{-1}(A^2-B^2) =C
$$

exists? 

It means that $\forall \varepsilon > 0$, $\exists \delta > 0$ such that $\forall x \in X$, $\|x-x_0\| < \delta \implies \|f(x) -a\| < \varepsilon$. 

Let $X=A-B$. 

$$
\lim_{\|X\| \to 0}X^{-1}(A^2-(A-X)^2)
$$

$$
=\lim_{\|X\| \to 0}X^{-1}(A^2-A^2+AX+XA-X^2)
$$

$$
=\lim_{\|X\|\to0}X^{-1}(AX+XA-X^2)
$$

$$
\lim_{\|X\|\to 0}+X^{-1}AX+X^{-1}XA-X^{-1}X^2
$$

$$
=\lim_{\|X\|\to 0}X^{-1}AX+A
$$

If $AX$ is communicative, then the limit is $2A$. Therefore, if $AB = BA$, then the limit exists. 

B. Does the limit exist when $A = \begin{bmatrix}1&0\\\\0&1\end{bmatrix}$? 

Yes. Because $IA = AI$, the limit exists and is $2I$. 

C. Does the limit exist when $A=\begin{bmatrix}0&1\\\\1&0\end{bmatrix}$? 

Yes. To illustrate, let $B=\begin{bmatrix}a&b\\\\c&d\end{bmatrix}$. $X= \begin{bmatrix}-a&1-b\\\\1-c&-d\end{bmatrix}$.
