# 1.6. Five Big Theorems

**Compact Set.** A nonempty subset $C \subset \R^n$ is compact if it is closed and bounded. 

## Bolzano-Weierstrass Theorem

**Theorem.** A compact set $C \subset \R^n$ contains a sequence $i \mapsto x_i$, if-and-only-if that sequence has a convergent subsequence $j \mapsto (x_i)_j$ whose limit is in $C$. 

If $i \mapsto x_i$ is a sequence in $C$, then it is bounded so it has a convergent subsequence, and it is closed so the limit is in $C$. 

If $C$ is unbounded, it contains a sequence $i \mapsto x_i \to \infty$, and that sequence has no convergent subsequence. There exists a sequence $i \mapsto x_i$ which converges in $\R^n$ to a point not in $C$.

## Existence of Maxima and Minima

**Theorem.** Let $X \subset \R^n$ be compact, and $f: X \to \R$ be continuous. There exists $x_\circ \in X$ such that $\forall x \in X, \space f(x) \leq f(x_\circ)$. In particular, $f$ is bounded.

**Proof.**  We use proof by contradiction. 

1. Say that $f$ is unbounded. There exists a sequence $n \mapsto x_n \in X$ such that $f(x_n) > n$. 
2. There exists a subsequence $(x_n)_i$ which converges to a point $y$ of $X$. 
3. Since $f$ is continuous at $y$,   $\forall \varepsilon > 0, \exists \space \delta > 0$ such that $|z-y| < \delta \implies |f(z) - f(y)| < \varepsilon$. 
4. Take $\varepsilon = 1$. Find such a $\delta$. $|z-y| < \delta \implies f(z)  < 1 + f(y)$.  Since $n \mapsto x_n$ converges to $y$, for $i$ sufficiently large, $|(x_n)_i - y |< \delta$. If $n_i > f(y) + 1$, then this is a contradiction. It is both smaller and bigger than $f(y) + 1$. 
5. Now that we know $f$ is bounded, let $M = \sup f(x)$. There must exist a sequence $n \mapsto x_n$ such that $\lim_{n \to \infty}f(x_n) = M$
6. Since $X$ is compact, there is a convergent subsequence $i \mapsto (x_n)_i$ converging to $y \in X$. 
7. $\forall \varepsilon > 0, \space \exists \delta > 0$ such that $|z-y| < \delta \implies |f(z) - f(y)| < \varepsilon$.  $\exists I$ such that $i > I \implies |(x_n)_i-y| < \delta \implies |f((x_n)_i) - f(y)| < \varepsilon$.
8. $\lim_{i \to \infty} f((x_n)_i) =f(y)= M$  because $f$ is continuous. 

## Mean Value Theorem

**Theorem.** If $f: [a,b] \to \R$ is differentiable, then there exists $c \in (a,b)$ such that $f'(c) = \frac{f(b) - f(a)}{b-a}$. 

<aside>
💡 The proof does not assume that $x\to f'(x)$ is continuous. There exists functions that follow the mean value theorem where their derivatives are not continuous.

</aside>

**Example:** 

$$
f(x) = \begin{cases}x^2 \sin\frac{1}{x} & \text{if }x \neq 0 \\
0 & \text{if }x = 0
\end{cases}
$$

Taking the derivative, 

$$
f'(x) = \begin{cases}2x\sin\frac{1}{x} -\cos\frac{1}{x} & \text{if } x \neq 0\\0 & \text{if } x =0\end{cases}
$$

This is terribly discontinuous. 

If a derivative of a differentiable single-variable function is discontinuous, it will always look erratically sinusoidal. Point discontinuities will never exist.  

**Proof.** 

**Lemma.** If $f$ is differentiable at $x \in \R$ and has a local maximum or minimum at $x$, then $f'(x) = 0$. 

**Proof of Lemma.** 

$$
f'(x) = \lim_{h \to 0}\frac{f(x+h) - f(h)}{h} \text{ if } h \neq 0
$$

If $h > 0$, then 

$$
\lim_{h \to 0}\frac{f(x+h) - f(h)}{h}  \leq 0
$$

If $h < 0$, then 

$$
\lim_{h \to 0}\frac{f(x+h) - f(h)}{h}  \geq 0
$$

The limit can be a limit of both positive and negative numbers only if $f'(x) = 0$. 

Let $g(x) = f(x) - \frac{f(b) - f(a)}{b-a}(x-a)-f(a)$. Because $f$ is continuous, $g$ is continuous. $g(a) = g(b) =0$. $g$ achieves its maximum or minimum at a point of $c \in (a,b)$. By the lemma, $g'(c) = 0 = f'(c) - \frac{f(b)-f(a)}{b-a}$. The theorem is proved. 

## The Fundamental Theorem of Algebra

**Theorem.** If $p(z)$ is a polynomial of degree $d \geq 1$, then there exists $z_\circ \in \Complex$ such that $P(z_\circ)=0$. 

D 'Alembert, Lagrange, Gauss all tried to prove it. There exists several proofs, all of which use topology somewhere. The general idea is that if $|z|$ is big, big powers dominate small powers. If $|z|$ is small, small powers dominate big powers. $10^3 >> 10$ and $\frac{1}{10}^3 << \frac{1}{10}$. 

 **Proof.** $z\mapsto |p(z)|$ is continuous because $p$ is a polynomial.

1. There exists $R$ such that $\inf_{|z|\leq R} |p(z)| = \inf_{z\in \Complex} |p(z)|$. In particular, since $\{z \space |\space  |z|\leq R\}$ is compact, there exists $z_0$ such that $\forall z \in \Complex, |p(z_0)| \leq |{p(z)}|$. This is also true for real polynomials.  
2. For some $R$, suppose $|z| > R$, and $R > 1$.  By the triangle inequality,

$$
|z^n+a_{n-1}z^{n-1}+\dots+a_0| \geq |z|^n-|a_{n-1}z^{n-1}+\dots+a_0|\\
\geq |z|^n - (a_{n-1}|z|^{n-1}a_{n-2}|z|^{n-1}+\dots+a_0|z|^{n-1})\\
\geq |z|^{n-1}(|z| - (|a_{n-1}|+\dots+|a_0|))\\
\geq |z|-(|a_n-1|+\dots+|a_0|) \geq a_0\\
R>|a_n-1|+\dots+|a_0|
$$

If $(z) > R = |a_{n=1}|+\dots|a_1|+|a_0|$, then $|p(z)| > a_0 = p(0)$. So $\inf_{|z| > R} p(z) > p(0) > \inf_{|z| \leq R}|p(z)|$.

Therefore, there exists $z_0 \in \Complex$, with $|z_0| \leq R$ such that $\forall z \in \Complex, |p(z_0) \leq |p(z)|$.  

The polynomial achieves its global minimum on a compact set $\{z\in \Complex \space | \space |z| \leq R\}$. 

### Dog on a Leash

$p(z_0) = 0$. $p(z_0+u) = q(u)$. $q$ is a polynomial of degree $n$. 

For example, suppose

$$
p(z) = z^2 + z + 1, \space z_0 =1.\\\space p(z_0+u) = (1+u)^2 + 1 + u + 1 = \underbrace{u^2 +3u + 3}_{q(u)}
$$

We set up our analogy as follows,

$$
q(u) = \underbrace{\underbrace{u^k+\dots}_{\text{leash}}+\underbrace{+b_{j+1}u^{j+1}+b_ju^j+\underbrace{b_0}_{\text{flag pole}}}_{\text{man}}}_{\text{dog}}
$$

Here, $b_0 = p(z_0)$. We need to show that $b_0 = 0$. By contradiction, suppose $b_0 \neq 0$, I will find $u$ such that $|q(u)| < |b_0|$. When $u$ goes in a circle, $q(u)$ also moves in a circle. The man takes a walk of radius $\rho =|u|$ around $b_0$. At some point during this walk, the man is exactly between the origin and the flag pole. 

If I choose $\rho>0$ sufficiently small, then there is an instance when the leash is shorter than the distance ($|b_j|\rho^j$) from the man to the flagpole. 

**Proof.** Let $\rho = |u|$. Set $B = \max\{|b_{j+1}|,\dots,b|_k|\}$ and choose $\rho$ satisfying

$$
\rho < \min\begin{Bmatrix}\frac{|b_j|}{(k-j)B}, \lvert\frac{b_0}{b_j}\rvert^{1/j},1\end{Bmatrix}.
$$

$$
|b_k|\rho^k > |b_{k+1}z^{k+1}+\dots+z^n|\\
|b_{k+1}z^{k+1} + \dots + z^n| \leq |b_{k+1}|\rho^{k+1}+\dots \rho^n
$$

If $\rho < 1$, 

$$
\leq |b_{k+1}|\rho^{k+1}+|b_{k+2}|\rho^{k+1} + \dots + \rho^{k+1}\\
\leq \rho^{k+1}(|b_{k+1}|+|b_{k+2}|+\dots1)\\
\stackrel{?}{<}\rho^k|b_k|\\
|b_k| > \rho(|b_{k+1}|+\dots + 1)\\
\rho<\frac{|b_k|}{|b_{k+1}|+\dots + 1}
$$

### Second Part of the Fundamental Theorem of Algebra

For $p(z)$, there exists $x_1,\dots, x_n \in \Complex$ such that $p(z) = (z-x_1)(z-x_2)\dots (z-x_n)$.

Suppose $p(x_1) = 0$. 

$$
p(z) = q(z)(z-x_1)+r(z)\text{, and }\deg r < 1\\ 
0=p(x_1)=q(x_1)(0)+r(x_1) \text{ so } r= 0\\
\deg q(z) = n-1 \space \space \space q(z)=(z-x_2)\dots(z-x_n)
$$

Inductively, we can find each root of $p(z)$. 

### Gauss's Proof

Complex numbers were not accepted at the time! Every real polynomial can be factored as a product of linear and quadratic factors. 

If $p$ is a real polynomial and $p(a+ib) = 0$, then $p(a-ib) = 0$, so $p(z)$ is divisible by $(z-(a+ib))(z-(a-ib)) = z^2-2az+(a^2+b^2)$.