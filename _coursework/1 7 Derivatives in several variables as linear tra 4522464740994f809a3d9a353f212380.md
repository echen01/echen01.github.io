# 1.7. Derivatives in several variables as linear transformations

# Chapter 1.7 Derivatives in several variables as linear transformations

The main takeaway here is that the derivative of functions in $\R^n$ can be defined as linear transformations. 

**Def 1.7.3. Partial derivatives.** Let $U$ be an open subset of $\R^n$ and $f:U\to \R$ a function. The *partial derivative* of $f$ with respect to the $i$th variable, and evaluated at $a$, is the limit

$$
D_if(a) = \lim_{h\to 0}\frac{1}{h}\begin{pmatrix}\begin{pmatrix}a_1\\\vdots\\a_i+h\\\vdots\\a_n\end{pmatrix} - f\begin{pmatrix}a_1\\\vdots\\a_i\\\vdots\\a_n\end{pmatrix}\end{pmatrix},
$$

if the limit exists. 

## The Jacobian Matrix

**Def 1.7.7. The Jacobian Matrix.** Let $U$ be an open subset of $\Reals^n$. The Jacobian Matrix of a function $f: U \mapsto \Reals^m$ is the $m\times n$ matrix composed of the $n$ partial derivatives of $f$ evaluated at $a$: 

$$
\begin{bmatrix}\mathbf{Jf(a)}\end{bmatrix} = \begin{bmatrix} 
D_1f_1(a) & \dots & D_nf_1(a)\\
\vdots & \ddots & \vdots\\
D_1f_m(a) & \dots & D_nf_m(a)
\end{bmatrix}
$$

**Prop and Def 1.7.9. The Derivative.** Let $U \subset \R^n$be an open subset and let $f: U \to \R^m$ be a mapping; let $a$ be a point in $U$. If there exists a linear transformation $L: \R^n \to \R^m$ such that

$$
\lim_{\vec{\mathbf{h}} \to \vec{\mathbf{0}}} \frac{1}{|\vec{\mathbf{h}}|} \mathbf{((f(a+\vec{h}) - f(a)) - (}L(\mathbf{\vec{h}})) = \vec{\mathbf{0}},
$$

then $f$ is differentiable at $a$, and $L$ is unique and is the derivative of $f$ at $a$, denoted $\mathbf{[Df(a)]}$. 

**Theorem.** If $f$ is differentiable at $a$, then all partial derivatives of $f$ at $a$ exist, and the matrix representing $[Df(a)]$ is $[Jf(a)]$. 

**Proof of 1.7.7**. We prove that the Jacobian Matrix is a unique linear transformation. Because the linear transformation $L$ is represented by the matrix whose $i$th column is $L(\vec{\mathbf{e_i}})$, we show that 

$$
L(\vec{\mathbf{e_i}}) = \vec{D_i}f(a),
$$

where $\vec{D_i}f(a)$ is by definition the $i$th column of the Jacobian matrix $\begin{bmatrix}\mathbf{Jf(a)}\end{bmatrix}$. Using the definition of the derivative (Def 1.7.9), we can set $\vec{\mathbf{h}} = t\mathbf{\vec{e_i}}$ and let $t$ tend to 0. 

 

$$
\lim_{t\vec{\mathbf{e_i}} \to \vec{\mathbf{0}}} \frac{1}{|t\vec{\mathbf{e_i}}|} \mathbf{((f(a+\mathnormal{t}\vec{e_i}) - f(a)) - (}L(t\mathbf{\vec{e_i}})) = \vec{\mathbf{0}},
$$

Because $|t\mathbf{\vec{e_i}}| = |t||\mathbf{\vec{e_i}}|$, and $|\mathbf{\vec{e_i}}|=1$,  $|t\mathbf{\vec{e_i}}| = |t|$. Because the value of the limit is $\vec{\mathbf{0}}$ when $t$ is positive or negative, we replace $|t|$ with $t$. 

$$
\lim_{t\vec{\mathbf{e_i}} \to \vec{\mathbf{0}}} \frac{1}{t} \mathbf{((f(a+\mathnormal{t}\vec{e_i}) - f(a)) - (}L(t\mathbf{\vec{e_i}})) = \vec{\mathbf{0}},
$$

Using the linearity of derivatives, we get that $L(t\vec{\mathbf{e_i}}) = tL(\vec{\mathbf{e_i}})$. Therefore, we can rewrite the equation as 

$$
\lim_{t\vec{\mathbf{e_i}} \to \vec{\mathbf{0}}} \frac{\mathbf{((f(a+\mathnormal{t}\vec{e_i}) - f(a))}}{t}-L(\mathbf{\vec{e_i}}) = \vec{\mathbf{0}},
$$

The first term is the partial derivative of $f(a)$. Therefore, $L(\vec{\mathbf{e_i}}) = \vec{D_i}f(a)$. In addition, an element in $\R^n$ goes to $0$ if and only if its length goes to $0$. 

**Example:**

The Jacobian of the mapping 

$$
f\dbinom{x}{y} = \dbinom{xy}{x^2 - y^2} \text{  is } \begin{bmatrix}Jf\dbinom{x}{y}\end{bmatrix} = \begin{bmatrix}y&x\\2x&-2y\end{bmatrix} 
$$

## Directional Derivatives

Directional derivatives are a generalization of partial derivatives. The partial derivative $\vec{D_i}f(a)$ describes how $f(x)$ varies as the variable $x$ moves from $a$ in the direction of the standard basis vector $\vec{\bold{e_i}}$.

The directional derivative describes how $f(x)$ varies when the variable moves in any direction $\mathbf{\vec{v}}$, i.e. when the variable $x$ moves at a constant speed from $a$ to $a+ \vec{\mathbf{v}}$ is 

$$
\lim_{h\to0}\frac{f(a+h\vec{\mathbf{v}})-f(a)}{h}
$$

**Proposition 1.7.14.** If $U \in \R^n$ is open, and $f: U \to \R^n$ is differentiable at $a \in U$, then all direction derivatives of $f$ at $a$ exist, and the direction derivative in the direction $\vec{\mathbf{v}}$ is given by the formula

$$
\lim_{h\to0}\frac{f(a+h\vec{\mathbf{v}})-f(a)}{h} = [Df(a)]\vec{\mathbf{v}}
$$

 

**Example:** Compute the derivative in the direction $\vec{\mathbf{v}} = \begin{bmatrix}1\\2\\1\end{bmatrix}$ of the function $f\begin{pmatrix}x\\y\\z\end{pmatrix} = xy\sin z$, evaluated at $a = \begin{pmatrix}1\\1\\ \pi /2 \end{pmatrix}$. 

First, we take its derivative.

$$
\begin{bmatrix}Df(x,y,z)\end{bmatrix} = \begin{bmatrix}y\sin z & x\sin z& xy \cos z\end{bmatrix}
$$

At $a$, this is $[1, 1, 0]$.  We use Prop 1.7.14 to compute the directional derivative. 

$$
[Df(x,y,z)]\vec{\mathbf{v}} = \begin{bmatrix}1,1,0\end{bmatrix}\begin{bmatrix}1\\2\\1\end{bmatrix} = 3
$$

**Example:** Let $f: \R^2 \to \R^2$ be the function $f\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}xy\\ x^2 - y^2 \end{pmatrix}$. At the point $\begin{pmatrix}1\\1\end{pmatrix}$, wat what rate is $f$ varying in the direction $\begin{bmatrix}2\\1\end{bmatrix}$? 

$$
[Df(x,y)] =  \begin{bmatrix}y & x \\2x & -2y
\end{bmatrix}
$$

$$
[Df(1,1)] =  \begin{bmatrix}1 & 1 \\2 & -2
\end{bmatrix}
$$

$$
[Df(1,1)]\vec{\mathbf{v}} =  \begin{bmatrix}1 & 1 \\2 & -2
\end{bmatrix}\begin{bmatrix}2\\1\end{bmatrix} = \begin{bmatrix}3\\2\end{bmatrix}
$$

## The Jacobian matrix: not always the right approach

Although computing derivatives with the Jacobian matrix is handy, it doesn't always work. In contrast, taking the limit as $\vec{\mathbf{h}} \to \vec{\mathbf{0}}$ always works. 

**Proposition 1.7.18** If $f$ is the function $f(A) = A^{-1}$, defined on the set of invertible matricies in $\text{Mat}(n,n)$, then $f$ is differentiable and 

$$
[Df(A)]H = -A^{-1}HA^{-1}
$$

**Proof.** For $f$ to be differentiable, it must be defined on an open subset of $\text{Mat}(n,n)$.  Because we know that the subset of invertible matrices is open,  we can take the derivative. We show that 

$$
\lim_{H\to [0]}\frac{1}{|H|}\underbrace{((A+H)^{-1}-A^{-1}}_{\text{increment to mapping}} - \underbrace{-A^{-1}HA^{-1}}_{\text{linear function of H}}) = [0]
$$

Because $H \to [0]$, we can assume that $[AH^{-1}] < 1$. Therefore, we can treat $[I A^{-1}H]$ as a sum of a geometric series. 

$$
(A+H)^{-1} = (A(I+A^{-1}H))^{-1} = (I+A^{-1}H)^{-1}A^{-1}
$$

$$
=(I-(-A^{-1}H))A^{-1}
$$

We replace $I -(-A^{-1}H)$with the infinite series $I + (-A^{-1}H )+ (-A^{-1}H )^2 + \dots$. 

$$
= (I + (-A^{-1}H )+ (-A^{-1}H )^2 + \dots)A^{-1}
$$

$$
= A^{-1}-A^{-1}HA^{-1}+((-A^{-1}H)^2+(-A^{-1}H)^3+\dots)A^{-1}
$$

Subtracting $A^{-1}-A^{-1}HA^{-1}$ from both sides, we get the increment and its linear approximation:

$$
\underbrace{((A+H)^{-1}-A^{-1})}_{f(a+h)-f(a)}-\underbrace{(-A^{-1}HA^{-1})}_{\text{linear function of increment H}}
$$

$$
=((-A^{-1}H)^2+(-A^{-1}H)^3+\dots)A^{-1}
$$

$$
=(-A^{-1}H)((-A^{-1}H)+(-A^{-1}H)^2+(-A^{-1}H)^3+\dots)A^{-1}
$$

The Cauchy-Schwarz Inequality gives us

$$
((A+H)^{-1}-A^{-1})+A^{-1}HA^{-1})\leq |A^{-1}H||(-A^{-1}H)+(-A^{-1}H)^2+(-A^{-1}H)^3+\dots||A^{-1}|
$$

And the triangle inequality gives us 

$$
\leq |A^{-1}H||A^{-1}||-A^{-1}H|+|-A^{-1}H|^2+|-A^{-1}H|^3+\dots
$$

$$
\leq |A^{-1}|^2|H|\frac{|A^{-1}||H|}{1-|A^{-1}H|}\leq \frac{|A^{-1}|^3|H|^2}{1-|A^{-1}H|}
$$

If $H$ is so small that $|A^{-1}H| < 1/2$, then 

$$
\frac{1}{1-|A^{-1}H|} \leq 2
$$

We see that

$$
\lim_{H\to [0]}\frac{1}{|H|}{((A+H)^{-1}-A^{-1}} +A^{-1}HA^{-1}) \leq\lim_{H\to [0]} \frac{1}{|H|}2|H|^2|A^{-1}|^3= [0]
$$