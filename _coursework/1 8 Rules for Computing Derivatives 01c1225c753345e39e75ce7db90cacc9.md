# 1.8. Rules for Computing Derivatives

# Chapter 1.8 Rules for Computing Derivatives

Many differentiation rules from single-variable calculus carry over to multiple variables. 

**Theorem 1.8.1.** Let $U \subset \R^n$ be open. 

1. If $f: U \to \R^m$ is a constant function, then $f$ is differentiable, and its derivative is $[0]$. 
2. If $f: \R^n \to \R^m$ is linear, then it is differentiable everywhere, and its derivative at all points $a$ is $f$, i.e., $[Df(a)]\vec{v} = f(\vec{v})$. 
3. If $f_1,\dots,f_m: U \to \R$ are $m$ scalar valued functions differentiable at a, then the vector-valued mapping $f = \begin{pmatrix}f_1\\ \dots \\ f_m \end{pmatrix}: U \to \R^m$ is differentiable at $a$, with derivative $[Df(a)]\vec{v} = \begin{bmatrix}[Df_1(a)\vec{v}\\\dots \\ [Df_m(a)]\vec{v}\end{bmatrix}$. Conversely, if $f$ is differentiable at $a$, each $f_i$ is differentiable at $a$, and $[Df_i(a)] = [D_1f_i(a),\dots,D_nf_i(a)]$. 
4. If $f, g: U \to \R^m$ are differentiable at $a$, then so is $f+g$, and $[D(f+g)(a)] = [Df(a)]+[Dg(a)]$. 
5. If $f: U \to \R$ and $g: U \to \R^m$ are differentiable at $a$, then so is $fg$, and the derivative is given by $[Dfg(a)]\vec v= f(a)[Dg(a)]\vec v +([Df(a)]\vec v )g(a)$. This is the product rule. 
6. If $f: U \to \R$ and $g: U \to \R^m$ are differentiable at $a$, and $f(a) \neq 0$, then so is $g/f$, and the derivative is given by $[D\begin{pmatrix}g\\f\end{pmatrix}(a)]\vec v = \frac{[Dg(a)]\vec v}{f(a)}- \frac{([Df(a)\vec v)g(a)}{(f(a))^2}$. 
7. If $f, g: U \to \R^m$ are both differentiable at $a$, then so is the dot product $f \cdot g: U \to \R$ and (as in one dimension) $[D(f\cdot g)(a)]\vec v = [Df(a)]\vec v \cdot g(a) + f(a) \cdot [Dg(a)] \vec v$. 

## The Chain Rule

If $U\subset \R^n$, $V \subset \R^m$ be open sets, let $g: U \to V$ and $f: V \to \R^p$ be mappings, and let $a$ be a point of $U$. If $g$ is differentiable at $a$ and $f$ is differentiable at $g(a)$, then the composition $f \circ g$ is differentiable at $a$, and its derivative is given by

$$
[D(f\circ g)(a)] = [Df(g(a))]\circ [Dg(a)].
$$