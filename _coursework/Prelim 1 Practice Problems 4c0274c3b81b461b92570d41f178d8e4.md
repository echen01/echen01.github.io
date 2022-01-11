# Prelim 1 Practice Problems

# Textbook

### Problem 1.6.10

![Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled.png](Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled.png)

- My Solution
    
    First, choose $z_0 = 0$ and $q(u) = u^8 + u^4 + u^2+1$. Flag pole is $1$. The position of the man is $u^2 + 1$. The leash is $u^8 + u^4$. $b_j = 1$. Now, we take $B= \max\{1,1,1,1\} = 1$. We take a $|u|=\rho$ and $\rho$ less than $\min\{\frac{1}{(8-2)(1)},  \frac{1}{1}^{1/2}, 1\}$. Our minimum must occur when $u < 1/6$. 
    
    For example, let's try $u = 1/10(1+i)$. $q(1/10(1+i)) =0.999600 + 0.02i$. $|q(u)| = 0.9998$. 
    

---

### Problem 1.8.7

![Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%201.png](Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%201.png)

- My Solution
    
    Let $L(t) = f(\gamma (t))$.
    
    $$
    [DL(t)] = [Df\begin{pmatrix}t\\t^2\\\vdots\\t^n\end{pmatrix}][D\gamma (t)]
    $$
    
    By Theorem 1.8.1, the derivative of a vector-valued function is the derivative of each entry in the matrix is given by its single variable derivative
    
    $$
    [D\gamma (t)] = \begin{bmatrix}1\\2t\\3t^2\\ \vdots\\nt^{n-1}\end{bmatrix}.
    $$
    
    The derivative of $f$ is the $1\times n$ matrix of partial derivatives $D_i$. 
    
    $$
    [Df(x, \dots, x_n)] = \begin{bmatrix}
    D_1\sum_{i=1}^{n-1}x_ix_{i+1}&\dots&
    D_n\sum_{i=1}^{n-1}x_ix_{i+1}
    \end{bmatrix}
    $$
    
    $$
    =\begin{bmatrix}
    x_2&
    x_1+x_3&
    x_2+x_4&\dots&
    x_{n-2} + x_n&
    x_{n-1}
    \end{bmatrix}
    $$
    
    Taking the dot product of these two matrices, we find that $[DL(t)]$ is
    
    $$
    t^2 + 2t(t+t^3)+3t^2(t^2+t^4)+\dots+(n-1)t^{n-2}(t^{n-2}+t^n)+nt^{n-1}(t^{n-1})
    $$
    
    $$
    =t^2 + \sum_{i=2}^{n-1}it^{i-1} (t^{i-1}+t^{i+1})+nt^{n-1}(t^{n-1})
    $$
    

---

### Problem 1.24

![Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%202.png](Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%202.png)

- My Solution
    
    The set $A_m$ converges only when $\theta = 2\pi$. If so, then $A_m$ would be equal to the identity matrix $I_2$. It has a convergent subsequence for all $\theta$ because it is a sequence on the compact set of the unit cube in $\R^3$. 
    

---

### Problem 1.29

![Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%203.png](Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%203.png)

---

### Problem 1.35

![Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%204.png](Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%204.png)

---

# 2015 Fall Final

![Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%205.png](Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%205.png)

# 2018 Fall Take Home Prelim

![Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%206.png](Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%206.png)

- My Solution
    
    Let $f$ be the reflection in the $xw$-plane and $g$ be the reflection in the $yw$-plane. 
    
    $$
    f=\begin{bmatrix}1&0&0&0&0\\
    0&-1&0&0&0\\
    0&0&-1&0&0\\
    0&0&0&1&0\\
    0&0&0&0&1
    \end{bmatrix}
    g=\begin{bmatrix}
    -1&0&0&0&0\\
    0&1&0&0&0\\
    0&0&-1&0&0\\
    0&0&0&1&0\\
    0&0&0&0&1
    \end{bmatrix}
    $$
    
    $$
    g\circ f = \begin{bmatrix}
    -1&0&0&0&0\\
    0&-1&0&0&0\\
    0&0&1&0&0\\
    0&0&0&1&0\\
    0&0&0&0&1
    \end{bmatrix}
    $$
    
    To project this into $\R^3$, we multiply it by a $4\times 5$ identity matrix. 
    
    $$
    \begin{bmatrix}
    1&0&0&0&0\\
    0&1&0&0&0\\
    0&0&1&0&0\\
    0&0&0&1&0
    \end{bmatrix}
    \begin{bmatrix}
    -1&0&0&0&0\\
    0&-1&0&0&0\\
    0&0&1&0&0\\
    0&0&0&1&0\\
    0&0&0&0&1
    \end{bmatrix}
    = \begin{bmatrix}
    -1&0&0&0&0\\
    0&-1&0&0&0\\
    0&0&1&0&0\\
    0&0&0&1&0\\
    \end{bmatrix}
    $$
    
    By multiplying it by a $5\times 4$ identity matrix, we get the final projection. 
    
    $$
    [T]=\begin{bmatrix}
    -1&0&0&0&0\\
    0&-1&0&0&0\\
    0&0&1&0&0\\
    0&0&0&1&0\\
    \end{bmatrix}
    \begin{bmatrix}
    1&0&0&0\\
    0&1&0&0\\
    0&0&1&0\\
    0&0&0&1\\
    0&0&0&0
    \end{bmatrix}
    $$
    
    $$
    [T] = \begin{bmatrix}
    -1&0&0&0\\
    0&-1&0&0\\
    0&0&1&0\\
    0&0&0&1\\
    \end{bmatrix}
    $$
    

![Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%207.png](Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%207.png)

- My Solution
    
    $$
    [Df(t)] = \begin{bmatrix}-\sin t\\\cos t\\1\end{bmatrix}
    $$
    
    $$
    L(t) =[Df\begin{pmatrix}1/\sqrt{2}\\
    1/\sqrt{2}\\
    \pi / 4\end{pmatrix}] = \begin{bmatrix}
    -\sin (\pi/4)t\\
    \cos (\pi/4)t\\
    t
    \end{bmatrix}+
    \begin{pmatrix}1/\sqrt{2}\\
    1/\sqrt{2}\\
    \pi / 4\end{pmatrix}
    $$
    
    $$
    L(t)=
     \begin{bmatrix}
    -(1/\sqrt{2})t\\
    (1/\sqrt{2})t\\
    t
    \end{bmatrix}+
    \begin{pmatrix}1/\sqrt{2}\\
    1/\sqrt{2}\\
    \pi / 4\end{pmatrix}
    $$
    
    $$
    L(t) = \begin{pmatrix}
    -\frac{1}{\sqrt{2}}t+ 
    \frac{1}{\sqrt{2}}\\
    \frac{1}{\sqrt{2}}t+ 
    \frac{1}{\sqrt{2}}\\
    t+\frac{\pi}{4}
    \end{pmatrix}
    $$
    

![Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%208.png](Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%208.png)

- My Solution
    
    The directional derivative is given by the equation
    
    $$
    \lim_{t\to 0}\frac{1}{t}f\begin{pmatrix}\begin{bmatrix}2&-1\\3&5\end{bmatrix}
    +t\begin{bmatrix}a&b\\c&d\end{bmatrix}\end{pmatrix} - f\begin{pmatrix}\begin{bmatrix}2&-1\\3&5\end{bmatrix}\end{pmatrix}
    $$
    
    $$
    =\lim_{t\to0 }\frac{1}{t}\begin{pmatrix}
    \begin{bmatrix}
    2+at&-1+bt\\
    3+ct&5+dt
    \end{bmatrix}
    \begin{bmatrix}
    2+at&3+ct\\
    -1+bt&5+dt
    \end{bmatrix}-
    
    \begin{bmatrix}5&1\\1&34\end{bmatrix}
    \end{pmatrix}
    $$
    
    $$
    = \lim_{t\to 0}\frac{1}{t}
    \begin{bmatrix}
    4at-2bt+a^2t^2+b^2t^2&
    ct+3at+5bt-dt+act^2+dbt^2\\
    ct+3at+5bt-dt+act^2+dbt^2& 10dt+6ct+d^2t^2+c^2t^2\end{bmatrix}
    $$
    
    $$
    =\begin{bmatrix}
    4a-2b&3a+5b+c-d\\
    3a+5b+c-d&6c+10d
    \end{bmatrix}
    $$
    
    The directional derivative is linear because each variable of $B$ only appears once for each entry in the matrix. 
    

# Worked Solutions

[http://pi.math.cornell.edu/~sjamaar/classes/2230/problems/2015-final-solution.pdf](http://pi.math.cornell.edu/~sjamaar/classes/2230/problems/2015-final-solution.pdf)

For 1.6.10, from Romin:

![Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%209.png](Prelim%201%20Practice%20Problems%204c0274c3b81b461b92570d41f178d8e4/Untitled%209.png)