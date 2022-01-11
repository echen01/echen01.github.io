# 1.3 Matricies as Linear Transformations

**Def 1.3.2.** A linear transformation $T: \R^n \to \R^m$ is a mapping such that for all scalars $a$ and all $\vec v, \vec w \in \R^n$,

 

$$
T(a\vec v) = aT(\vec v) \text{ and  }T(\vec v + \vec w) = T(\vec v ) + T(\vec w) 
$$

**Theorem 1.3.4.** 

1. Any $m\times n$ matrix $A$ defines a linear transformation $T: \R^n \to \R^n$ by matrix multiplication:

$$
T(\vec v) = A\vec v
$$

2. Every linear transformation $T: \R^n \to \R^m$ is given by multiplication by the $m \times n$ matrix $[T]$:

$$
T(\vec v) = [T]\vec v
$$

where the $i$th column of $[T]$ is $T(\vec e_i)$. 

**Proof that Matrix Multiplication is Linear**

We prove that $A(B+C) =AB+BC$ and $(A+B)C = AC+BC$. 

Suppose $A$ is a $n \times m$ matrix, and $B$ is a $m\times p$ matrix. We define matrix multiplication such that the $i$th $j$th element is given by $\sum_{k=1}^m a_{ik}b_{kj}$. 

$$
(AB)_{ij} = \sum^m_{k=1}a_{ik}b_{kj}
$$

By this definition, 

$$
(A(B+C))_{ij} = \sum_{k=1}^{m}a_{ij}(b_{kj}+c_{kj})
$$

$$
= (AB)_{ij}+(AC)_{ij} = \sum_{k=1}^m a_{ik}b_{kj} + \sum^m_{k=1}a_{ik}c_{kj}
$$

We have shown that this is a linear map.