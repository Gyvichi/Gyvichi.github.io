---
title: 'Attention is not all you need: pure attention losses rank doubly exponentially with depth'
date: 2024-08-15
permalink: /posts/2024/Attentionisnotallyouneed/
tags:
  - Deep Learning
---

**Pure attention kills rank.**

>MHA doesn't work without skip connection and FFN : Rank converges to 1 as depth increases, regardless of queries.

MHA concatenates the outputs from each heads, followed by linear transformation and bias. This allows us to reformulate the formula in terms of its individual heads.

$$MHA(X)=Concat(head_1, \cdots, head_H)W^O+b^O=\sum_{h=1}^Hhead_hW_h^O+b^O$$

For $H$ heads and $L$ MHA layers, the output $SA(X)$ from input $X$ can be expressed as follows (where $head_h=P_hX$, ignoring bias):

$$SA(X)=\sum_{h \in [H]}P_hXW_h$$

Here $P_h$ represents the attention matrix for each head, which can be interpreted as a row-stochastic matrix (i.e., each row sums to 1). Since the attention matrix is repeatedly multiplied on the left, while a linear transformation is applied on the right, the overall transformation can be rewritten as follows:

$$Y=\sum (P_{h_L}^L \cdots P_{h_1}^1)X(W_{h_1}^1 \cdots W_{h_L}^L)$$

The key insight is that the product of row-stochastic matrices remains row-stochastic (proof provided below). Therefore, we can simplify the expression as:

$$Y=\sum_{path \in [H]^L}P_{path}XW_{path}$$

A _path_ simply records which sequence of heads a given input follows, represented as a tuple.

![](https://i.imgur.com/4ttgvVt.png)


Now, we introduce a metric called _residual_, which quantifies how close a matrix is to being rank 1.

$$res(X)=X-\mathbf{1}x^T,~x = argmin_x ||X-\mathbf{1}x^T||_{\infty}$$

Minimizing $||X - \mathbf{1}x||_{\infty}$ norm corresponds to minimizing the absolute error between $X_{ij}$ and $\mathbf{1}x$. In this sense, $x$ represents the most significant low-rank approximation of $X$, meaning that rank collapse occurs when the residual error $res(X)$ is small.

The key inequality proposed in the paper is as follows, where $\beta$ denotes an upper bound on the weights:  
($||W_{QK}||_1 ||W_V||_{\infty} \leq \beta$)

$$||res(Y(X))||_{1, \infty} \leq (\frac{4 \beta H}{\sqrt{d_{qk}}})^{\frac{3^L-1}{2}}||res(X)||_{1, \infty}^{3^L}$$

It follows that if the term inside the parentheses is less than 1, $||res(Y)||$ decays exponentially as $L$ increases.

**(Proof not yet fully understood—will be posted in the future.)**

Below is an interesting diagram illustrating how expressivity diminishes when using pure attention alone.

![](https://i.imgur.com/7LFYONe.png)

In conclusion, **FFN and skip connections prevent rank collapse. Moreover, it is not the path length but rather the number of paths that influences expressivity.** In my personal understanding, this result is intuitive: attention focuses on smaller regions of the input, inherently reducing rank. We can say this paper suggests that Transformers outperform other models by strategically collapsing and expanding rank, thereby capturing essential features while maintaining high-dimensional feature subspaces.

---

#### Proof

>Multiplication of row-wise stochastic matrices are row-wise stochastic.

Row-wise stochastic means sum of each rows are 1 and not negative. 
For $C=AB,~c_{ik}=\sum_j a_{ij}b_{jk}$, 

1. $c_{ik} \geq 0$ since both are not negative.
2. $\sum_k c_{ik} = 1$

$$\sum_k{c_{ik}}=\sum_k \sum_j a_{ij}b_{jk}= \sum_j a_{ij}\sum_k b_{jk}=1$$
