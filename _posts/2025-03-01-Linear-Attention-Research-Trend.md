---
title: 'Linear Attention Research Trend'
date: 2025-03-01
permalink: /posts/2025/linearattn/
tags:
  - Deep Learning
---

##### 0. Single Head Standard Softmax Attention

Let's review softmax attention. 

1. Parallel Training

$$
\mathbf{O} = \text{softmax}(\mathbf{QK}^\top \odot \mathbf{M})\mathbf{V} \quad \in \mathbb{R}^{L \times d}
$$
![](https://i.imgur.com/Cu5Svut.png)

Attention requires $\mathcal{O}(L^2d+Ld^2)$ but can be done in $\mathcal{O}(1)$ steps by parallelization. Therefore space complexity will be $\mathcal{O}(L)$, performing $\mathcal{O}(L^2)$ at FLOPs.

cf. Causal mask $\mathbf{M} \in \mathbb{R}^{L \times L}$ is :

$$M_{i,j} =
\begin{cases} 
-\infty & \text{if } j > i \\
1 & \text{if } j \leq i
\end{cases}
$$

2. Iterative Inference $$
\mathbf{o}_t = \sum_{j=1}^{t} \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j)}{\sum_{l=1}^{t} \exp(\mathbf{q}_t^\top \mathbf{k}_l)} \mathbf{v}_j \quad \in \mathbb{R}^{d}
$$
![](https://i.imgur.com/DOt5a9c.png)

While inference, you keep around KV-cache that takes $\mathcal{O}(L)$ memory since they are vectors. Now here's the point : 

![](https://i.imgur.com/cQIa1Bf.png)

**Quadratic computational load** is so bad!
# 1. Kickoff : Linear Attention (Katharopoulos et al. 2020)

Paper : **Transformers are RNNs : Fast Autoregressive Transformers with Linear Attention**.

This paper first proposes the method to make transformers into recurrent form by removing softmax. We can say that kernel function is changed for some new approaches : **associativity**.

1. Parallel training

$$\mathbf{O} = (\mathbf{QK}^\top \circ \mathbf{M})\mathbf{V} \quad \in \mathbb{R}^{L \times d}$$
2.  Iterative Inference
$$\mathbf{o}_t = \sum_{j=1}^{t} \mathbf{q}_t^\top \mathbf{k}_j \mathbf{v}_j \quad \in \mathbb{R}^{d}$$
where $\mathbf{M}$ is the causal mask for linear attention:

$$M_{i,j} =
\begin{cases} 
0 & \text{if } j > i \\
1 & \text{if } j \leq i
\end{cases}$$

Unlike in training mode, we don't need masking in inference mode which allows making KV pairs (pre-computable!) : 

$$
\mathbf{o}_t = \sum_{j=1}^{t} (\mathbf{q}_t^\top \mathbf{k}_j) \mathbf{v}_j= \sum_{j=1}^{t} \mathbf{v}_j (\mathbf{k}_j^\top \mathbf{q}_t), \quad \mathbf{k}_j^\top \mathbf{q}_t = \mathbf{q}_t^\top \mathbf{k}_j \in \mathbb{R}
$$

$$
= \left( \sum_{j=1}^{t} \mathbf{v}_j \mathbf{k}_j^\top \right) \mathbf{q}_t
$$

We define state matrix $S_t$, which brings us both **recurrent formula** and **matrix-valued hidden state**.
$$
\mathbf{S}_t = \sum_{j=1}^{t} \mathbf{v}_j \mathbf{k}_j^\top, \quad \mathbf{S}_t \in \mathbb{R}^{d \times d}, \quad \mathbf{S}_t=\mathbf{S}_{t-1}+\mathbf{v}_t\mathbf{k}_t^\top \in \mathbb{R}^{d \times d}
$$

Therefore, linear attention is **a linear RNN with a matrix-valued hidden state**. You just accumulate the outer product of key and value, and expanding the dimension of state will enable higher expressivity intuitively since the state works as the memory that compresses previous inputs. (Associative memory) This can avoid the bottleneck caused by state size.

+) removing the KV cache can enhance inference latency.

- Autoregressive inference : $\mathcal{O}(L^2d) \rightarrow \mathcal{O}(Ld^2)\rightarrow \mathcal{O}(L)$ (parallelization)
- Space complexity : $\mathcal{O}(Ld) \rightarrow \mathcal{O}(d^2)\rightarrow \mathcal{O}(1)$ (parallelization)

![](https://i.imgur.com/H7zgFeF.png)

This branch of approach is shared with RetNet, DeltaNet, etc. These papers usually perform sophisticated mathematical operations to implement hardware-friendly parallelization : leveraging tensor cores or idea of flash attention. (You can use flash linear attention to use these kinds of models.)

# 2. Limitations of Katharopoulos et al.

#### 2.1. Error on Value Retrieval
Problem emerges when you try to retrieve $\mathbf{v}_t$ from $\mathbf{k}_t$. 

$$\mathbf{Sk}_j = \sum \mathbf{v}_i(\mathbf{k}_i^\top \mathbf{k}_j) = \mathbf{v}_j+\sum_{i \neq j}(\mathbf{k}_i^\top \mathbf{k}_j)\mathbf{v}_i=\mathbf{v}_j
$$

**We need $\mathbf{k}_i^\top \mathbf{k}_j$, i.e. all keys should be orthogonal**. This explains why increasing head dimension helps : we should provide higher dimension in the vector space to store distinct KV pairs. (RetNet paper demonstrated.) Since we **cannot remove KVs but add them**, longer sequence leads to retrieval errors. Researcher Songlin Yang says this is the main reason that vanilla linear attention underperforms compared to softmax attention. She says : 

>The enemy of memory is not time; it's other memory.

This is the main motivation of gating mechanism in linear attention architectures : GLA (Gated Linear Attention), Mamba, etc. However, these models still doesn't show satisfying performance. (Overally DeltaNet 2 seems as the best choice now.)

#### 2.2. Training Mode Left & Materialization of each time step's matrix-valued hidden states is  expensive

One of the solution is **Chunkwise parallel form** (Hua et al, '22, Sun et al, '23) which makes a *chunk*- splits a sequence length $L$ into $L/C$ chunks of size $C$. if $C=1$ reduces to recurrent, if $C=L$ reduces to the parallel form. We can say its an interpolation between recurrent and parallel one, and it is NOT AN APPROXIMATION. You get this satisfactory complexity eventually : 

![](https://i.imgur.com/5emX7JI.png)

Since chunkwise form is just a computation method, it is a de facto standard for training modern linear attention models. **Sequential Chunk-Level State Passing** can be implemented by culmulative sum of chunkwise outer product of key and value.

Side note : $C$ dimension should be multiplier of 16 to utilize tensor cores. You should set appropriate size with your GPU. (M32, M8, K16, etc. e.g. Mamba2 uses 256 for chunk size.)

$$S[t+1] = S[t] + V[t]^T K[t] ~~\in~~ ℝ^{(d×d)},~~ S[t] \in 
ℝ^{(d×d)},~    V[t],~K[t] \in ℝ^{(C×d)}
$$

# 3. Follow-Up Models and Ideas

To summarize, maintaining hardware-friendly implementation and reducing error on value retrieval will be the main goal. Below is the main formula of each linear models that mitigates

| Model            | Recurrence                                                                                                                                                 | Memory Read-Out                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Linear Attention | $\mathbf{S}_t=\mathbf{S}_{t-1}+\mathbf{v}_t\mathbf{k}_t^\top$                                                                                              | $\mathbf{o}_t=\mathbf{S}_{t}\mathbf{Q}_t$                                                      |
| RetNet           | $\mathbf{S}_t=\gamma\mathbf{S}_{t-1}+\mathbf{v}_t\mathbf{k}_t^\top$ (decayed)                                                                              | $\mathbf{o}_t=\mathbf{S}_{t}\mathbf{Q}_t$                                                      |
| DeltaNet         | $\mathbf{S}_t=\mathbf{S}_{t-1}(\mathbf{I}-\beta_t\mathbf{k}_t\mathbf{k}_t^\top))+\beta_t\mathbf{v}_t\mathbf{k}_t^\top$                                     | $\mathbf{o}_t=\mathbf{S}_{t}\mathbf{Q}_t$                                                      |
| Mamba            | $\mathbf{S}_t=\mathbf{S}_{t-1}\odot exp(-(\mathbf{\alpha}_t\mathbf{1}^\top)\odot exp(\mathbf{A}))+(\mathbf{\alpha}_t \odot \mathbf{v}_t)\mathbf{k}_t^\top$ | $\mathbf{o}_t=\mathbf{S}_{t}\mathbf{Q}_t+\mathbf{d} \odot \mathbf{v}_t$                        |
| RWKV-6           | $\mathbf{S}_t=\mathbf{S}_{t-1}Diag(\mathbf{\alpha}_t)+\mathbf{v}_t\mathbf{k}_t^\top$                                                                       | $\mathbf{o}_t=(\mathbf{S}_{t-1}+(\mathbf{d} \odot \mathbf{v}_t)\mathbf{k}_t^\top)\mathbf{q}_t$ |

You can find some less-structurized but experimental models which searches for good token-mixers : QRNN (Quasi-RNN), PoolFormer, FNet, MLP-Mixer, etc. For example you can see short convolution operations added after the affine projections for $\mathbf{QKV}$ in DeltaNet. Since short convolution (or depthwise separable Conv1D) with small window size (4?) has emerged as a crucial component in recent linear attention models, they do be important. Songlin Yang says it seems to provide a shortcut to form induction heads within a single layer which is beneficial for ICL.

There were some mathematical approximation to attention itself, but they showed poor performance against vanilla transformers. I think they are just an "idea" popped out, but here are lists with brief explanation.

1. Sparse Transformer & Strided Attention, Generating Long Sequence with Sparse Transformers (2019) : You define sparse pattern for interaction between query and key and calculate the part of it.
2. Linformer (2020) : Dimensionality reduction of attention matrix via SVD
3. Performer (FAVOR+, 2020) : Approximation of self-attention with random Fourier feature.

From section `4` I will introduce some recent linear models.

# 4. RetNet(2023)

RetNet is consisted of MSR(Gated Multi-Scale Retention) and FFN module.
![](https://i.imgur.com/ldF7gRc.png)


1. Tokenize input by linear projection : $X_n \cdot \omega_V = v(n)$
2. Autoregressive: $s_n=\mathbf{A}s_{n-1}+\mathbf{K}_n^Tv_n$
$$Retention(X)=(QK^T \odot D)V$$
$D$ is **time decaying** causal mask which can be viewed as a gate or modifier too.

$$D_{nm} = \begin{cases} \gamma^{n-m}, & \text{if } n \geq m \\ 0, & \text{if } n < m \end{cases}$$

This operation allows both **recurrent** and **parallel**, breaking away softmax. Reason why **decaying** is in the limelight is that recent tokens are more important than distant tokens in language modelling usually and linear attention doesn't have mechanism to weigh more on recent tokens.  RetNet worked well in practice, proposing decaying factor to linear attention models.

# 5. DeltaNet

#### 5.1. Data-dependent decaying

While RetNet utilized data-independent decaying, DeltaNet applies data-dependent decaying to control memory retention/forgetting. 

$$S_t = \gamma t S_{t-1} + v_t k_t^T \in R^{d \times d}$$
$$o_t = S_t q_t \in R^d$$
$$O = (QK^T \odot D)V \in R^{L \times d}$$
$$D_{i,j} =
\begin{cases}
\prod_{m=j+1}^{i} \gamma_m & \text{if } i > j \\
1 & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}$$

This allows to selectively reflect long-term dependency or recent tokens, or special locations. 

In case of language modeling, we have a strong recency bias. While linear attention does not have such recency bias, DeltaNet shows dynamic decaying helps briding the perplexity gap between linear attention and softmax attention.
#### 5.2. Delta Rule

We have two commonly used update rule for neural net.

- Hebbian : $S_t = S_{t-1} + v_t k_t^\top$
- Delta : $S_t = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top$
 
Especially, Delta rule can be regarded as SGD toward online loss function since it updates its state based prediction errors instead of simply accumulating key-value outer product.

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t (\mathbf{S}_{t-1}\mathbf{k}_t - \mathbf{v}_t)\mathbf{k}_t^T \\
= \mathbf{S}_{t-1} - \beta_t \mathbf{S}_{t-1}\mathbf{k}_t\mathbf{k}_t^T + \beta_t \mathbf{v}_t\mathbf{k}_t^T
$$
* $\beta_t \in \mathbb{R}$ acts as the learning rate
* $\mathbf{k}_t \in \mathbb{R}^d$ is the input data
* $\mathbf{v}_t \in \mathbb{R}^d$ is the target
- $\mathbf{S}_{t-1}\mathbf{k}_t \in \mathbb{R}^d$ is our current prediction

There's another intuitive way to understand this update rule. Think of $\mathbf{S}_{t-1}\mathbf{k}_t$ as retrieving the "old value" associated with the current key $\mathbf{k}_t$ from memory. When we encounter a newly associated value $\mathbf{v}_t$ for the same key, rather than blindly overwriting, we make a careful update:

$$
\mathbf{v}_t^{\text{new}} = (1 - \beta_t) \mathbf{v}_t^{\text{old}} + \beta_t \mathbf{v}_t,
$$
$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \underbrace{\mathbf{v}_t^{\text{old}}\mathbf{k}_t^T}_{\text{erase}} + \underbrace{\mathbf{v}_t^{\text{new}}\mathbf{k}_t^T}_{\text{write}}
$$

where \(\mathbf{v}_t^{\text{new}}\) is a learned combination of the old and current values, controlled by a dynamic \(\beta_t \in (0, 1)\): when \(\beta_t = 0\), the memory content remains intact, and when \(\beta_t = 1\), we completely replace the old associated value with the new one.

#### 5.3. Beyond $TC^0$

![](https://i.imgur.com/GBcTwVy.png)

Since Transformers are SSMs in $TC^0$, we need some techniques like CoT to solve problems on $NC^1$ class. Interestingly, **we can actually achieve expressiveness beond $TC^0$ with non-linear RNN or linear RNN with data-dependent nondiagonal transition matrices.** (Merrill, Petty, and Sabharwal 2024)

DeltaNet utilizes this strategy starting from changing delta rule into form of accumulating key-value outer product. We call the blue term **Generalized Householder (GH) transition matrix**, and $\beta$ relaxes the range.
$$ S_t = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^T $$
$$ = S_{t-1} \left( 1 - \beta_t k_t k_t^T \right) + \beta_t v_t k_t^T$$

If you unroll the recurrence :

$$S_t = S_{t-1} \left( 1 - \beta_t k_t k_t^T \right) + \beta_t v_t k_t^T = \sum_{i=1}^t \beta_i v_i k_i^t \prod_{j=i+1}^t (1 - \beta_j k_j k_j^T) $$

Key is that when allowing negative eigenvalues in GH matrices, the cumulative products of GH matrices can represent any matrix with Euclidean norm < 1. (Grazzi et al. 2024) Also cumulative products of general matrices cannot be computed in $TC^0$. (Mereghetti and Palano 2000)

By experiment in state tracking performance, relaxing the range of the eigenvalue to negative boosted the performance for both Mamba and DeltaNet. You can see Transformer's performance is nearly 0 due to its $TC^0$ complexity.
![](https://i.imgur.com/bIss33A.png)

Besides, there is a problem : cumulative product of matrices are so expensive. However since we have structured transition matrix, researchers found an algorithm called **WY representation** (Bischof and Loan 1985) which allows cumulative product term to be cumulative sum.

$$\prod_{j=i+1}^t (1 - \beta_j k_j k_j^T) = 1 - \sum_{i=1}^t w_i k_i^\top$$


# 6. RWKV

RWKVv4 (Dove, 2023) to RWKVv6 (Finch and Eagle). RWKVv7 is being trained now.

##### 6.1. AFT(Attention Free Transformer, 2021)
RWKV's inspiration is clear : AFT. AFT changes QKV attention as below :

$$Attn(Q, K, V)=\sigma\left(\frac{Q_i (K_i)^T}{\sqrt{d_k}}\right) V_i ~ \rightarrow~
\sigma_q(Q_t) \odot 
\frac{
    \sum_{t'=1}^T \exp(K_{t'} + w_{t,t'}) \odot V_{t'}
}{
    \sum_{t'=1}^T \exp(K_{t'} + w_{t,t'})
}
$$

Query passes sigmoid to control, and $softmax(K)$ gets **learned pair-wise positive bias** matrix $w$ for positional encoding and dynamic approximation.

#### 6.2. Idea of RWKV

RWKV is consisted of time-mixing and channel-mixing which corresponds to self-attention and FFN. It works with a **receptence** vector which helps to consider 일부 token.

First, you get $\mathbf{R, K, V}$ to control information flow by token-mixing. First three vectors are vectors for time-mixing, and the latter three ones are for channel-mixing.
$$
r_t = W_r \cdot (\mu_r \odot x_t + (1 - \mu_r) \odot x_{t-1}),
$$

$$
k_t = W_k \cdot (\mu_k \odot x_t + (1 - \mu_k) \odot x_{t-1}),
$$

$$
v_t = W_v \cdot (\mu_v \odot x_t + (1 - \mu_v) \odot x_{t-1}),
$$

$$
r'_t = W'_r \cdot (\mu'_r \odot x_t + (1 - \mu'_r) \odot x_{t-1}),
$$

$$
k'_t = W'_k \cdot (\mu'_k \odot x_t + (1 - \mu'_k) \odot x_{t-1}).
$$

Second, time-mixing is an alternative of QKV attention. 
$$TM:=\sigma(R)\cdot \sum W \cdot softmax(K)V$$


 Third, channel-mixing works as FFN and is inspired from GeGLU. 

$$CM:=\sigma(R)\cdot \sum W \cdot GeLU(K) \cdot V$$

한편 hybrid Transformer-RNN 구조라 할 수 있다. 

![](https://i.imgur.com/amMFBlG.png)

RWKVv7 applies            for beyond TC0 limitation.


# 7. Thoughts

1. Operations for going beyond TC0 limitation is crucial.
2. Recurrent CoT?
3. Need to be trained >30B...
4. Are there mathematical methods to store orthogonal keys in lower dimension?

+) I didn't introduce some main variants such as SSMs. You should surf internet.
