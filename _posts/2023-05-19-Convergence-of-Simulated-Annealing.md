---
title: 'Convergence of Simulated Annealing'
date: 2023-05-19
permalink: /posts/2023/05/Convergence-of-Simulated-Annealing/
tags:
  - CS
  - Mathematics
---

### 0. SA for Derivative-Free Optimization
Derivative-Free Optimization은 현재 상태의 gradient를 구하는 것이 어렵거나 불가능한 문제의 최적화를 지칭한다. SA가 이 task를 잘 근사해준다. SA의 메커니즘은 아래와 같다.

1. 현재 지점의 value를 계산한다.
2. 이웃 지점의 value를 평가하고 갱신 여부를 결정한다.
   - $value_{neighbor}>value_{now}$: 갱신
   - $value_{neighbor} \leq value_{now}$: 확률 $p$로 갱신
   - 시간에 따라 $p$를 점진적으로 낮춰 global optimal point에 다가간다.

일반적인 SA는 $p$로 Boltzmann Distribution을 활용한다.
$$\Large{\pi(x)=e^{\frac{-E(x)}{kT}}=e^{\frac{E(x)-E(x_{new})}{kT}}}$$
#### 0.1. Code

#### 0.2. Questions
자연스럽게 이런 궁금증이 떠오른다.
1. 수렴하는가?
2. 다른 $p(x)$를 써도 항상 수렴하는가? 어떤 $p(x)$를 사용해야 수렴하는가?

### 1. Proof



e.g 1. BOJ18789 814-2
읽을 수 있는 정도를 점수로 생각하고 최적화를 진행하면 되는데, 문제는 이 점수의 범위가 마치 강화학습에서 reward가 너무 작고 이산적이라 즉각즉각 제대로된 feedback을 주지 못한다는 것이다.





e.g. 2. Risky Lottery

e.g. 3. Ax+Bsinx=C
