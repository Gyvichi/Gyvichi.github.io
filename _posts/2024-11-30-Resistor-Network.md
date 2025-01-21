---
title: 'Resistor Network'
date: 2024-11-30
permalink: /posts/2024/Resistor-Network/
tags:
  - Phsyics
  - Mathematics
---

1학기 때 남겨둔 Resistance Distance에 대해 이해했다. Resistance Distance는 어떤 그래프의 간선 위에 저항이 있을 때 그 합성 저항을 구하는 일반화된 함수를  말한다. 흥미로운 점은 저항 네트워크을 연구하는 분야가 수학의 그래프 이론과 결합되어 존재한다는 점이었고, 직병렬 회로인 그래프를 series-parallel graph라는 용어가 있다는 것이었다.

Series-parallel graph는 직관적으로 합성저항 식을 사용하면 되므로 실제로 합성저항 구하기가 어렵지 않으며, 그래프 이론에서는 직렬 회로를 추가하는 series composition, 병렬 회로를 추가하는 parallel composition이라는 연산을 통해 연산을 해결한다. 

![](https://i.imgur.com/JVFKMW6.png)
Series-parallel network의 주요 성질은 아래이다.

>그래프 G가 series-parallel network일 필요충분조건은 G가 confluent한 것이다. (Duffin, 1965)

그래프 이론에서 confluent는 그래프 내 임의의 cycle을 잡았을 때 각 cycle 내 간선을 지나는 방향이 동일한 것이다. 즉, 전류 방향이 정해져있다는 것이다. 하지만 문제는 휘트스톤 브릿지 같은 회로이다. 휘트스톤 브릿지의 경우 non-confluent할 수 있고, 그러한 경우 저항값에 따라 전류의 방향이 바뀔 수 있다는 것을 암시한다. (실제로 휘트스톤 브릿지는 $R_1R_4=R_2R_3$ 평형점을 기준으로 가운데 저항인 $R_x$에 전류가 흐르는 방향이 바뀐다.) 

따라서 키르히호프 법칙을 일반적인 그래프에 대해 적용할 수 있어야 한다. 회로 $G$에 $n$개의 점이 있고, $i$번 점과 $j$번 점을 연결하는 저항 $R_{ij}$의 도선이 있다고 하자. (도선이 없는 간선은 $R_{ij} = \infty$로 둘 수 있다. 이때 $a$번 점과 $b$번 점 사이에 전류 $I$가 흐른다 하자.

키르히호프 제1법칙과 옴의 법칙으로부터, $n$개의 정점의 전압을 각각 $V_1, \cdots, V_n$이라고 두면 $i$번 정점에 대해 다음 식이 성립하게 된다.

1. $i \neq a, b$일 경우:
   $$
   \sum_{j \neq i} \frac{1}{R_{ij}} (V_i - V_j) = 0
   $$

2. $i = a$일 경우:
   $$
   \sum_{j \neq a} \frac{1}{R_{aj}} (V_a - V_j) = I
   $$

3. $i = b$일 경우:
   $$
   \sum_{j \neq b} \frac{1}{R_{bj}} (V_b - V_j) = -I
   $$

- $V_i$의 계수는 $\sum_{j \neq i} \frac{1}{R_{ij}}$, $V_j$에 대해 $-\frac{1}{R_{ij}}$이다.
- 행렬 $L$의 **원소 $L_{ij}$를 $i$번 정점에 대한 $V_j$의 계수로 두면, 그래프 $G$의 라플라시안과 같다.**
- 벡터 $F$의 원소 $F_i$를 $i$번째 식의 우변 (i.e. $i$번 점에서 빠져나가는 전류의 총량)으로 둔다.

$V = (V_1, \cdots, V_n)^t$로 두면, 아래와 같은 라플라시안을 푸는 방정식이 곧 일반적인 구조의 회로의 합성저항을 푸는 것과 동치임을 알 수 있다.
$$
LV = F
$$


키르히호프 정리 : https://infossm.github.io/blog/2022/09/18/kirchhoffs-theorem/

https://blog.queuedlab.com/posts/kirchhoffs-theorem

