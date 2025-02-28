---
title: "An Efficient and Generalizable Symbolic Regression Method for Time Series Analysis"
excerpt: "NEMoTS does symbolic regression by MCTS with PUCT"
collection: portfolio
---

>NEMoTS can perform symbolic regression on given time series by structuring formula backbone with MCTS, and optimizing coefficients with NNs.

Abstract 내용이 너무 SF 같아서 읽게 되었다. 논문에서 제안하는 Neural-Enhanced MCTS(NEMoTS)는 MCTS로 주어진 시계열에 대한 symbolic regression을 수행한다. KAN과 같이 트리 구조로 수식을 나타내며, MCTS로 search space를 줄여 비선형 동역학계의 symbolic regression도 가능하다고 한다.  

### 1. Pipeline
다음 4가지로 구성된다:

- Pre-defined basic function library.
- MCTS
- Policy-value network
- Coefficient optimizer

![](https://i.imgur.com/7Y8dJrZ.png)


KAN과 같이 트리 구조로 수식을 표현하며, policy-value network가 NEMoTS의 NN에 해당하는 부분으로 MCTS의 simulation phase에서 노드를 평가하는데 사용된다.

### 2. Process
먼저 MCTS가 Pre-defined basic function library를 참고하여 주어진 시계열에 맞는 공식 형태를 추정한다. i.e. 주어진 기저 함수와 기저 연산을 바탕으로 트리 구조를 만들어서 시계열에 맞는 수식 형태를 만든다. 이 과정에 Policy-value network가 개입하여 MCTS의 성능을 높인다. 이후 Coefficient optimizer가 수식 내부 계수들을 최적화하는 형태이다. 자세히 뜯어보자.

#### 2.1. Objective
입력은 $N$개 점으로 이뤄진 시계열 $\mathcal{D} = (t_i, v_i)_{i=0}^{N-1}$ 로 timestamp $t_i \in \mathbb{R}$와 그에 해당하는 값 $v_i \in \mathbb{R}$ 로 구성된다. 우리의 목표는 analytical expression $f(\cdot)$를 찾고 아래 보상 함수로 평가하는 것이다.

$$
0 \leq \mathcal{R} = \frac{\eta^s}{1 + \sum_{i=0}^{N-1} \sqrt{(v_i - f(t_i))^2}} \leq 1
$$

$\eta$는 1보다 약간 작은 상수, $s$는 $f$의 크기를 말한다. 즉 위 보상 함수 값이 1에 가까울수록 $\sqrt{(v_i - f(t_i))^2}$가 0에 가까우므로 정확도가 높다고 이해할 수 있다. 또 $s$가 커질수록 보상이 작아지므로, 높을수록 수식 형태가 간단함을 의미하기도 한다.
.
#### 2.2. Generating Backbone
MCTS로 수식 backbone을 어떻게 만드는지 알아보자. 

1. Selection
NEMoTS는 PUCT(Polynomial UCT)를 활용하는데, 아래와 같은 PUCT score를 사용한다. 각 트리의 노드는 보상 $Q$와 노드 방문 횟수 $N$을 저장하고 있음을 유의하자.
$$
Score(S, a) = Q(S, a) + c \cdot P(S, a) \cdot \frac{\sqrt{\sum_b N(S, b)}}{1 + N(S, a)}
$$
- $Q(S, a)$: state $S$에서의 action $a$의 예상 reward.
- $P(S, a)$: state $S$에서 action $a$를 취할 확률.
- $N(S, a)$: state $S$에서 action $a$를 취한 상황의 수.
- $c$: exploration과 exploitation을 조절하는 상수.

탐색한 트리 내 탐색은 위 식에 따라 가장 높은 점수를 가진 노드를 따라간다.

2. Expansion
새 노드를 탐색하는 expansion에서는 pre-defined basic function library에서 한 함수가 랜덤으로 선택되어 트리에 새 노드로써 연결된다. ($N$과 $Q$는 0으로 초기화.) 

3. Simulation
Simulation에서 가장 중요한 것은 적은 시간복잡도로 빠르게 평가하는 것일텐데, 기존의 time-intensive한 random simulation에서 벗어나 NASNet 등에서 신경망 구조만 보고 추측하여 평가하듯 Policy-value network를 끝에 달아 보상 추정을 진행한다. 이후 Coefficient optimizer에 의해 coefficient가 결정된다.

4. Backpropagation
CFG에 따라 트리 구조로 형성된 수식을 파싱하고 입력 데이터셋과 비교하여 backpropagation을 진행한다.


논문 내용이 너무 추상적이고 코드 제공도 안되어 있어서 여전히 답답한 구석이 있다. 그래도 Symbolic Regression 모델 중에선 아마 새로운 접근일텐데 신선했다. 언젠가 KAN이나 NEMoTS에서 CFG 파서로 수식을 나타내는 기법에 대해서도 소개하고자 한다. (한 2년 뒤?ㅋㅋ)
