---
title: "Understanding Lambda Optimization with Aliens"
date: 2023-04-19 08:26:28 -0400
categories: jekyll update
---

Lambda Optimization(Alien Trick, WQS Binary Search, Lagrangian Relaxation)은 Penalty Method를 활용하는 컨벡스 최적화와 관계 깊다. 아쉽게도 Lambda Optimization에 대해 정성적인 설명이 많고 Lagrangian Multiplier와의 관계가 덜 서술되어 있어 정리한다.

이 기법은 IOI 16' Aliens에서 먼저 제시되었는데, 당시 만점자가 1명이었다. 그 분은 중국 커뮤니티 CP에서 배웠다고 한다. 하지만 람다 최적화는 그 내용이 꽤 간단해서, 충분한 인터넷 서치와 함께라면 이해할 수 있다.

## 1. 이론적 배경
#### 1.1. 라그랑주 승수법(Lagrange Multiplier)
라그랑주 승수법은 제약이 있는 함수의 최적화 알고리즘이다. 라그랑주 승수법의 메인 아이디어는 **두 함수가 접할 때 접선을 공유하여 gradient가 평행**하다는 것이다.
따라서 최적화하고자 하는 함수 $f$와 제약 $g$에 대해 $\nabla f = \lambda \nabla g$라 표현할 수 있다. (이때 $\lambda$를 Lagrange Multiplier라 부른다.)

예를 들어 $x^2+y^2=1$을 만족하는 $x, y \in \mathbb{R}$에 대해 $x+y$의 최대$\cdot$최소를 구할 때 라그랑주 승수를 적용하면 아래와 같다.
$$(2x, 2y) = \lambda (1, 1)$$
즉 $x=y=\frac{\lambda}{2}$이므로 이를 대입해서 풀어낼 수 있다.

한편 이 과정은 다변수함수 $f=c$에 대해 아래와 동일하다. (마지막 $g=c$가 대입 과정)
$$
\nabla f = \lambda \nabla g \rightarrow \nabla f - \lambda \nabla g = 0 \rightarrow
\begin{cases}
\frac{\partial f}{\partial x_1}(x_1, \ldots, x_n) - \lambda \frac{\partial g}{\partial x_1}(x_1, \ldots, x_n) = 0 \\
\frac{\partial f}{\partial x_2}(x_1, \ldots, x_n) - \lambda \frac{\partial g}{\partial x_2}(x_1, \ldots, x_n) = 0 \\
\vdots \\
\frac{\partial f}{\partial x_n}(x_1, \ldots, x_n) - \lambda \frac{\partial g}{\partial x_n}(x_1, \ldots, x_n) = 0
\end{cases}, \quad g(x_1, \ldots, x_n) = c
$$
마지막 연산이 일반화하기 까다로운데, 이는 **Lagrange function** $\mathcal{L}\equiv f+\lambda(c-g)$을 정의하여 $\nabla \mathcal{L}=0$을 계산함으로써 해결된다. $\mathcal{L}$을 $\lambda$로 편미분할 때 $c=g$가 도출되기 때문이다. 이렇게 함으로써 제약 조건이 사라진 하나의 함수로 만들 수 있다. 그리고 제약이 $g_1 \cdots g_k$일 때 그에 해당하는 $\lambda_i$를 정의함으로써 $\lambda$를 벡터로 일반화할 수 있다.

>$\therefore$ 제약 $g$가 걸린 함수 $f$의 최적화는 아래 수식을 통해 가능하다.
>$\nabla \mathcal{L} = \nabla f - \nabla \lambda (c-g) = 0$

#### 1.2. IOI 16' D2 Q6 Aliens
![](https://i.imgur.com/zAOa4ep.png)
문제는 아래와 같다.
>$m \times m$의 정사각형 행렬 위 $n$개의 중복 가능 점들이 $r[i],~c[i]$로 주어진다. $k$개의 **대각선이 주대각선과 일치하는 정사각형**을 그려 모든 점들을 포함시킬 때, 그린 정사각형들의 합집합이 차지하는 영역을 최소화하고자 한다. 이때 그 영역을 출력하시오. ($1 \times 1$의 넓이를 1로 두자.)
>
>**시간 제한 2초, $1 \leq n \leq 5e5,~1 \leq m \leq 1e6$

이 문제는 본래 점화식을 구성하여 Brute forcing보다 빠르게 푸는 다이나믹 프로그래밍 문제이다. $i$개 점을 포함하는 $k$개 정사각형이 가지는 최소 영역을 $a[i][k]$로 두자. 


이 문제의 중요한 점은 바로 주어지는 입력의 크기가 너무 커 시간 제한 내에 푸는 알고리즘을 작성하려면 많은 최적화가 필요하다는 것이다.

#### 1.3. 용어
- Penalty Method
	- 위 Lagrange Multiplier처럼 제약 문제를 제약되지 않은 문제로 풀어내는 기법을 모아 Penalty method라 한다. 제약되지 않은 함수로 풀어졌을 때 그 함수를 Penalty function이라 하는데, 이를 구성하는 parameter를 Penalty parameter라 한다.
- Epigraph $epi~f$
	- Epi-는 above를 뜻하여, 함수 $f$ 윗부분을 의미한다.

## 3. 탐구 내용 및 결과
람다 최적화의 아이디어는 **다변수 점화식 중 제약에 해당하는 변수를 penaly parameter화하여 차원 하나를 죽인 penalty function을 만들어 제약을 완화시키는 것**이다.

$i$번째 점까지를 포함하는 영역 값을 수열 $a_{i, \cdots}$로 두자. 
여기서 아이디어는 $k$가 커질수록 비용 $\lambda$가 든다고 생각하는 것이다. 만약 $k=1$이면 한 정사각형이 모든 점을 포함해야 하므로 답이 최대일 것인데, $k=2,~3,~\cdots$이 될수록 cost $\lambda$가 들어 점점 영역이 감소하는 것이다. 


$k$를 무시하자. $i$째 점까지 덮는 최소 영역을 탐색하고자 한다. 이때 답에 해당하는 정사각형 개수는 무엇이 될 지 모른다. 그래서 $k$가 변수로 들어가야 하는 것인데, 람다 최적화의 아이디어는 정사각형으로 나눌 때마다 Cost $\lambda$를 부여한다는 것이다. 만약 $\lambda$가 0이라면 정사각형을 계속 사용해도 Cost가 없으므로 답은 모든 점을 위한 최소한의 정사각형을 사용하는 경우일 것이고, $\lambda$로 매우 큰 값을 사용한다면 한 번 새 정사각형을 사용할 때마다 Cost가 커져 큰 정사각형 하나만 사용될 것이다.

따라서, 모든 $\lambda$에 대해 탐색해가며 그 $\lambda$에서 $k$로 덮는 것이 최선인 경우를 찾는 것이다. 

한편 다시 생각해보면 실제 답에 매번 $\lambda$가 더해졌으므로, $a_i-k \lambda$가 답임을 알 수 있다.

필요한 것인데
이를 참고하여 복잡한 귀납적 관계를 $f$로 두고 $\lambda$를 추가하여 점화식을 세워보자.
$$a[i]=min~\{f(a[i-1])\}+\lambda$$

즉 이전 $i-1$개 점을 덮는 영역에서의 답에서 

#### 3.1. 직관적 접근




의 접근 이전에 $k$에 따른 답 함수를 생각해보자. 직관적으로 $k$가 늘어날수록 답 영역은 점점 줄어들며, 볼록할 것이라 추측할 수 있다. 우리는 $f(k)$를 구하면 된다.
![](https://i.imgur.com/iSYzGf2.png)
*$x$축이 $k$, $y$축이 답일 때 위와 같은 그래프 개형이 나타날 것이라 추측할 수 있다.* 

여기에 라그랑주 승수의 아이디어를 적용하여, 

즉 $k$가 커질수록 
이때 어떤 실수 $\lambda$에 대해 각 $k=k_i$마다의 접점을 구하는 건 어떨까?

$f(x)+\lambda x$이 최소화되는 지점 
여기서 Lagrange Multiplier의 아이디어를 따라 입력에 따라 결정되는 접선 $cx$가 있으면 $f(x)+cx$를 최소화하여 답을 구할 수 있다. 접선의 기울기는 $f(x)$의 볼록성에 의해 계속 감소하므로 이분탐색을 통해 $c$ 탐색을 가속할 수 있다. 

그래서 놀랍게도 시간복잡도 항은 $k$가 사라진 $\mathcal{O}(n log m)$이다!


$X$: 가능한 정사각형 $k$개들의 집합
$f(x)$를 $g(x)=y$ 제약에서 최소화해야 한다.

따라서 이에 대한 Lagrange dual function은 아래와 같다.
$$t(\lambda)=inf_{y \in Y}[h(y)-\lambda \cdot y]$$
#### 3.3. $\lambda$의 존재성과 정수가 아닐 가능성
$k$개 정사각형을 사용할 때 optimal $\lambda$값을 $f(c)$라 정의하자. 

$$f(i)=min_{t<i}~\{g(t)\}+(r_i-l_{i+1}+1)^2-max(r_t-l_{t+1}+1, 0)+\lambda$$





결론적으로 위와 같은 유의미한 산출물을 얻었다.
#### 4. 제언 및 소감

#### 5. Ref
- https://www.youtube.com/watch?v=WZKOdorb1Dg
- https://convex-optimization-for-all.github.io/
- https://www.yohandi.me/blog/lagrange-relaxation/
https://mamnoonsiam.github.io/posts/attack-on-aliens.html

### 2.1. Monge Array
$1\leq a < b \leq n,~1 \leq c < d \leq m$인 $a, b, c, d$에 대해 아래를 만족하면 $n \times m$ 행렬 $A$는 monge array이다.
$A[a, c]+A[b, d] \leq A[a, d]+A[b, c]$

