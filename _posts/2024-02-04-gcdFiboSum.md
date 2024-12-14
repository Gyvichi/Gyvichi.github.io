---
title: "오일러 피  (BOJ 17372)"
date: 2024-02-04 01:25:43 -0400
categories: PS
---

> $$\sum_{i=1}^n \sum_{j=1}^n \gcd(F_i, F_j)$$
> 
> 주어지는 자연수 $1 \leq n \leq 1e9$에 대한 mod 1e9+7 값을 출력하시오.

찾아보니 xudyh's sieve라는 것이 정해인 것 같은데 

#### 접근

아래에서 시작하자.

$$\sum_{i=1}^n \sum_{j=1}^n F_{\gcd(i, j)}=2*\sum_{i=1}^j \sum_{j=1}^n F_{\gcd(i, j)}-\sum_{i=1}^n F_i$$

이걸 $\mathcal{O}(N)$에 구현해야 하는데... 아래가 실제 계산해야 할 항이다.

$$\sum_{j=1}^n \sum_{i=1}^j F_{\gcd(i, j)} = \sum_{d=1}^n F_d \times \sharp \{ (i, j) \mid 1 \leq i \leq j \leq n, \gcd(i, j) = d \}$$

즉, $gcd(i, j) = d$의 개수만큼 피보나치 값을 더해줘야 하는데, 



https://chatgpt.com/c/675d2fee-0d3c-8011-9903-c3ccd258eaa0?model=gpt-4o
