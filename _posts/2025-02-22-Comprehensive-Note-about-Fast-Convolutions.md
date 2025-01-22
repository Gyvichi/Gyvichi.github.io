---
title: 'Comprehensive Note on Fast Convolutions'
date: 2025-02-22
permalink: /posts/2025/Conv/
tags:
  - CS
  - Mathematics
---

Today I will summarize the entire FFT/CZT stuffs for PS and other fields in everything I know, I am sure there may be some interesting codes.




## 10. Hell-Joseon Style NTT
Hell-Joseon style NTT is an algorithm implemented by `DeobureoMinkyuParty` which went to 24th in ICPC WF '19. I may call HJNTT below, HJNTT utilizes AVX for acceleration.

#### 10.1. SIMD
SIMD(Single Instruction Multiple Data)는 특정 명령어 연산을 병렬적으로 수행하는 CPU 명령어 셋이다.

기존 우리 코드는 SISD(Single Instruction Single Data)를 따른다. SISD의 경우 레지스터에 연산할 값들을 각각 저장하고 결과를 저장하는 형태를 따르는데, SIMD는 주어지는 여러 데이터를 모아 한 번에 instruction을 수행한다. i.e. 256 byte에 그보다 크기가 작은 자료형 여러 개를 쌓아 한 번에 연산하는 방식이다.
![](https://i.imgur.com/3L5Ad2t.png)

그래서 벡터/행렬 연산이 잦을 때 SIMD의 효력이 잘 나타난다.

Intel에서 지원하는 SIMD 중 AVX/AVX2는 백준에서도 사용할 수 있다! AVX(Advanced Vector Extension, Intel, 2008)는 고성능 ISA(Instruction Set Architecture)로 C의 instruction function 연산을 가속할 수 있다.

여러 최적화와 AVX를 위해 앞으로 cpp 함수 위 아래 스니펫을 복붙하자.

```cpp
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define fastio ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
#pragma GCC optimize("O3")
#pragma GCC optimize("Ofast")
#pragma GCC optimization("unroll-loops")
#pragma GCC target("avx,avx2,fma")
#pragma pack(1)

#include <bits/stdc++.h>
#include <immintrin.h>
using namespace std;
```

자료형을 SIMD를 위해 쌓는 과정에서 load 함수를 사용하고, 연산 이후 다시 기존 자료형으로 옮길 때 store 함수를 사용한다.
**메모리 정렬을 위해 시작 주소를 해당 자료형 크기의 배수로 맞춰야 한다.** `alignas` 혹은 int는 `_m256i`, double은 `_m256d`, float은 `_m256`로 선언하자.


### 3. Hell-Joseon Style NTT
```cpp
__m256d mult(__m256d a, __m256d b){
    __m256d c = _mm256_movedup_pd(a);
    __m256d d = _mm256_shuffle_pd(a, a, 15);
    __m256d cb = _mm256_mul_pd(c, b);
    __m256d db = _mm256_mul_pd(d, b);
    __m256d e = _mm256_shuffle_pd(db, db, 5);
    __m256d r = _mm256_addsub_pd(cb, e);
    return r;
}

void fft(int n, __m128d a[], bool invert){
    for (int i=1, j=0; i<n; ++i){
        int bit = n>>1;
        for (; j>=bit; bit>>=1) j -= bit;
        j += bit;
        if (i < j) swap(a[i], a[j]);
    }
    
    for (int len=2; len<=n; len<<=1){
        double ang = 2*3.14159265358979/len*(invert?-1:1);
        __m256d wlen; wlen[0] = cos(ang); wlen[1] = sin(ang);
        for (int i=0; i<n; i+=len){
            __m256d w; w[0] = 1; w[1] = 0;
            for (int j=0; j<len/2; ++j){
                w = _mm256_permute2f128_pd(w, w, 0);
                wlen = _mm256_insertf128_pd(wlen, a[i+j+len/2], 1);
                w = mult(w, wlen);
                w = _mm256_extractf128_pd(w, 1);
                __m128d u = a[i+j];
                a[i+j] = _mm_add_pd(u, vw);
                a[i+j+len/2] = _mm_sub_pd(u, vw);
            }
        }
    }
    
    if (invert){
        __m128d inv; inv[0] = inv[1] = 1.0/n;
        for (int i=0; i<n; ++i) a[i] = _mm_mul_pd(a[i], inv);
    }
}

vector<int64_t> multiply(vector<int64_t>& v, vector<int64_t>& w){
    int n = 2; while (n < v.size() + w.size()) n <<= 1;
    __m128d* fv = new __m128d[n];
    for (int i=0; i<n; ++i) fv[i][0] = fv[i][1] = 0;
    for (int i=0; i<v.size(); ++i) fv[i][0] = v[i];
    for (int i=0; i<w.size(); ++i) fv[i][1] = w[i];
    fft(n, fv, 0);  // (a+bi) is stored in FFT
    for (int i=0; i<n; i += 2){
        __m256d a;
        a = _mm256_insertf128_pd(a, fv[i], 0);
        a = _mm256_insertf128_pd(a, fv[i+1], 1);
        a = mult(a, a);
        fv[i]=_mm256_extractf128_pd(a, 0);
        fv[i+1]=_mm256_extractf128_pd(a, 1);
    }
    fft(n, fv, i);
    vector<int64_t> ret(n);
    for(int i = 0; i <n; i++) ret[i] = (int64_t)round(fv[i][1]/2);
    delete[] fv;
    return ret;
}

```

https://justicehui.github.io/hard-algorithm/2021/11/15/simd-in-ps/

https://cgiosy.github.io/posts/fast-io
