---
title : "[선형대수] Positive definite, Rank, LS problem에서의 closed-form solution"
excerpt: "Quadratic Programming(QP)를 공부함에 있어서 알아두어야 할 선형대수 개념들을 정리해보자"

category :
    - Fundamental
tag :
    - linear_algebra

toc : true
toc_sticky: true
comments: true

---
Quadratic Programming(QP)를 공부함에 있어서 알아두어야 할 선형대수 개념들을 정리해보자


## Introduction

Mobile Manipulation 관련 논문들을 읽으면서 **Quadratic Programming(QP)**과 관련된 내용들이 자주 언급되는 것을 확인할 수 있었다. **Quadratic Programming(QP)**이란 비선형 계획법의 한 유형으로 특정 수학적 **최적화 문제를 해결**하는 프로세스이다. 이번 포스팅에서는 [QP에 대해 잘 설명한 블로그](https://velog.io/@wjleekr927/Quadratic-program#subsumes-lp)를 통해 QP를 공부함에 있어서 사전에 알아두어야 할 선형대수 개념들을 정리해보도록 하겠다.


## Positive definite와 Positive-semi definite의 차이 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/8bfd6307-8ab8-4598-aebd-320ebd22ddb0" ></p>  

대칭의 구조를 가지는 (n x n) 실수 행렬 M이 있다고 하자. 여기서 n개의 실수 벡터 x에 대해서 위와 같은 관계를 가질 때 행렬 M을 positive definite 이라고 부른다. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/b9d55070-9e2e-4941-952d-3234467e0e30" ></p>

이와 달리 positive semi-definite의 경우 위의 그림에서도 확인할 수 있듯이 positive definite의 상황에 0이 포함된 상황을 의미한다.

먼저 앞에 $x$를 transpose 해서 곱해지는 부분은 무시하고 우선은 그 뒤에 $M$과 $x$가 곱해진 상황만 집중해보자. 이것은 만약 $M$이라는 operator가 있다고 생각했을 때, 이 operator를 $x$에 적용했을 때의 출력물이라고 볼 수도 있다. 즉, 입력 $x$에 뭔가 함수를 씌워서 나온 출력 같은 느낌이다. 이 때 출력된 벡터는 차원이 같게 변화가 일어난다. 다시 말해서 입력이 $x$라고 했다면 본래 차원은 (n x 1)이고, $Mx$를 하게 되면 (n x n) x (n x 1) = (n x 1)으로 같은 차원에 머문다. 이렇듯 차원은 같지만 안에 있는 값들은 변화를 가질 것이다.

이렇게 달라진 벡터 앞에 $x$를 transpose 하여 곱하면, 차원이 (1 x n) x (n x 1) = (1 x 1)로 각각의 원소들을 각각 곱하고, 더하여 하나의 스칼라의 값을 얻는 것이다. 이렇게 입력과의 내적을 구하여 나온 inner product가 양수(positive) 값을 가진다면 positive definite 상황이 되는 것이다. 반대로 음수가 나온다면 negative definite 상황에 해당한다.


## 행렬에서 Rank란 ?

임의의 행렬 A가 있을 때, 이 행렬의 Rank 라는 것은 이 행렬의 열들로 생성될 수 있는 벡터 공간의 차원을 의미한다. 다시 말해서, 행렬 A의 열들 중에서 **선형 독립인 열들의 최대의 개수**를 Rank라고 하고, 이것은 행에 대해서 나타내어 지는 공간의 차원과도 같다.

좀 더 많은 Rank에 관한 정의를 살펴보자.

A라는 행렬의 Column rank는 Column space의 차원을 의미하고 Row rank는 Row space의 차원을 의미한다. 이는 행에서 선형 독립인 벡터의 개수와 열에서 선형 독립인 벡터의 개수를 구분해서 정의하겠다는 것을 의미한다.

여기서 중요한 사실은, 선형 대수에서는 이렇게 따로 계산할 수 있는 **행과 열의 Rank는 항상 동일하다**는 것이다. 따라서, 굳이 따로 계산할 필요 없이 한쪽만 계산해서 A의 Rank라고 일반적으로 말해도 된다.

### Full-Rank

Full Rank는 해당 행렬이 가능한 최대로 가질 수 있는 Rank의 값을 의미한다. 그런데 앞서 말했듯이, 행과 열의 각각의 Rank는 서로 같은 값을 가진다고 하였으므로, 아래와 같이 어느 한 쪽에서 작은 값의 사이즈가 Full Rank의 값이 되겠다. 

아래 수식에서 m은 행의 선형 독립인 벡터의 개수를, n은 열의 선형 독립인 벡터의 개수를 의미한다.

- $Rank(A) = Min(m,n)$

즉, Full Rank는 한 행에서 전부 다 선형 독립이거나, 또는 한 열에서 전부 다 선형 독립인 벡터 기저들을 가진 경우라고 볼 수 있겠다. 

### Example

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/3be90d1f-2a79-498d-a278-b76a37609ddf" ></p>  

예를 들어 위 그림과 같은 정방 행렬이 있다고 생각해보자.

위의 행렬의 경우 Rank는 2이다. 왜냐하면, 첫 번째와 두 번째 열은 서로 선형 독립 관계에 있다고 볼 수 있지만 세 번째 열의 경우, 첫 번째 열에서 두 번째 열을 빼주게 되면 세 번째 열이 되기에 선형 독립 관계에 있다고 볼 수 없다. 따라서, 이 행렬의 Rank는 2가 된다. 


<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/0d47cc1c-d4b5-44a9-8b44-fad2da94f716" ></p>  

다음으로 위 그림과 같이 정방 행렬이 아닌 A 행렬이 있다고 생각해보자.

위의 행렬의 경우에는 세 번째 열을 제외 하고 나머지 열들은 서로 모두 의존적(선형 독립적이지 않다)이다. 첫 번째 열은 정확히 두 번째 열과 같고, 네 번째 열은 첫 번째 열의 두 배와 정확히 같기 때문이다. 그런데 세 번째 열은 0의 값을 가지므로, 랭크에서는 제외된다. 결국 세 번째 열을 제외한 나머지 세 개의 열들은 하나의 기저로 역할을 할 수 있으므로, 하나의 랭크를 가진다고 볼 수 있다. 따라서, 위의 행렬의 Rank는 1이다. 


<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/09881122-38f1-4fa5-a53e-d10912966c59"  ></p>  

다음으로 앞서 말한 A 행렬의 전치 행렬에 대해서도 생각해보자.

A의 전치 행렬의 경우 열만 보게 되면 두 열은 서로 의존적이라는 것을 쉽게 알 수 있다. 왜냐하면 첫 번째 열에 마이너스 1을 곱하게 되면 두 번째 열이 되기때문이다. 그러므로, 결국 이 벡터 공간을 지배하는 기저는 하나이기에 전치 행렬의 경우에도 Rank는 1이 된다. 

우리는 행렬 $A$와 전치행렬 $A^{T}$에 대해서 Rank를 확인해봄으로써 앞서 언급한 행과 열의 랭크는 서로 같다는 법칙이 진실이라는 것을 확인할 수 있다.

### 간단한 요약

행렬의 Rank는 행렬이 나타낼 수 있는 벡터 공간에서 기저의 개수를 의미하고, 이 기저는 서로 독립인 행 또는 열의 벡터의 개수에 의해서 결정된다. 따라서 서로 선형 독립인 벡터가 몇 개가 되는지만 확인하면 된다. 또한 행과 열의 Rank는 서로 같은 값을 가지므로, 행렬의 랭크를 구할 때에는 한쪽의 Rank만 계산하면 된다. 

## Least-Squares(LS) Problems(최소자승법)

현실의 문제에는 여러가지 오차가 포함되어 있기에 $Ax=b$ 문제를 풀 때 해가 없는 경우가 대부분이다. 이런 경우 b와 제일 근접한 $x$를 찾게 되는데 이때 이용하는 방법이 최소자승법이다.

이 때 b와 가장 근접한 $x$를 optimal value 라고 하며 $x^* $로 표현한다. 최적화 문제는 이러한 $A x^* $와 $b$의 차이를 최소로 만들어야 한다. 즉, $\lVert Ax - b \rVert^{2}$ 가 최소가 되어야 한다.

### LS Problem 에서의 Closed-form solution

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/e9328391-ac3c-46fe-81a1-f3589229bc6b" ></p>

- a. $Ax =b$ 에서 b에 대한 최소자승해는 유일하다.
- b. 행렬 $A$ 의 columns는 선형 독립이다.
- c. $A^{T}A$는 역행렬이 존재한다.

일반적인 Least-Squares(LS) Problem 에서의 Closed-form solution은 위의 조건 하에 다음과 같다.

- $x^* = (A^{T}A)^{-1}A^{T}b $

여기서 행렬 $A$가 선형 독립 Column으로 이루어져 있으면 $A = QR$로 분해 가능하다. 이 때 $Q$는 orthonomal column으로 이루어진 행렬이고, $R$은 upper triangular matrix 이다.

따라서 solution을 다음과 같이 표현할 수도 있다.

- $x^* = R^{-1}Q^{T}b$ 

혹은
- $R x^* = Q^{T}b$

R의 역행렬을 계산할 때 많은 연산이 필요하기에 일반적으로 두 번째 식을 사용한다.

### Example

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/8a62afda-fe99-4c59-a7b9-c22a7d444bfd" ></p>
<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/fce1b176-8c7d-4523-a6e2-0617c038f035" ></p>
<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/dd2c6a81-02c9-483b-b88e-ddf3cd919e6e" ></p>  
<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/962ab28d-6d1d-47cf-b925-38496df528fe" ></p>  
<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/82edac01-3ccf-44d8-8005-97bd7c73190c" ></p>
<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/6fdb1855-21fd-4d1d-bb8a-d279f7484e54" ></p>  

위와 같은 방법으로 행렬 $A$와 $b$가 주어져 있을 때 최소자승해를 구할 수 있다. 

## 그 외 간단한 개념들

- Under-determined system : 선형 시스템에서 'm'은 선형 시스템의 방정식 수를, 'd'는 변수의 수를 나타낸다고 할 때 'm'이 'd'보다 작은 상태를 의미한다. 이는 방정식의 수가 변수의 수보다 적은 상태를 말하는데, 일반적으로 이 경우 무한한 해가 존재하기에 목표 함수(objective function)을 0으로 만드는 것이 상대적으로 쉽다.

- Closed-form solution : 주어진 문제가 일반적으로 알려진 함수나 수학 연산으로 해를 구할 수 있는 식, 또는 문제에 대한 해를 식으로 명확히 제시할 수 있는 것을 말한다.

- Invertible Matrix : 역행렬이 존재하는 행렬을 의미하며 Not invertible 이라는 말은 곧 full-rank가 아니다, 다시 말해서 어떠한 column은 다른 column들을 통해 표현될 수 있음(linearly dependent)을 의미한다. 

- trivial : 증명이 필요없을 정도로 결과가 너무 직관적이고 당연하게 유도되는 것을 의미하며, trivial solution는 대개 어떤 문제를 풀 때 고려할 필요가 없는 자명한 해를 의미한다. 이와 반대로 non-trivial solution은 고려를 해주어야 한다.

- i.e. 와 s.e. : i.e.는 라틴어 id est의 약어로 '즉' 이라는 의미를 가지며 s.t.는 such that의 약어로 '뒤에 있는 것을 만족하는' 이라는 의미를 가진다.

- $\forall, \exists, \in$ 기호 : $\forall$ 기호는 All의 A를 뒤집어 만든 모양으로, **모든 ~에 대해**를 의미하고, $\exists$ 기호는 Exist의 E를 뒤집어 만든 모양으로 **어떤 ~가 존재하여**를 의미하며 $\in$ 기호는 **~안에**를 의미한다. 

예를 들어, **$\exists 0 \in F, \forall a \in F, a+0 = a$** 의 경우                                                   
**F에 0이라고 하는 어떤 원소 a가 존재하여, F의 모든 원소 a에 대해, a+0 = a가 만족된다** 는 것을 의미한다.


## Reference 

- [QP에 대해 잘 설명한 블로그](https://velog.io/@wjleekr927/Quadratic-program#subsumes-lp)

- [Positive definite와 Positive-semi definite의 차이에 대해 잘 설명한 블로그](https://blog.naver.com/sw4r/221495616715) 

- [Rank에 대해 잘 설명한 블로그](https://blog.naver.com/sw4r/221416614473)

- [Least-Squares(LS) Problems에 대해 잘 설명한 블로그](https://deep-learning-study.tistory.com/390)
