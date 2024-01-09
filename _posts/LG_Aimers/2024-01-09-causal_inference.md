---
title : "[LG Aimers] 인과추론"
excerpt: "인과성에 대해 추론하고 경험적 데이터를 사용하여 인과 관계를 결정하는 방법에 대해 알아보자"

category :
    - LG_Aimers
tag :
    - causal_inference

toc : true
toc_sticky: true
comments: true

---

인과성에 대해 추론하고 경험적 데이터를 사용하여 인과 관계를 결정하는 방법에 대해 알아보자

> 본 포스팅은 LG Aimers 수업 내용을 정리한 글로 모든 내용의 출처는 [LG Aimers](https://www.lgaimers.ai)에 있습니다 

## 인과성(Causality)

인과성이란 하나의 어떤 무엇인가가 다른 무엇을 생성함에 있어서 영향을 미치는 것 원인과 결과의 관계를 기술한 것이며 인과추론은 인과적인 통찰을 이용해서 하는 모든 추론을 말한다.

### 인과성과 데이터 사이언스의 연관성

- 강화학습 : 주어진 상황에서 어떤 행동을 취할지 학습하는데 환경에 변화를 줘서 원하는 상태로 변화시키는 인과관계로 해석이 가능하다.

- 기계학습 : 데이터의 상관성 학습이 목적

- 데이터 사이언스 : 수집하고 분석한 데이터로 대중들과 소통할 때 상관성과 인과성을 모두 고려해야함

### Pear's 인과계층

Pear's 인과계층은 총 3단계로 이루어져 있다.

- Level 1.관측 계층 (Associational or Observational) : 시스템을 건들지 않고 그대로 관찰하면서 변수들의 상관성 관찰한다.

- Level 2.실험 계층 (Interventional or Experimental) : 실험을 함으로써 나오는 결과를 관찰한다. 

- Level 3.반사실적 계층 (Counterfactual) : 관측값과 실험 통제를 통해 나온 결과가 아닌 실험 통제를 하지 않았으면 어떠한 결과가 나왔을지를 생각한다.


<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/07f66e17-5e04-4792-8c0c-5778239cd914" ></p>  


### 데이터 분석 시 고려요소

데이터를 분석하고자 할 때 다음과 같은 두 가지 요소를 고려해야한다.

- 주어진 데이터가 상관성을 가지고 있는지 인과성을 가지고 있는지
- 우리가 알고자 하는 것이 조건부확률 같은 상관성인지 아니면 인과성인지


### 계층을 넘나드는 추론

인과 추론은 우리가 알 수 없는 실험 결과를 관측 데이터와 연결하는 것이기에 블랙박스에 대한 형식적인, 수학적인 이해가 필요하다.

따라서 아래 그림과 같이 모든 관측가능한 변수들의 값을 생성해내는 인과적 메커니즘인 SCM모델이 탄생했다. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/df6b416d-f495-426d-83e9-96c3477edb43" ></p>

SCM 모델을 기반으로 우리가 어떤 중재도 하지 않을 경우 관측 가능한 모든 변수들에 대한 관측 분포를 볼 수 있고 우리가 임의의 변수를 중재하게 될 경우 즉, 실험을 할 경우에는 실험에 대한 결과 분포가 나오게 된다.

여기서 U는 exogenous variable 즉, 관측되지 않는 변수를 의미하고 P(U)는 관측되지 않는 변수에 대한 불확실성을 의미한다. 또한 V는 endogenous variable 즉, 관측 가능한 변수들의 집합을 의미한다. F는 관측할 수 있는 각각의 변수들에 대해서 값들이 어떻게 정의되는지를 함수로 정의한다. 이 함수의 인자에는 U나 V의 subset(하위 집합)이 있다.


## 인과 효과 계산 방법

우리에게는 주어진 관측 데이터가 있고, 이 데이터는 상관성만을 지니고 있는데, 우리가 원하는 Causal Effect를 계산하기 위해서는 Causal Diagram에 내포되어 있는 모든 정보를 이용해서 인과 효과를 계산한다.

이러한 정보들에는 변수들 간의 인과관계, 인과관계를 통해서 그려진 그래프에서 나타나는 조건부 독립성 등이 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/9ff4a2a0-44ea-4dc1-8cda-022612d1d625" ></p>

위와 같은 그래프 예제의 2번에서 season과 wet 사이에는 sprinkler외에도 rain을 통해 지나갈 수 있으므로 Conditional Independence하다고 볼 수 없는 반면 3번에서는 Rain과 slippery 사이에서는 어떠한 path든 모두 wet을 지나야하므로 이 경우 wet이라는 정보가 주어져 있으면 path가 막혀있다고 말하며 이 경우 Rain과 slippery는 Conditional Independence하다고 볼 수 있다.

참고로 어떤 변수 y가 집합이 아닌 x라는 변수를 중재를 했을 때 어떠한 식으로 변할 것인지, 다시 말해서 인과 효과는 $P_{x}(Y) = P(y\|do(x))$로 표시한다.

또한 그래프에 나타나는 모든 변수들이 관측 가능한 변수들일 때, 결합 확률 $P(V)$는 아래 식으로 표현이 가능한데 만약 Sprinkler를 중재 즉, Spinkler가 켜져 있으면 아래 식과 같이 결합확률을 계산하한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/60423b7c-cb11-4c5f-a104-2032dc1591a3" ></p>

만약 관측 되지 않은 변수들이 존재할 때는 어떻게 해야할까?

똑같은 문제에서 Season이 관측되지 않고, sprinkler에 중재를 가하면 인과 효과를 아래 그림과 같이 구할 수 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/7f0c1c82-f6b4-44e8-bd2a-37a9d4cc5884" ></p>

### Back - door

아래 그림과 같이 X와 Y의 상관성은 2가지의 원인으로 생각해볼 수 있다. 첫번째로는 X가 Y에 직접적으로 연결되어 있기 때문에 나타나는 인과적인 상관성이 있고, 두번째로는 신장결석(Z변수)에 따라서 X와 Y변수에 영향을 미치기에 Z(교란변수)에 의한 상관성이 있다. 참고로 Z변수는 X와 Y에 서로 영향을 미치는 교란 변수, W변수는 X와 Y 중간에 있는 변수로 mediator라고 한다. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/2fe70eed-e54e-4459-8bc9-26368b065856" ></p>

인과효과를 계산한다는 것은 X가 변하면 Y가 어떻게 변할지에 대해 생각하는 것 이므로 교란에 의한 다시 말해서 Z변수에 의한 제거하고자 하는 것이다. 이러한 방법을 Back - door 방법이라고 하며 위의 식처럼 표현이 가능하다.

아까의 예제에서 Sprinkler(X)에 대해서 wet(Y)이 어떻게 바뀔 것인가에 대한 인과효과를 계산할 때, 여기서 Z변수는 {season}, {rain}, {season,rain} 들이 될 수 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/de459052-9375-4cdd-91c3-8b58ac332fc3" ></p>

위와 같은 복잡한 예제에서는 $z_{1}, z_{2}, z_{4}$의 뒷문이 존재하는데 이 3가지 뒷문을 모두 막을 수 있는 집합 $z$를 찾는게 목표이며 집합 z에는 {$z_{1}, z_{4}$} 이 포함된다.

### Do - calculus

Do - calculus는 Back - door Criterion의 문제점을 해결하는 것으로 여러 가지 다른 중재 조건에서 나오는 확률들끼리 서로 연결고리를 만들어주고, 서로 다른 중재로 확률 분포를 바꾸어주는 역할을 하며 3가지의 Rule로 이루어져 있다.

- Rule 1 : Adding / Removing Observations : 관찰에 대한 것이 추가되거나 삭제될 수 있다. 다시 말해서 조건부 독립은 중재된 상황에서도 적용 가능하다.
- Rule 2 : Action / Observation Exchange : Action과 Observation을 바꿀 수 있다.
- Rule 3 : Adding / Removing Actions : Action이 추가되거나 제거될 수 있다. 다시 말해서 확률을 계산하는 데에 있어서 Action은 아무런 영향을 주지 않는다.

정리하자면 Do - calculus는 어떤 규칙에 의해 조건부 독립을 만족하면 확률을 다른 확률로 변경시킬 수 있으며 Sound & Complete 하다. 즉, 모든 Identifiable 한 Formula에 대해서 인과 효과를 계산하는 식을 Do - calculus 와 공리를 이용해서 이끌어낼 수 있다.


## 인과 추론의 다양한 연구 방향

지금까진 주어져 있는 도메인에서 관측 데이터와 인과 효과를 얻으려고 했다. 하지만 여러 종류의 데이터를 한번에 활용하면 좋지 않을까? 또 데이터는 다양한 특성들을 가지고 있는데 이러한 특성들을 다 고려해서 원하는 인과 효과를 계산하는 방법에 대해 알아보자.

### Generalized Identifiability Example

만약 두 개의 실험 데이터가 있다고 가정하자. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/9c7fd14b-2fd9-41bd-8618-4753730bab16"></p>   

위 예제에서 $x_{1}$은 혈압을 치료하는 약, $x_{2}$는 당뇨를 치료하는 약, B는 혈압에 대한 정보, Y는 심장병이며 약을 동시에 2개 복용했을 때 심장은 어떻게 반응할 것 인지가 궁금하다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/0f4a8217-f98d-40f5-870e-68b39c2a472f"></p>

우리는 Do - calculus와 d-seperation을 통해 위와 같은 식으로 인과 효과를 계산할 수 있다.

### Transporability

Machine Learning 에서는 Training과 Test가 같은 환경에서 나온 것을 가정하는데 이는 같은 도메인을 가정한다는 말과 같다. Transporability는 주어져 있는 데이터의 소스와 우리가 인과 효과를 계산하고자 하는 타겟이 서로 다른 도메인일 때의 인과추론을 다룬다.

실험이 이루어 지고 있는 환경인 소스(Source)가 있다고 하자. 이곳에서 데이터가 만들어지는데 첫 번째 가정으로 source와 Target이 같다면 실험실에서 나온 실험 결과를 Target에 그대로 적용가능하다. 두 번째 가정으로 source와 Target이 다르다면 not transportable 하다. 

따라서 Source와 Target에는 공통점이 존재하지만 부분적으로 어떤 변수에 대해서는 다를 수 있음을 인정하고 인과 추론을 어떻게 할 것인지가 Transporability의 목적이다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/9b21b25b-4d61-477a-bb0d-a1887cac8d28"></p>

위 그림을 통해 Source 와 Target 간에 차이점이 존재하는 방식에 따라 Transporability 개념을 통해 차이점을 표현하고 원하는 인과 효과를 다르게 얻을 수 있음을 알 수 있다.

보통 무작위 실험을 통해 얻는 결과를 절대적으로 받아들이는 경우가 많은데 실험이 일어난 모집단과 인과효과를 적용하고자 하는 집단 간의 차이가 존재한다면 실험 데이터의 인과 효과를 타겟에 사용할 수 없다. 따라서 Transportability개념을 통해 Causal Diagram을 그리고 어떤 변수들이 다른지 명시화하고 알고리즘을 통해 문제를 풀어야한다.

### Sampling 과정에서 선택편향이 발생한 경우

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/9ce5ac1f-7f68-4826-9c5b-9e4882931c18"></p>

위 그림은 기존의 Causal Diagram에 새로운 변수 S가 추가된 것인데 여기서 S변수란 샘플이 데이터에 포함되었는지 아닌지를 표현하는 변수로 1이면 포함되어 있고, 0이면 포함되어 있지 않은 것이다. 왼쪽 그림은 무작위 상황에서의 데이터의 분포와 같고, 반대로 편향이 있는 오른쪽 그림은 무작위 상황에서의 데이터의 분포와 같다고 할 수 없다.

여러 가정들을 적용함으로써 인과 효과가 편향없이 계산될 수 있는지 확인해보고 계산해야한다.


### Missing Data

센서가 일시적으로 동작하지 않거나 설문에 공란이 있는 등 데이터가 누락된 경우 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/8da7eeb9-709a-486e-b520-50f613d0ac6b"></p>

위 그림과 같은 방법으로 누락된 데이터로부터 누락되지 않은, 즉, 편향되지 않은 상태에서의 결합 확률을 구할 수 있다. 여기서 $O$는 누락되는 변수 다시 말해서 예제 상에서는 실제 학생의 비만도이고, $R_{0}$는 누락되는 메커니즘으로 $R_{0}$값이 1이 되면 누락이 되어 missing 된 값이 $O^* $에 들어가고 0이 되면 누락이 되지 않는 것으로 비만도가 실제 비만도 $O$가 $O^* $에 들어간다. 마지막으로 $O^* $ 는 실제 리포트 된 비만도이다.


<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/8fb1b03b-c162-4c76-9195-1dd4d22fbd43"></p>

위 그림을 통해 비슷한 방법으로 누락된 데이터로부터 인과 효과를 계산할 수 있다. 즉, Causal Diagram의 누락 메커니즘이 어떤 식으로 일어났는지 가정에 의해서 표현을 하고 그림을 통해 위처럼 식을 전개 가능하다.















 
