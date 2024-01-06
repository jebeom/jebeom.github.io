---
title : "[LG Aimers] Machine Learning 개론"
excerpt: "Machine Learning의 기본 개념에 대해 알아보자"

category :
    - LG Aimers
tag :
    - machine_learning

toc : true
toc_sticky: true
comments: true

---

Machine Learning의 기본 개념에 대해 알아보자

> 본 포스팅은 LG Aimers 수업 내용을 정리한 글로 모든 내용의 출처는 [LG Aimers](https://www.lgaimers.ai)에 있습니다.

## Machine Learning 이란?
기계학습은 인공지능의 한 분야로써 실험적으로 얻은 Data로부터 점점 개선될 수 있도록 하는 알고리즘을 설계하고 개발하는 것을 말한다.

### Tom Mitchell's definition

톰 미첼은 기계학습 알고리즘을 E와 P와 T로 정의했다.

- **T(ask)** : 기계학습을 가지고 어떤 작업(ex: 분류,회귀 등)을 할 것인지
- **P(erformance Measure)** : 어떤 성능지표(ex: error rate, accuracy 등)를 사용해 평가할 것인지
- **E(xperience)** : 어떤 Data를 활용할 것 인지

### Generalization

예를 들어 Spell Checking을 하는 기계학습 알고리즘에서 Example로 학습 Data를 100개를 주었을 때, 우리는 컴퓨터가 학습 Data를 통해 특정 패턴을 배워서 학습 Data에 없는 Example에서도 Spell Checking을 하는 것을 기대하기에 기계학습에서 일반화(Generalization)는 중요한 목표라고 할 수 있다.

### No Free Lunch Theorem for ML

어떤 기계학습 알고리즘도 다른 기계학습 알고리즘보다 항상 좋다고는 할 수 없다. 다시 말해서 하나의 알고리즘이 모든 경우에 다 좋을 수는 없다. 현재 성능이 좋다고 말하는 Deep Learning 기반의 방법들도 Task가 바뀌면 알고리즘을 바꿔주어야 한다.

## 기계학습의 종류

### 1. 지도학습(Supervised Learning) 

지도학습이란 기계학습 알고리즘에게 특정 input이 들어오면 특정 output이 나와야 한다고 명시적으로 가르쳐주는 것으로 대표적인 작업에는 Classification 과 Regression 이 있다. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/71afd913-e1d2-4d50-aadd-ba365fb3fd00" width = "500" ></p>

**Classification**은 y가 범주형(Categorical) 변수이다. 예를 들어 Binary Classification 이면 Positive Class, Negative Class로 나뉘는 것이다.

**Regression**은 y가 연속형(Continuous) 변수이다. 따라서  Output이 실수로 예측이 된다.


### 2. 비지도학습(UnSupervised Learning)

비지도학습은 학습 Data가 x로만 구성되어 있기에 이러한 x를 가지고 어떤 식으로 했으면 좋겠다라고 알려준 것이 없기에 마음에 안 드는 부분이 발생할 수 있다. 비지도학습에 속하는 작업에는 Clustering, anomaly detection, density estimation 등이 있다.

### 3. 준지도학습(Semi-supervised Learning)

준지도학습은 지도학습과 비지도학습의 중간으로 몇몇 data들은 desired output, 다시 말해서 x와 y를 주는데 몇몇 data들은 x만 주는 것이다. 이러한 준지도 학습은 y값을 주는 작업을 사람이 하게 되어 시간이 많이 걸리기에 몇몇 Data에 대해서는 사람이 Labeling(y를 줌)을 하고 몇몇 Data에 대해서는 Labeling을 하지 않은 채로 두게 된다.

준지도학습에서는 주로 2개의 시나리오를 고려한다.

**LU learning** : 몇몇 Data에 대해서는 사람이 Labeling(y를 줌)을 하고 몇몇 Data에 대해서는 Labeling을 하지 않는 경우

**PU learning** : Positive data는 Positive하다고 알려주는데 Positive하지 않는 Negative한 Example에 대해서는 학습 단계에 아무런 label을 주지 않는 case이다. 다시 말해서 Positive case와 Unlabeld case 만 존재하는 경우이다.


<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/24740f5f-ea2f-49fa-bcd5-74128db13b27" width = "600" ></p>

위와 같은 그림에서 준지도학습은 Unlabel된 Data(검은색 점)가 추가된 형태로 Unlabed 된 Data 같은 경우 Label은 없지만 Label을 확률적으로 줄 수 있다. 이러한 Soft한 Label을 통해 Decision Boundary를 더 정확하게 예측가능하다. 따라서 이와 같은 과정을 통해 Classification 성능을 높일 수 있다.

### 4. 강화학습(Reinforcement Learning)

강화학습은 Dataset이 주어지는 대신에 환경이 주어져 있어 환경의 상태(State)를 보고 Agent가 Action을 선택하는데 그러면 환경이 State Transition을 통해 새로운 State와 Reward 를 준다. 이러한 보상(Reward)은 어떤 State에서 어떤 Action을 취했을 때 얼마나 좋은 Action이었는지 환경이 평가를 해서 주었다고 볼 수 있다. 이러한 환경과의 interaction 을 통해 학습을 하는 과정이다.  

많은 경우 State에서 Action을 취하면 State 변화만 있고, Reward는 주어지지 않는다. 따라서 이러한 과정을 여러 번 반복해야 Reward가 주어진다. 이렇게 Delay 된 Reward를 통해 앞서 했던 Action들을 평가하는 것이 상대적으로 어렵기에 강화학습은 학습하는 데에 걸리는 시간이나 난이도가 다른 학습에 비해 길고 높다. 

## 알맞은 Model을 찾기 위한 과정

기계학습을 할 때에 우선 학습 data를 모으는데 학습 data의 개수가 N이라고 가정하고, x(input)의 차원은 D dimension, Binary classification을 다루기 위해 y(desired outpu)는 -1과 +1로 되어 있다고 가정하자.

그 후에 모델 class를 정해야 한다. 예를 들어 $W^{T}x+b$ 라는 선형 모델을 가정했을 때 $W$와 $b$가 Parameter가 되며 이러한 Parameter들은 Training Data를 통해 결정된다.

이러한 모델이 잘 동작하는지는 Model의 예측 값과 정답 값이 틀리면 틀릴수록 큰 값을 주는 Loss Function(손실함수)를 통해 판단한다. Loss까지 정의를 한 후에 학습을 Loss를 최소화하는 Parameter $W$와 $b$를 찾는 최적화 문제로 결정할 수 있다.

### Overfitting(과적합)

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/aa44a9c3-7710-4a11-b80c-7cdc8f8a921b" width = "500" ></p>

위의 왼쪽 그림처럼 학습 data상에서 세세한 차이까지도 맞추도록 기계학습을 복잡하게 설계를 한다면 보지 못한 영역(학습 Data가 아닌 영역)에서 Up,Down이 심하다.(과적합)
 
반면에 오른쪽 그림처럼 사소한 up,down은 맞추지 않고 전체적인 패턴만 파악한다면 일반화 능력이 좋아 보지 못한 영역에서도 부드럽게 변화하는 것을 확인할 수 있다.

**True Distribution($P(x,y)$)** 이란 Data x와 Label y 와의 모든 상관관계를 표현하는 분포로 우리는 알 수 없으며 Training Set, Test Set은 True Distribution 에서 Sampling 됐다고 볼 수 있다. 또한 Training Set, Test Set등을 만들 때 Data 하나, 하나를 얻는 과정이 독립이고, Sampling 하는 과정상에서 분포가 바뀌지 않는다는 IID(independent and identically distributed)라는 가정이 존재한다.
  
**Generation Error** 이란 x와 y가 True Distribution 을 따른다고 했을 때 Loss 의 평균값, 혹은 기댓값을 의미하며 

과적합(Overfitting)은 너무 과하게 학습 데이터에 적합된 즉, 복잡한 모델의 상태로 
- Generation Error(모든 데이터상에서 에러의 평균값) > Training Error(학습 데이터상에서만의 에러) 

반대로 과소적합(Underfitting)은 다음과 같다. 
- Generation Error < Training Error

**우리는 우선 학습 data에서 잘 되는 즉, Training error가 작게 Overfitting을 한 다음, 과도한 Overfitting을 피하기 위해 Training error에 있어서 조금 손해를 보더라도 Test error를 낮춰 우리가 볼 수 없는 Generation Error를 낮춰 나가야 한다. 만약 Underfitting이 나는 경우 학습과정이나 모델에 문제가 있는 상황이다.**
  
### Regularization

학습 Data에 대해서 Loss Function을 정의하면 과적합에 빠질 수 있기에 그것을 보상하기 위한 Term 을 추가하게 되는데 이를 Regularization Term이라 한다. Regularization(정규화)이란 특정 Solution에 대한 우리의 Preference(선호)인데 만약 우리가 낮은 차수의 함수를 쓰는 것이 좋기에 낮은 차수를 선호한다면 Modeldml Capacity(함수의 차수를 의미)가 증가할수록 값이 커지는 Regularization Term을 사용해 Loss의 최소화 뿐 아니라 Model의 Capacity도 최소화 할 수 있도록 목적함수를 구성할 수 있다.

$$J(W) = (error) + \lambda W^{T}W$$

위와 같은 목적함수에는 2가지 Term 이 존재하고 이 2가지 Term은 서로 다른 목적으로 유래 되었기에 우리는 $\lambda$라는 Hyperparameter(Tuning parameter)를 통해 2가지 Term에 대한 상대 중요성을 준다. 만약 $\lambda$ 가 0이라면 첫 번째 Term만 고려하는 것이고, 만약 $\lambda$가 엄청 크다면 두 번째 Term을 더 많이 고려하겠다는 것이다.

**정리하자면  Regularization은 두 번째 Term을 추가함으로써 Training Error를 높이더라도 우리가 원하는 즉, Test Error를 줄여주는 Parameter를 얻어가는 과정이다**.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/ac61973c-396d-42e7-b8d1-7af6b0b0b855" width = "700" ></p>

위의 그림과 같이 9차수 모델을 사용한다고 했을 때 $\lambda$ 를 과도하게 크게 주면 왼쪽 그림처럼 선형 함수가 되고, $\lambda$를 적절히 주면 모든 data에 대해 적절히 fitting이 되는 smooth한 함수가 나타나고 $\lambda$ 를 거의 0으로 준다면 오른쪽 그림처럼 과적합상태가 된다.

따라서 최적의 Capacity 모델을 바로 찾거나(힘들겠죠 아무래도?) 좀 더 높은 Capacity 모델을 활용하고 대신에 Regularization을 사용해 적절한 $\lambda$값을 사용해야 한다.

### Bias and Variance


**Bias(편향)** : 우리가 볼 수 없는 Data들(예측값)의 평균값과 True값(우리가 도달하고자 하는 값)과의 차이를 의미
**Variance(분산)** : 예측값들의 평균값과 예측값 사이의 거리의 제곱의 평균값을 의미하며 True값을 알 필요 없음

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/5fa2b3bb-27e7-4720-8386-8cab4e623f4d" width = "700" ></p>

위의 그림에서 볼 수 있듯이 화살을 잘 쏘려면(좋은 모델을 만드려면) Bias 도 낮고 Variance 도 낮아야한다.

한편 학습 data를 제외한 data들에서의 에러인 (Test Error) = (Bias) + (Variance) 이기에 Generallization 혹은 Test Error를 낮추려면 Bias 도 낮고 Variance 도 낮아야한다. 그런데 Bias 와 Variance 는 상충관계(하나가 올라가면 하나가 내려감) 즉, Trade - off 관계 이기에 기계학습이 어렵다. 이 두가지 모두를 낮추기 위해 많이 활용되는 것이 Ensemble Learning 이다.

### Bias,Variance 와 Overfitting,Underfitting 사이의 관계

Bias,Variance 는 예측값 즉, Test data들에서의 개념이다. 

Overfitting 된 상태, 다시 말해서 Model의 Capacity(함수차수)가 증가하면 증가할수록 예측값들의 에러 즉, Test Data Error가 커지므로 분산(Variance)이 커지게 된다. 이에 반해 Bias가 높다는 것은 영점이 맞지 않다 즉, Underfitting된 상태라는 것이다. 예를 들어 선택한 Model이 선형함수여서 Capacity가 떨어져(모델의 복잡도가 낮다) Training Error를 어느정도 이상 줄이지 못하는 경우이다.

따라서 모델의 복잡도가 올라갈수록 Bias는 줄어들지만 Variance가 올라갈 확률이 높아져 앞서 이야기한 Trade-off가 존재하는 것을 확인할 수 있다. Variance가 크면 학습 Data를 모아 줄일 수 있지만 Bias는 Data를 모아준다고 하더라도 도움이 안되고 이 경우에는 모델의 복잡도를 높이는, 즉, 함수를 고차함수로 사용해야 한다.

제일 좋은 것은 Test set에서 나올 수 있는 모든 경우에 대해 학습 Data를 모은 경우 그냥 Overfitting 만 시키면 문제가 쉽게 풀리지만 그렇지 않기에 앞서 나온 Regularization 등의 방법으로 Overfitting을 줄인다.
