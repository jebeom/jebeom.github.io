---
title : "[LG Aimers] 지도학습(분류/회귀) [1]"
excerpt: "ML의 한 부류인 지도학습(분류/회귀)에 대해 알아보자"

category :
    - LG_Aimers
tag :
    - supervised_learning

toc : true
toc_sticky: true
comments: true

---

ML의 한 부류인 지도학습(분류/회귀)에 대해 알아보자

> 본 포스팅은 LG Aimers 수업 내용을 정리한 글로 모든 내용의 출처는 [LG Aimers](https://www.lgaimers.ai)에 있습니다.

## 지도학습

지도학습에서 사용하는 data sample은 입력 x와 출력 y의 쌍으로 구성되어 있는데 이때 출력 y를 label 또는 정답이라고 말한다. 이러한 지도학습은 크게 model의 training, 과 test 단계로 구분된다. 

### Training & Test

지도학습의 경우 y 즉, 학습 sample의 label이 주어지므로 model의 parameter값을 변경해가며 model output과 정답의 차이인 error를 줄여가며 학습이 진행되며 Test 단계에서는 우리가 선정한 모델이 Training 단계에서 보지 못한 입력값(Unseen data)을 사용하며 이러한 Test단계를 통해 최종적으로 모델의 성능을 평가한다.

### Purpose

모든 Data에 대해서 잘 맞아 떨어지는 Target Function 을 만드는 데에 있어서 한계가 있기에 수백 ~ 수천의 sample로 구성된 Dataset을 활용해서 Target Function에 근접하는 Hypothesis H함수 즉, Machine Learning Model을 만들어야 한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/34dc7fda-3489-4a9b-97da-9de4022a9884" width = "500" ></p>

위 그림과 같이 모델의 복잡도가 늘어나면 Overfitting이 발생하기 쉬워져 Trainig Error와 Test Error의 간격이 커지고(Variance 증가) 반대로 모델의 복잡도가 감소하면 Underfitting이 발생하기 쉬워 Bias가 높아진다. 따라서 Machine Learning Model을 만듦에 있어서 위의 그림의 점선과 같이 적절한 복잡도의 모델을 선정해 Generalization Error를 최소화 해야한다.

## Linear Regression

Regression은 model의 출력이 연속인 값을 갖게 되므로 연속되는 출력을 예측하고 추론하기 위해 data set에서 입력과 정답으로 구성되어 있는, 다시 말해서 label이 있는 data set을 사용한다.

선형(Linear)모델은 단순하고, 입력 변수가 출력에 얼마나 영향을 주는지를 대략적으로 파악이 가능하다는 장점이 있으며, 일반화 즉, 성능이 높지는 않더라도 다양한 환경에서 안정적인 성능의 제공이 가능하다.

또한, 선형 모델의 경우 모델의 형태 즉, 가설(Hypothesis)함수가 $\theta_{0}+\theta_{1}X$ 형태로 정해지고, 손실(Loss)함수 역시 MSE(최소제곱)를 사용하기에 Parameter($\theta_{0},\theta_{1}$)들을 Optimization만 해주면 된다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/30fc84f7-312e-4571-a0b3-9d4720e14962" width = "700" ></p>

위의 그림에서 확인할 수 있듯이 입력 vector x는 D-dimenstion이지만 $\theta_{0}$를 포함하기 위해 1을 붙여주고 N개의 데이터가 있기에 입력 데이터 X 행렬의 차원은 NxM이고, N개의 sample마다 하나씩 정답(y)이 있기에 target vector y의 차원은 N-dimension이다. 마지막으로 parameter vector $\theta$는 $\theta_{0}$ 부터 $\theta_{d}$까지 있기에 차원은 D+1 이다.

이러한 $\theta$와 $X$의 선형결합을 통해 Score($X\theta$)를 계산할 수 있으며 Score 값과 y의 차이의 제곱을 평균낸 것을 손실함수(Loss Function) 혹은, 비용함수(Cost Function)이라고 한다. 최적의 Parameter들은 이러한 Cost Function을 가장 최소화 하게 만드는 값이며 손실함수가 미분가능하고, Convex 함수라는 가정하에 $\theta$에 관한 Derivative Term을 구하고 이 방정식이 0이 되도록 하는 $\theta$값을 구함으로써 Parameter값을 찾을 수 있다.

위와 같은 과정을 Least Square Problem이라 하고 방정식을 Normal Equation이라 부르며 방정식을 Solve 하는 과정은 아래 그림과 같다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/ad6cb9b9-5951-4e19-a3ef-ddbde6af6da1" width = "700" ></p>
<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/e9b40857-a4ba-4451-94b3-b821bed48ba8" width = "700" ></p>
<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/951e7c54-8bac-43d3-8e42-364b22c03d96" width = "700" ></p>

따라서 앞서 구한 Matrix에 y를 곱하는 하나의 Step을 통해 Parameter를 구할 수 있지만 Data의 수 즉, N이 커지면 Matrix 의 inverse 연산이 복잡해지고 역행렬이 존재하지 않을 수도 있기에 iterative 알고리즘인 **Gradient Descent** 방식을 사용한다.

## Gradient Descent
여기서 Gradient란 함수를 미분하여 얻는 term으로 해당 함수의 변화하는 정도를 표현하는 값이다. 이러한 Gradient가 0인 지점에서 값이 최소가 된다. 따라서 Gradient Descent 는 Gradient가 0인 지점까지 반복적으로 Parameter $\theta$를 바꾸어 나가며 최적의 Parameter 값을 찾는다.

Gradient Descent의 과정은 다음과 같다.
- 1. 최적화하고자 하는 Loss Function 또는 Objective Funciton을 세운다.
- 2. 사전에 Step size인 $\alpha$와 같은 알고리즘 수행에 필요한 Parameter를 설정한다. 이렇게 사전에 설정한 Parameter를 Hyperparameter라고 하며 항상 양의 값을 가진다.
- 3. 수렴하기 까지 다음의 식을 수행한다. $\theta_{j} = \theta_{j} - \alpha \frac{\partial}{\partial\theta_{j}} J(\theta_{0},\theta_{1})$

Linear Regression 에서 Gradient Descent 알고리즘을 수행하면 다음과 같다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/22834feb-c373-480c-a53f-a172003c5b27" width = "700" ></p>
<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/79a08271-bbb2-46e3-b7ca-bd983473f3d6" width = "700" ></p>

### Local Optimun과 Global Optimum의 차이

Gradient Descent 알고리즘은 경우에 따라 Local Optimun만을 달성하기 쉽다. Global Optimum은 전체 Error Surface에서 가장 최소인 값을 갖는 지점을 의미하며 반대로 Local Optimun는 특정 지역에서는 최소가 될 수 있지만 전체 영역에 대해서는 최소가 되지 않을 수 있는 지점을 의미한다. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/922de71d-6147-4b0e-98d8-7f59556ef48c" width = "700" ></p>

위 그림에서도 확인할 수 있듯이 현재 지점에서 가장 가파른 변화도를 보고 움직이기에 서로 다른 두 시작점에서 Gradient Descent를 수행하는 경우 어느 하나는 Global Optimum 을, 다른 하나는 Local Optimun을 가질 수 있다.

### Batch Gradient Descent

Linear regression model 에서 목적 함수 J의 partial derivative term을 넣어서 $\theta_{0}$ 와 $\theta_{1}$ 을 바꾸어 나가는데 Batch Gradient Descent의 경우 $\theta_{0}$ 와 $\theta_{1}$을 업데이트 하는 과정에서 각각 모든 Data에 대해 계산해야 하므로 Data 의 개수가 증가할수록 복잡도가 커진다는 단점이 있다.

### Stochastic Gradient Descent

Batch Gradient Descent의 문제를 해결하기 위해 전체 데이터(Batch) 대신 일부 데이터의 모음(Mini-Batch)을 사용하여 Loss Function을 최소화 시키는 알고리즘으로 빠르게 iteration이 가능하지만 적은 샘플에 대해서만 parameter값을 업데이트 하기에 noise의 영향을 받기 쉬워 Oscillation이 발생할 수 있다는 단점이 있다.

## Gradient Descent with Momentum
### Gradient Descent 알고리즘의 단점

Gradient Descent 알고리즘은 시작하는 Point에 따라서 Global Optimum이 아니라 Local Optimum에 빠질 수 있다는 단점이 있다. 따라서 다양한 변형 알고리즘들이 개발되었는데 그 중에서도 가장 대표적인 것이 바로 **Momentum**을 이용하는 것이다.

**Momentum**은 과거에 Gradient가 업데이트 되어오던 방향 및 속도를 어느 정도 반영해서 현재 포인트에서 Gradient가 0이 되더라도 계속해서 학습을 진행할 수 있는 동력을 제공한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/044c02b6-3411-4a91-b45e-370b31a1e9b9" width = "600" ></p>

위 그림과 같이 현재의 Momentum은 과거의 Momentum에 $\rho$만큼의 값을 곱하고 과거의 Gradient Term들을 누적해서 계산해주는데 현재의 값에서부터 멀수록 $\rho$값이 연속적으로 곱해지는데 $\rho$값은 1보다 작기에 먼 과거의 값은 더욱 작아지고 비교적 가까운 값들은 비교적 적게 작아지게 된다. 따라서 이러한 Momentum을 이용해 현재 포인트에서의 Saddle Point나 작은 Noise Gradient 같은 변화에 보다 안정적으로 수렴할 수 있다.

### Nestrov Momentum

Nestrov Momentum은 Momentum을 이용하는 Gradient Descent에서 조금 더 발전한 방식으로 기존의 방식과 다르게 Gradient를 먼저 평가하고 업데이트 한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/23a2da1a-5d9f-407f-a97d-4866668faee4" width = "600" ></p>

위의 오른쪽 그림과 같이 momentum step만큼 이동을 하고 그 지점에서 lookahead gradient step을 계산하여 actual step을 계산한다.

## Local Minimum을 피하기 위한 다른 방법
### AdaGrad, RMSProp, Adam

이외에도 Local minimum을 피하기 위해 r을 업데이트 할 때 gradient의 제곱을 그대로 곱하는 AdaGrad가 있다. 다만 이 방식은 gradient의 값이 계속해서 누적이 됨에 따라서 learning rate의 값이 작아져 학습이 일어나지 않게 된다. 이러한 단점을 보완하기 위한 방법에 RMSProp 방법이 있다. RMSProp은 r을 업데이트 할 때 기존의 r에 $\rho$ 값을 곱하고 $(1-\rho)$를 gradient의 제곱에다가 곱함으로써 gradient 값이 누적됨에 따라 $\theta$ 값이 줄어드는 것이 아닌 완충된 형태로 학습속도가 줄어들게 된다.

앞으로 우리가 가장 많이 접하게 될 Gradient Descent 알고리즘은 Adam일 텐데 Adam(Adaptive moment estimation)은 RMSProp과 Momentum 방식을 혼합한 방법이다.

과정은 아래 그림과 같다. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/9cbd5405-b1b5-46ef-ad77-f1ef3f74b12b" width = "700" ></p>

### Optimizer 계보

지금까지 Gradient Descent, Momentum, Adam 등 Loss Function을 최소화 하는 Parameter들을 구하는 다양한 Optimizer들에 대해 알아보았다. 아래의 그림을 통해 어떠한 방법으로 Optimizer가 개선되었는지 확인할 수 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/d387d80a-d6ea-4980-a0ec-ebdc7c08fbe6" ></p>

## Model을 학습할 때 유용한 Tip들 
### Learning rate Scheduling 방법

아래 그림과 같이 low learning rate의 경우 천천히 수렴하지만 loss를 줄일 수 있다. 반대로 high learning rate의 경우 학습이 진행되며 수렴하는 정도가 low learning rate보다 줄어들지 않지만 빠르게 학습을 진행할 수 있다. Hyperparameter인 Step size $\alpha$를 학습 과정에 따라 줄여나가면 초기에는 학습을 빠르게 진행할 수 있지만 이후에 $\alpha$ 값을 늘리게 되면서 Loss가 줄어들지 못하는 문제를 Learning rate를 점차적으로 줄임으로써 해결해 학습을 용이하게 할 수 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/411d4f35-298f-4554-817a-e94af53f5372" ></p
>
### Model의 Overfitting 문제를 해결하는 방법

Overfitting 문제를 해결하는 대표적인 방법에는 **Regulalization**가 있다.

- Regulalization 관련 내용은 이전 포스팅에서 설명했기에 생략하겠습니다. [여기](https://jebeom.github.io/lg_aimers/machine_learning/)를 참고해주세요 !!


