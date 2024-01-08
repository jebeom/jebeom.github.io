---
title : "[LG Aimers] 지도학습(분류/회귀) [2]"
excerpt: "지난 포스팅에 이어 지도학습(분류/회귀)에 대해 알아보자"

category :
    - LG_Aimers
tag :
    - supervised_learning

toc : true
toc_sticky: true
comments: true

---

지난 포스팅에 이어 지도학습(분류/회귀)에 대해 알아보자

> 본 포스팅은 LG Aimers 수업 내용을 정리한 글로 모든 내용의 출처는 [LG Aimers](https://www.lgaimers.ai)에 있습니다.

## Linear Classification

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/0dc388cb-b2e0-4187-97e3-43f4e5156367" ></p>

Classification은 모델의 Output이 discrete(이산적인)한 값을 가지며 위 그림과 같이 Linear Binary Classification은 2차원 Coordinate을 구분하는 Hyper plane(직선)이 존재한다. Multiclass Classification의 경우 입력 신호 공간에서 Hyper plane이 여러개 존재한다. 이러한 Hyper plane에 Sample을 입력했을 때 양수 즉, $h(x)>0$ 이면 Positive sample 이 되고 반대의 경우 Negative sample이 된다. 

Linear Classification은 다음과 같은 방법으로 진행된다.
- Hypothesis함수 설정
- Loss Fuction 선정
- Parameter 최적화 

### Classification Hypothesis 함수 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/2f4caa14-2d82-48af-8f3b-ea5be4c8c4d6" ></p>

Hypothesis 함수는 위 그림과 같이 입력변수와 Parameter의 곱으로 Score를 계산한 후 그 출력에 Sign함수를 적용한다. 여기서 Hyper plane을 구성하는 model parameter는 $w$이며, $w_{0}$는 Bias 또는 offset값이다. 또한 Score에 y를 곱한 것을 Margin이라 하는데 이러한 Margin을 통해 Model이 얼마나 정확한지를 알 수 있다. 

예를 들어서 Score값이 (+), y값이 1인 경우 model이 Positive sample 이라 예측을 했는데 실제 정답이 1이므로 우리가 정답을 맞춘 것이기에 margin값이 굉장히 늘어나게 된다. 반대로 Score값이 (-), y값이 1인 경우 model이 Negative sample 이라 예측을 했는데 실제 정답이 1(Positive Sample)이므로 Margin이 (-)값을 가지게 되어 이 경우 Model Prediction이 실패했음을 의미한다.

sign함수를 통한 분류 예시는 아래 그림과 같다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/457fbfb2-5aad-4804-aa01-1eac5e8b0c76" ></p>

우리의 목표는 이러한 Parameter $w$를 학습시키는 것인데 $w$가 바뀜에 따라 Positive sample 이 Negative sample로 판단되는 Classification Error가 발생할 수 있다. Classification 문제에서 이러한 Error는 Zero-one loss 등을 통해 해결가능하다.

### Classification Loss Fuction 

Regression 문제에서 Loss Function으로 MSE를 사용한 것과 달리 Classification 문제에서는 Model Paramter를 fitting 하기 위한 Loss Fuction으로 Zero-one loss, Hinge loss, Cross-entropy loss 등을 사용한다.

**Zero-one loss**는 내부의 logic을 판별하여 맞으면 0, 틀리면 1을 출력하는 함수로 gradient descent 알고리즘에 적용하려면 이러한 loss function의 partial derivative term을 구해야하는데 Zero-one loss의 경우 미분하면 0 즉, gradient가 0이 되기에 학습을 할 수 없게 된다. 따라서 이러한 문제를 해결하기 위해 Classification에서는 hinge loss를 사용한다. 

**hinge loss**는 1-(margin) 값과 0 중에 큰 값을 갖는다. 만약, margin값이 크다, 다시 말해서 모델이 정답을 잘 맞춘 경우에는 1-(margin) 이 음의 값을 가지므로 이때의 Loss 값은 0을 가지게 된다. 반대로 margin이 음의 값을 갖게 되면 loss값은 1-(margin)값을 출력하여 선형적으로 증가한다.

**Cross-entropy loss**는 Classification 모델을 학습하는데 가장 많이 사용하는 Loss Function이다. cross-entropy는 서로 다른 $p$와 $q$의 tmf 거리를 의미한다. 즉, 두 개의 서로 다른 Parameter $p$와 $q$가 유사한지 아닌지의 정도에 따라 에러의 정도가 달라진다. $p$와 $q$ 가 서로 유사하다면 loss는 줄어들게 되고 다를 경우 loss는 올라가게 된다. 그런데 우리가 구한 Score값은 실수이기에 Cross-entropy에서는 확률 함수를 통해 확률값으로 Mapping을 해주어야한다.

이러한 Mapping에 사용하는 함수가 Sigmoid 함수이다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/b83efef8-32a7-4944-b49f-dd5465dfeef8" ></p>  

위의 그림처럼 Sigmoid 함수의 y축은 확률값(0~1)인 것을 확인할 수 있으며 우리가 구한 Score를 Sigmoid에 대입해 Mapping 즉, Score값을 0~1사이 값으로 변환 시켜준다. 이러한 형태를 **Logistic Model** 이라고 한다.

### Multiclass Classification

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/b7c354b4-d292-49e3-adb4-ba054535bc0b" ></p>

Multiclass classification은 위 그림과 같이 Binary linear Clasffier 개념을 확장해 A,B,C 의 Score값을 얻을 수 있는데 이러한 Score값을 Sigmoid함수에 대입해 확률 값으로 매핑할 수 있다. 또한 해당위치에 1을 signaling 함으로써 label의 정보를 기록하는 one hot encoding이라고 하는데 이렇게 one hot encoding 된 label값과 Sigmoid 함수가 출력하는 확률값을 서로 간에 비교하며 Loss function을 통해 error를 계산함으로써 학습을 할 수 있게 된다.


## Advanced Classification

Hyper plane을 통해 Classification을 하는 것을 알겠는데 이러한 Hyper plane을 어떻게 그어서 sample들을 classificaton해야할까 ?

이에 대한 답으로 SVM(Support Vector Machine)이 있다. 다음으로는 SVM을 통해 Positive sample과 Negative sample 사이에 어떠한 방식으로 Hyper plane을 그어야 할지 알아보자.

### SVM(Support Vector Machine)

Support vector란 아래 그림과 같이 Positive sample, Negative sample 각각 Hyper plane과 가장 가까운 거리에 있는 sample을 의미하며 이러한 Support vector 끼리의 거리를 가장 최대화 하도록 하는, 다시 말해서 Support vector 의 중간에 Hyper plane이 위치해 Maximum Margin을 설정하는 방식이 SVM이며 이러한 SVM은 Outlier(다른 Data들과 특출나게 차이 나는 Data)들에 대해서도 안정적인 성능을 제공한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/2b1575b7-1a4f-4f88-be06-f9dbb52f20cf" ></p> 

SVM에서는 Hard margin SVM, Sorf margin SVM, Nonlinear transform & kernel trick 등 다양한 최적화 방식을 사용한다.


### Hard Margin SVM

아래 그림과 같이 Hard Margin SVM에서는 margin 즉, support vector 간 거리를 최대화하기 위해서 $\|\|w\|\|$를 최소화 해야 하는데 이 문제를 Convex 문제로 바꾸어 생각해보면 $\|\|w\|\|^2$ 을 최소>화 해야하고, Constraint 는 $y_{i} (w^{T} x_{I} + b) \geq 1$ 이기 때문에 Constraint Optimization 문제로 생각해볼 수 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/7617b624-d7e4-4016-b683-5a7227b753b3" ></p>
<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/6359f2a6-def9-47ed-9eb4-627ba94fca22" ></p> 

만약 데이터 sample들이 linearly sepable 하지 않다고 했을 때 Kernel함수를 사용하여 차수를 높여 linearly sepable하게 만들어 non-linear classification을 문제를 풀 수 있다. 

### ANN(Artificial Nueral Network)

ANN은 인간의 뇌 신경을 모사한 형태로 만들어졌고, non-linear classification 모델을 제공하며, DNN(Deep Nueral Network)의 기본이 된다. 최근에는 미분을 해도 1로 유지되는, 다시 말해서 gradient값이 1로 유지되어 학습량이 줄어들지 않는 ReLu가 ANN의 activation function으로 사용되고 있다. 

ANN에서는 non-linear 함수들이 계층적으로 쌓여감에 따라서 signal spcae에서의 복잡한 신호들의 패턴들을 조금 더 정확하게 분류할 수 있다. 다시 말해서 linear clssification에서 수행할 수 없는 XOR 문제와 같이 non-linear한 문제들을 풀 수 있다.

하지만 학습을 진행하며 계층(layer)이 깊어질수록 gradient값이 계속해서 줄어들어 깊은 계층에 대해서는 학습이 효과적으로 진행되지 않는 Gradient Vanishing Problem이 발생할 수 있다. 이러한 문제는 최근에서야 CNN(Convolutional neural networks)과 같은 방법으로 해결이 가능해졌다.


## Ensemble Learning

Ensemble Learning은 머신러닝에서 알고리즘의 종류에 상관없이 서로 다르거나, 같은 메커니즘으로 동작하는 다양한 머신러닝 모델을 묶어 함께 사용하는 방식으로 다양한 model의 각 장점을 살려서 예측성능을 올려주어 Supervised Learning Task 에서 성능을 올릴 수 있다.

또한 독립적으로 동작하는 여러 개의 model의 결정으로 최종 예측 결과를 제공하기에 model parameter의 튜닝이 많이 필요없고, noise로부터 보다 안정하고 쉽게 구현이 가능하다는 **장점**이 있다. 하지만 다양한 모델들을 혼합해 사용하기에 compact한 표현이 되기는 어렵다는 **단점**이 있다.

### Bagging

Bagging은 Ensemble을 구성하는 가장 기본적인 요소 기술 중 하나로 학습 과정에서 Training sample을 랜덤하게 나누어 선택해 학습하는 것이다. 다양한 Classifier들이 랜덤하게 선택 된 다양한 sample들로 학습이 되어 같은 모델이더라도 다른 특성을 학습할 수 있다. Bagging을 통해 Lower Variance의 안정적인 성능을 제공할 수 있다.

앙상블은 서로 다른 model의 형태로 다르게 동작하기에 같은 sample을 사용할 수 있다. 따라서 이러한 과정을 m번 반복하면 1개의 데이터 세트로 m개의 데이터 세트를 사용하는 효과가 있기에 noise에 robust(안정)하게 된다.

### Boosting

Boosting은 Bagging과 같이 Ensemble을 구성하는 가장 기본적인 요소 기술 중 하나로 Classifier의 결과를 다음 Classifier가 학습할 때 사용하는 것이다. 즉, 어떤 sample이 중요한지 결정을 가능해서 중요한 sample에 weight를 주어 다음 Classifier의 성능을 향상시키는데 도움을 준다. 대표적인 Boosting 알고리즘으로 Adaboost가 있다.

또한 이러한 Bagging과 Boosting을 활용한 대표적인 알고리즘에는 Random Forest가 있으며 kaggle과 같은 대회에서도 우수한 성능을 낸다. 

## 모델의 성능 평가 방법

지도학습에서 모델의 성능을 평가하는 방법에는 다음과 같은 것들이 있다.

**1. 정확도(accuracy)측정** : Confusion Matrix에서 대각 성분을 합한 값을 전체 성분으로 나눈 값으로 계산한다.

**2. Precision, Recall값 측정**

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/ebc13057-9371-44a5-a01a-d114bdee7f06"></p>  

위 그림에서 확인할 수 있듯이 Positive로 예측했지만 실제론 Negative인 False positive error와 Negative로 예측했지만 실제론 Positive인 False negative error를 통해 Precision과 Recall을 정의가능하다. 만약 FN 값을 5로 늘린다면 Acuuracy는 조금 줄어들지만 이에 반해 Precision과 Recall은 상대적으로 크게 줄어든다. 따라서 Unbalanced data set에서는 모델의 성능을 평가할 때 Acuuracy뿐 아니라 Precision과 Recall값도 보아야 한다.

**3.ROC Curve** : 서로 다른 Classifier에 대해서 여러개의 ROC Curve를 그렸을 때 왼쪽 상단으로 갈수록 동일 Sensitivity에 대해 더 낮은 FPR을 제공하기 때문에 성능이 더 좋다.  

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/85829015-86a8-4d5a-9ed3-ec71779a322e"></p>





