---
title : "[LG Aimers] 딥러닝(Deep Learning) [1]"
excerpt: "이미지와 언어모델 학습을 위한 딥러닝 모델과 학습원리에 대해 알아보자"

category :
    - LG_Aimers
tag :
    - deep_learning

toc : true
toc_sticky: true
comments: true

---

이미지와 언어모델 학습을 위한 딥러닝 모델과 학습원리에 대해 알아보자

> 본 포스팅은 LG Aimers 수업 내용을 정리한 글로 모든 내용의 출처는 [LG Aimers](https://www.lgaimers.ai)에 있습니다

## 심층신경망(Deep Neural Networks)

심층신경망은 두뇌의 동작 과정을 모방해서 수학적인 인공지능 알고리즘으로 만든 것으로 신경세포 하나는 다른 신경 세포들과 연결되어 있어 다른 신경세포들로부터 전기 신호를 입력받는데 입력받은 전기 신호에 특정한 값을 곱해서 더해서 전기 신호를 만들어내고 이 전기 신호를 다른 신경 세포들에게 전달해주는 과정을 거친다.

이러한 뉴런의 동작 과정을 수학적으로 본따 만든 알고리즘을 **Perceptron**이라 하며, Perceptron은 입력 정보($x_{1},x_{2}$)를 가중치($w$)와 곱하고 특정한 상수 값을 더한 가중합을 만들어내고 Sigmoid, tanh, ReLu 등의 활성화 함수를 거쳐서 최종 출력 신호를 만들어주게 되며 다음 계층(layer)의 뉴런들에게 해당 출력을 전달해준다.

가중합의 값이 0보다 큰 경우 다시 말해서 Decision Boundary 함수보다 위에 위치하는 경우, 활성함수의 Rule에 따라 뉴런의 Output이 1이 되고, 0보다 작은 경우에는 뉴런의 Output이 0이 된다.   

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/a3658c2f-1ff9-41ab-a75d-5abf9f306ee8"></p>  

위와 같은 그림에서 하나의 layer(계층)만으로는 표현할 수 없었던 XOR gate를 하나의 layer를 추가하고 해당 layer의 뉴런에 AND gate와 OR gate를 구성하도록 가중치를 설정함으로써 XOR gate를 표현가능하다. 이렇게 Input layer(입력 계층)와 Output layer(출력 계층) 사이에 있는 layer를 Hidden layer라고 부른다.
 
### Forward Propagation

아래 그림처럼 특정 layer에서 특정 뉴런의 출력 값을 activation이라는 의미로 a라는 notation을 사용하고 아래 첨자는 layer, 그리고 위 첨자는 layer 내에 몇 번째 노드인지를 의미하는 index로 사용하며 하나의 계층 내에 여러 개의 입출력 노드가 존재하는데 이 노드(뉴런)들의 가중치들을 다 모아 행렬로 정의한 가중치 행렬 W를 정의할 수 있으며 위 첨자는 layer를 의미한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/3068b6ff-28ba-489f-a305-04a54654095e"></p>

특정 뉴런에 있는 output 값을 계산할 때 뉴런의 입력으로 주어지는 vector를 column vector로 만들고, 뉴런이 가지는 가중치를 row vector로 만들면 아래 그림과 같이 행렬의 내적 형태로 가중합을 쉽게 나타낼 수 있으며 이러한 가중합으로 나타내어지는 scalar값이 활성 함수(sigmoid 등)를 통과해 최종적인 output 값을 나타내며 다음 계층에서는 이 output값이 input값으로 전달되어 같은 과정을 반복한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/a43c1339-94ae-4c12-beab-b653492e0cd3"></p>

### Softmax Classifier

예를 들어 어떤 이미지가 1~10 까지의 값을 가질 수 있다고 가정했을 때, 이를 분류해주는 모델을 만들 때, 1일 확률이 5%, 2일 확률이 75% ... 등으로 10개에 걸쳐서 나타나는 확률 값의 총합이 1이 되도록 하는 확률 분포에 해당하는 vector를 얻어야 한다.

다시 말해서 이러한 multi-class classification을 목적으로 하여 output vector의 형태가 다 합쳤을 때 1이 나오는 확률 분포의 형태로 output을 내어줄 수 있도록 하는 특정한 활성 함수를 Softmax layer 혹은 Softmax classifier 라고 한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/5ddea758-e910-4b83-923a-9bdfbd77b8fa"></p>

위의 그림과 같이 주어진 이미지가 dot, cat, frog 세 개 중에 하나로 분류하고자 하는 task에서 해당 layer의 입력 vector가 주어져 있고, 상수에 해당하는 붙박이로 1이라는 값이 부여된 입력 노트 x1이 주어져 있을 때, 출력 노드는 3개이고, 입력 노드는 5개로 이루어진 3 by 5 짜리의 가중치 행렬을 가지는 형태에서 vector간의 내적을 통해 각 출력 노드의 값들이 계산이 되고, 각각의 값들을 softmax layer라는 활성 함수에 통과 시킨다. 이 활성 함수에는 다양한 값의 출력 노드의 값들이 입력되는데 합이 1인 형태의 확률 분포로 만들어주기 위해 지수함수로 통과 시켜주어 양수인 값들로 변환을 시켜주고, 전체 합으로 나누어 상대적인 비율을 계산해준다. 따라서 위 그림에서 Dog가 될 확률은 36%, Cat은 62%, Frog는 2%의 확률이다 라고 해석한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/0c08e18d-bda6-456e-a08c-0c3376117499"></p>

또한 위 그림과 같이 softmax layer에는 MAE(최소제곱법) loss 대신 softmax loss, 즉, Cross-Entropy Loss를 사용하게 되며 정답 Class에 해당하는 확률 값이 최대한 1이 나오도록 Loss function을 위 그림에서의 수식과 같이 사용해주게 된다.

여기서 $\hat{p}_{c}$는 우리의 예측된 확률 vector이며, $y_{c}$는 Ground truth vector로 해당하는 class 일 때는 1, 아니면 0 소위 말하는 one-hot vector형태로 주어지며 index i는 i번째 training data 아이템이다. (예를 들어 i가 37이면 Cat에 속함)

최종적으로 $ loss(L) = -log(\hat{p}_{y_{i}}})$ 그래프가 오른쪽 아래처럼 그려지기 때문에 정답 class에 부여된 확률 값이 작아질수록 Loss가 매우 커지고, 확률 값이 1에 가까워질수록 loss값이 0에 가까워진다.

Binary classification 즉, class가 2개만 있을 때 많이 사용하는 logistic regression도 아래 그림과 같이 softmax classifire의 Special case로 이해가능하다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/95e49ed8-f89c-442d-8c0b-f4a7cd212cf0"></p>

위 문제는 class가 1개만 있고, 맞다 아니다에 해당하는 경우이고, class가 2개 인 경우 W에 한 줄의 가중치가 더 부여가 되어 output 노드를 2개로 만들고, 지수함수를 통과시켜 상대적인 비율을 계산한다. class의 개수가 2개인 BCE loss(L) 수식도 위의 그림에서 확인할 수 있다. 여기서 $y_{i}$는 positive class 일 때는 1이고, negative class 일 때는 0으로 정의된다.

## 심층신경망의 학습 과정

심층신경망의 학습 과정은 학습 data를 parameter들로 이루어진 neural network에 입력으로 집어넣고 gruond truth값(실제값)과 비교함으로써 차이를 최소화하는 Loss function을 만들고 이를 최적화시키는 parameter들을 찾는 과정이다.

이러한 parameter들을 찾는 과정은 현재 주어진 Parameter값을 각각의 Parameter들의 미분 방향의 마이너스 방향으로 그리고 step size 혹은 learning rate을 곱해서 해당 Parameter값을 업데이트하면서 진행된다.

### Back Propagation

Loss를 최소화하기 위해 gradient descent 과정을 수행하는데 이를 위해 편미분 값이 필요하다. 이러한 편미분 값을 구하기 위해 Back Propagation을 진행한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/df43ad2c-ea0f-43a9-9d84-4c1129c4d9c7"></p>

위 그림과 같이 초기엔 output의 편미분 값을 1로 두고 함수에 대한 Gradient 값을 계산하는데 왼쪽을 입력 x, 오른쪽을 출력 y라고 볼 때 y를 x값에 대해 미분 했을 때 Lacal Gradient 함수가 나오고 여기에 Foward Propagation 당시에 발생된 함수의 입력값 1.37을 대입하면 -0.53이라는 값이 나오는데 이 값에 y, 즉 output에 해당하는 편미분 값에 같이 곱한다. 여기서 초기엔 편미분 값을 1로 두었으니 결과적으로 gradient 값을 -0.53이 된다. 이와 같은 과정을 Computational graph 즉, 위 그림에서의 역과정으로 진행한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/b185071e-37a9-429e-a386-74ba64960e90"></p>

예를 들어 위 그림과 같은 exponential 함수에서 당시 입력 값이 -1이고 그걸 Lacal Gradient 함수에 넣었을 때 나오는 $e^{-1}$ 값을 해당 함수의 output 값의 gradient 값인 -0.53 곱해주어 입력값에 대한 gradient -0.2를 얻을 수 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/4d0f7aff-9d52-4a28-b5e3-955391c1c665"></p>

덧셈 노드에서는 위 그림처럼 x, y 각각에 대해 미분해주고 그 값에 output gradient 값인 0.2를 곱해서 계산해주고

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/5d4dbf02-d6fd-48c0-bbcb-a68e6d605b20"></p>

곱셈 노드에서는 위 그림처럼 x, y 각각에 대해 미분해주면 y, x가 나오므로 x의 input gradient 는 output gradient인 0.2에 y의 입력값 -1을 곱해 -0.2 가 나오고 y의 input gradient 는 output gradient인 0.2에 x의 입력값 2를 곱해 0.39가 나오는 것을 알 수 있다. (output gradient가 실제론 0.195인데 반올림해서 0.2로 표시했기 때문) 

따라서 gradient descent를 통해 업데이트 시켜야 하는 parameter $w_{0}$ 는 현재 값에서 gradient값과 learning rate를 곱한 값을 빼서 업데이트 된다. 즉, Learning rate를 0.1이라 가정하면 업데이트 된 값은 $ 2 - 0.1 \times (-0.2) = 2.02$ 가 된다. 다른 parameter들도 이와 같은 gradient descent 방법을 통해 업데이트 한다.

sigmoid 에서 gradient값을 위의 예제처럼 분할해서 계산하지 않고 한번에 유도했을 때 $\sigma(x) \times (1-\sigma(x))$ 로 표현되는데 이러한 2차식의 범위는 0부터 1/4 이기 때문에 sigmoid 함수를 back propagtion할 때 마다 output값에서 계산되었던 gradient 값에 0 ~ 1/4 사이의 값을 곱해 줌으로써 입력 값의 gradient가 결정되어 back propagation을 수행할 때마다 gradient 값이 점차 작아져 0에 가까워지는 문제점을 야기해 특정 learning rate를 사용했을 때 앞쪽에 있는 parameter들의 gradient 값이 작음으로 인해 parameter들의 업데이트가 거의 일어나지 않는, 다시 말해서 nueral network의 학습이 느려지는 gradient vanishing 문제가 발생한다.  

## 다양한 활성 함수

이러한 gradient vanishing 문제를 해결하기 위해 다양한 형태의 활성 함수들이 제안되었다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/8101a6d9-a057-4729-8822-40122682ceff"></p>  



### Tanh Activation

tanh는 sigmoid 함수에 2배를 곱하고 -1을 해준 함수로 입력값을 -1부터 1로 Mapping 시켜준다.

하지만 tanh함수도 gradient 값이 0부터 1/2 사이의 값을 곱해주므로 여전히 gradient vanishing 문제가 발생하게된다.

### ReLU(Rectified Linear Unit)

따라서 실제 layer를 굉장히 많이 쌓았을 때 gradient vanishing 문제를 해결할 수 있는 활성 함수로 ReLU 함수를 사용하게 된다. 이 함수의 경우 0보다 작은 경우 0으로 0보다 큰 경우는 입력값을 그대로 내보내 준다. 따라서 입력 값이 커질수록 함수가 거의 상수값 1이라는 값으로 수렴함으로써 이 영역에서는 접선의 기울기 gradient 값이 0에 가까워져 gradient vanishing 문제를 해결 가능하고, 계산이 빠르다.

하지만 함수에 주어진 입력 값이 0보다 작은 다시 말해서 선형 결합의 가중치가 음수인 경우, gradient값이 0이 되어 output node에 무슨 값이 주어졌든 간에 gradient 값에 0을 곱해주어 이후 gradient 값은 모두 다 0이 되는 단점이 있다.


## Batch Normalization

그래서 우리는 forward propagation 할 당시 다양한 활성 함수들의 입력으로 주어지는 값들의 대략적인 범위를 0근처로 제한할 수 있다면 gradient vanishing 문제를 해결할 수 있다.
 
이러한 원리를 바탕으로 gradient vanishing 문제를 해결하기 위해 특정 활성 함수를 썼을 때 학습을 용이하게 하는 추가적인 특별한 형태의 뉴런 혹은 layer를 생각해볼 수 있는데 그것이 바로 Batch Normalization라는 layer이다. 따라서 선형 결합을 수행한 이후에 활성 함수로 들어가기 전에 Batch Normalization layer를 추가하여 gradient vanishing 문제를 해결한다.

Batch Normalization의 기본적인 동작 과정은 다음과 같다.

하나의 mini batch가 주어졌을 때 이 mini batch의 data의 개수 즉, size가 10이라라고 하면 특정 노드에서 tanh노드를 통과하기 직전에 입력으로 발생되는 값이 10개의 mini batch내에 data item에 하나씩 존재할텐데 우선 첫 번째 단계로 이 값들을 다 모아서 평균과 분산을 계산하고 평균이 0이 되도록, 그리고 분산이 1이 되도록 하는 **정규화(Normalization)**과정을 진행한다.

이러한 정규화 과정을 수행하면 tanh의 입력으로 주어지는 값의 대략적인 범위를 0을 중심으로 하는 분포로 형성가능하다. 여기서 분산을 컨트롤하는 이유는 값들의 평균이 0이라 하더라도 분산이 작으면, 다시 말해서 값들의 변화 폭이 너무 작으면 tanh의 output값 또한 변화 자체가 작아서 차이를 구분하기 어렵고, 분산이 너무 크면, 평균이 0이라도 특정값이 너무 크거나 작을 수 있기에 적절한 범위를 넘어서기 때문이다.

하지만 이러한 정규화 과정, 다시 말해서 평균을 0, 분산을 1로 만드는 과정은 neural network이 추출한 중요한 정보를 잃어버리게 만들 수 있다. 따라서 잃어버려진 정보를 스스로 원하는 정보로 복원할 수 있게끔 평균과 분산을 각각 0과 1로 만든 값에다가 gradient descent를 통한 학습에 의해서 최적화하게 하는 parameter들을 도입해서 y=ax+b 꼴로 변환을 수행하는 **scale and shift**과정을 거치며 b는 x값이 가지던 10개의 값의 평균값이 되고, a는 분산의 제곱근값이 된다. 이러한 두 번째 변환을 통해 Deep Learning이 생각하는 최적의 평균 분산 값을 스스로 결정할 수 있게, 다시 말해서 Loss function을 최적화하는 과정에서 도출 되는 최적의 parameter값으로 결정되게 할 수 있게한다. 이러한 추가적인 layer를 통해 data가 가지는 고유의 평균과 분산을 복원해낼 수 있고, Batch Normalization의 두 번째 단계로 삽입한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/92e2dd49-1daf-4a1f-893a-8eb794a793ad"></p>

Batch Normalization의 과정을 요약하면 위의 그림과 같다.




















