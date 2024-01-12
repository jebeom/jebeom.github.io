---
title : "[LG Aimers] 딥러닝(Deep Learning) [2]"
excerpt: "이미지와 언어모델 학습을 위한 딥러닝 모델과 학습원리에 대해 알아보자"

category :
    - LG_Aimers
tag :
    - deep_learning

toc : true
toc_sticky: true
comments: true

---

지난 포스팅에 이어 이미지와 언어모델 학습을 위한 딥러닝 모델과 학습원리에 대해 알아보자

> 본 포스팅은 LG Aimers 수업 내용을 정리한 글로 모든 내용의 출처는 [LG Aimers](https://www.lgaimers.ai)에 있습니다


## 합성곱 신경망(CNN)

Convolutional neural network 혹은 줄여서 Convnet 혹은 CNN은 DNN을 응용한 알고리즘으로써 computer vision 혹은 영상처리에 많이 쓰이는 구조이다.

### CNN 동작원리

어떤 이미지가 있을 때 그게 X를 의미하는지 O를 의미하는지 분류하는 task를 수행한다고 생각해보자. 이때 X나 O의 크기가 작거나, 각도가 틀어져 있거나, 아니면 굵은 형태로 나타내지는 등의 변화 요소에도 일관되게 X나 O라고 분류할 수 있어야 하며 딥러닝이나 머신러닝 알고리즘의 입력으로 줄 때 이미지는 하나의 숫자들로 이루어진 2차원 배열로 나타내지는데 하얀색 픽셀을 1, 검은색 픽셀을 -1로 정의한다.

CNN은 특정 class에 존재할 수 있는 작은 특정 패턴들을 정의하고, 이러한 패턴들이 주어진 이미지 상에 있는지 판단한다. 예를 들어 9x9 패치가 있다고 했을 때 어떠한 패턴을 가지는 작은 이미지 3x3패치를 9x9패치에 오버랩 시켰을 때 3x3패치에 정의된 pixel의 값과 9x9패치에서 해당하는 위치의 값을 곱한다. 만약 두 값이 같다면 1이라는 값을, 다르다면 -1이라는 값을 출력하게 된다. 결과적으로 3x3패치를 오버랩시켰을 때 매칭이 되었는가 안 되었는가를 이런 두 값들 간의 곱셈으로 나타낸 후에, 이 값을 다 합하고 총 pixel의 개수(이 경우, 9)로 나누면 총 매칭 된 정도의 확률이 나오게 된다. 이러한 확률은 가운데 픽셀의 위치에 기록하고 모든 위치들에 대해서 위의 작업을 반복하면 아래와 같이 매칭의 정도를 나타내는 그림을 얻을 수 있게 된다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/eabaacb8-64ac-4551-92ae-f2649011f1b3"></p>

위와 같은 그림에서 매칭의 정도를 나타내는 그림인 오른쪽 이미지를 **활성화 지도(activation map)**이라 부른다.

또한 convolution layer에는 여러 개의 특정한 패턴을 가진 3x3패치(convolution filter)가 존재하는데 이러한 패치를 각각 주어진 입력 이미지에 적용했을 때 패치(filter) 별로 활성화 지도가 나오게 되고, 이것을 하나의 이미지라고 본다면 filter 개수만큼의 이미지가 나오게 된다.

그런데 입력 이미지가 1장이 아니라 3장의 이미지를 가질 수 있다. 이 때 입력 이미지를 채널이라 부르는데 3개의 채널로 이루어진 입력 이미지가 있을 때, 각각의 채널마다 3개의 convolution 필터를 오버랩할 수 있고, 아까처럼 해당하는 값들끼리 곱하고 더하는 과정을 3개의 채널에 대해서 모든 위치마다 실시한 후에 모두 합치면 최종적으로 각 위치별로 scalar 값(매칭된 확률)을 가진 3개의 activation map을 얻을 수 있다. 따라서 activation map의 개수는 filter의 개수와 동일하다. 이 때 한 filter는 아>래 그림처럼 3개의 특정 패턴을 가진 패치로 구성되어 있다.


<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/3931eca6-ff5a-41f6-bc93-950a931a96c7"></p>

### Pooling Layer

Pooling layer는 CNN을 구성하는 중요한 두 번째 타입의 layer로 기본적인 동작 과정은 다음과 같다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/a568b807-d16a-4567-b2d9-96168bc56b24"></p>

위의 그림처럼 활성화 지도가 있고, 특정한 사이즈 가령 2x2패치를 생각해보자. 이 패치를 주어진 이미지에 왼쪽 이미지에서부터 2칸씩 옮겨가면서 2x2 패치를 오버랩하게 되고, 이 오버랩된 2x2이미지 내에서 가장 최대 값을 골라내는 과정을 수행하며 output 이미지를 얻어낸다. 이러한 과정을 통해 가로 세로 사이즈를 반으로 줄이되, 특정 위치에서 검출하고자 하는 패턴을 강한 세기로 반영했기에 size가 절반이 된 이미지를 만들 수 있다. 다시 말해서 찾고자 하는 패턴이 이만큼의 세기로 나타났다는 요약 과정을 pooling 과정을 통해 수행한다.

이러한 pooling 과정은 활성화 지도를 만든 다음에 수행 하게 되며, 입력으로 들어오는 채널의 개수, 즉 활성화 지도의 개수만큼 절반으로 사이즈가 감소된 활성화 지도를 얻을 수 있다.


### ReLU Layer

선형 결합을 한 후 activation function을 적용해줄 때 DNN의 학습을 용이하게 해주는 ReLU라는 activation function을 적용해준다. 이전 convolution layer에서 나왔던 활성화 지도를 입력으로 받아서 각각의 값에다가 ReLU함수를 적용하면 양수인 값은 그대로, 음수인 값은 0으로 clipping 해주는 변형된 활성화지도를 얻을 수 있다. 다수의 채널로 이루어진 활성화 지도 각각에 대해 ReLU function을 적용하기에 output으로 출력되는 활성화지도의 수는 입력 채널의 개수와 동일하다.

### 정리

최종적으로 CNN에서는 처음에 설명했던 선형 결합의 가중치를 적용(패치를 오버랩해서 곱하고)해서 가중합(곱한 값들을 더해줌)을 구하는 Convolutional operation을 Convolution layer에서 먼저 수행하고 ReLU layer를 통해 몇 번 stacking을 한 후, 이미지를 축약된 형태로 바꾸어주는 Max Pooling layer를 적용한다.

최종적으로 Convolution, ReLU, Pooling의 패턴을 반복하는 형태로 딥러닝의 layer들을 쌓게 된다.

이렇게 반복적인 패턴으로 layer들을 쌓다가 특정 시점부터는 아래 그림처럼 2x2의 사이즈이고, 채널 수가 3인 activation map을 한 줄에 쭉 vector로 피게 된다. 이러한 vector를 입력으로 주어서 fully-connected layer를 구성하게 된다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/2826ea36-b4bb-46ec-9536-f1cc01ef4807"></p>

이러한 fully-connected layer통해 2개의 output 노드를 구성하고 학습 과정을 통해 fully-connected layer의 가중치도 특정한 값들로 최적의 값으로 도출되고 x라는 class에 대응하는 score는 0.92, o라는 class 대응하는 score는 0.51 이 나와 최종적으로 x로 분류를 할 것이다.

그 다음에 Binary cross-entropy loss나 MSE loss를 통해 loss값을 계산하고 gradient descent를 수행하기 위해 Backpropagation을 수행하면 Parameter들이 gradient 계산을 통해 학습이 진행된다. ReLU, Pooling layer에는 trainable parameter가 없고, Convolution layer에서는 convolution filter의 가중치 값이라는 trainable parameter가 존재해 이 값들에 대한 gradient를 계산하고 gradient descent를 통해 filter의 coefficient를 최적의 값으로 도출해낸다.

## CNN의 다양한 Architecture

다양한 Computer vision task에 대해 잘 동작하는 특정한 Architecture, 즉 layer의 순서나 위치, 그리고 각 layer 별로 convolution filter의 filter 사이즈나 filter의 개수들을 잘 정의해둔 Architecture들이 존재한다.

이 중에서도 가장 좋은 성능을 내는 Architecture에는 다음과 같은 것들이 존재한다.

- AlexNet
- VGGNet
- GoogLeNet
- ResNet

### VGGNet

VGGNeT은 convolution layer에서 사용하는 convolution filter의 가로 세로 사이즈를 무조건 3x3으로 고정하지만 layer를 깊이 쌓아 좋은 성능을 낸다.

### ResNet(Residal Network)

ResNet(Residal Network)은 convolution layer를 통해 나온 output에 layer를 또 convolution layer를 추가하는 경우 수많은 layer에 있는 parameter이 순기능을 내기까지 시간이 많이 걸릴 수 있는데 ResNet은 layer를 필요할 때 건너뛸 수 있도록 하는 skip connection을 통해 layer를 여러 개 쌓았을 때도 학습이 잘 안 된다거나, 수많은 parameter들이 헤매서 학습이 잘 안되어 성능이 안 좋아지는 단점들을 해결할 수 있다.

결과적으로 ResNet은 152개의 layer까지 layer를 많이 쌓아도 성능을 꾸준히 향상시킬 수 있다.

## 순환 신경망(RNN)

순환 신경망(Recurrent Neural Networks), 줄여서 RNN은 CNN과 마찬가지로 DNN을 응용한 알고리즘으로써 sequence data나 Time-series 혹은 시계열 data에 대해서 적절하게 적용가능한 구조로 동일한 function을 반복적으로 호출하는 특징을 가지고 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/72eaa86b-d1a7-4d84-9d7c-c77dcef0a37d" ></p>

위 그림과 같이 RNN의 기본동작 원리는 현재 time step에서의 입력 신호 $x_{t}$와 그 이전 time step에서 동일한 RNN function이 계산했던 Hidden state vector인 $h_{t-1}$을 입력으로 받아서 현재 time step에서 RNN module의 output인 currrent hidden state vector($h_{t}$를 만들어주게 된다. 여기서 매 time step마다 동일한 parameter set을 가지는 layer인 $f_{w}$가 반복적으로 수행되며, prediction을 해야 하는 특정 time step에서 $h_{t}$를 다시 입력으로 output layer에 전달해 줌으로써 최종 예측 결과를 만들어주게 된다.

### RNN을 사용한 자연어 처리 원리

RNN을 사용한 자연어 처리에서 language model이라는 기본적인 task에 대해서 살펴보자. 아래 그림과 같이 h,e,l,l,o라는 다섯 개의 character 로 이루어진 training data가 주어져 있을 때 입력을 매 time step 마다 받았을 때, 무엇일지 예측하는 다시 말해서 delay를 허용하지 않는 many-to-many 형태의 task를 생각해보자.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/7f12cfd3-e9db-48e0-ac26-5553073cdc1d" ></p>

먼저 각각의 character들으 하나의 vocabulary로 구성하고 각 vocabulary 상에서 각각의 워드를 categorical variable 형태로 encoding 하면 h는 vocabulary 상의 첫 번째 dimension으로서 1,0,0,0 형태로처럼 각각의 워드는 위의 그림의 input layer처럼 one-hot vector로 나타낼 수 있다.

그 다음 RNN module에서 기본적으로 one-hot vector 형태의 입력 vector를 현재 time step의 $x_{t}$ 입력 벡터로 받게 되고, 이전 time step에서 넘어오는 $h_{t-1}$이라는 추가적인 입력 vector(일반적으로 t=1일 때 $h_{0}$에는 0 vector를 입력으로 줌)를 받아서 현재 Hidden state vector($h_{t}$)를 계산하게 된다. 위의 그림에서 볼 수 있듯이 해당 예시에서 $h_{t}$의 dimension은 3으로 설정된 것을 알 수 있고, $x_{t}$는 4차원 vector, $W_{xh}$는 4x3 matrix 형태, $h_{t-1}$은 3차원 vector, 그리고 $W_{hh}$는 3x3 matrix 형태로 만들어져서 최종적으로 3차원 output vector가 나오고 이 vector가 tanh 함수를 통과해 현재 time step의 output vector를 얻는다.

다음으로 각 time step을 기준으로 다음에 나타나야 할 character를 예측을 해야하기에, 매 time step마다 4개의 class중에 1개의 class를 예측하는 multi-class classification을 수행해야한다.

따라서 각 time step의 Hidden state vector($h_{t}$)를 $W_{hy}$라는 linear transformation을 적용해서 그림의 Output layer처럼 classification을 수행할 4개의 class에 대응하는 4차원 logit vector들로 만들고, 이 4차원 logit vector를 softmax layer를 통과시킨 후 h,e,l,o 4개의 class에 대응하는, 다시 말해서 예측하고자 하는 대상인 4개의 character중에 하나로 classification해주는 probability vecor를 얻게 되며 h다음은 e, e다음은 l이 나와야한다는 ground truth class정보를 알고 있기에 정답에 해당하는 확률값에 log값을 취하고 마이너스를 취해서 softmax loss 값을 최소화하는 방식으로 학습을 진행한다.

결과적으로 최종 예측 값은 가장 큰 확률을 부여하게 된 해당 class 혹은 character를 그 다음에 나타날 character로 예측하게 된다.


### Test time에서 사용법

학습이 완료된 model을 test time에서 사용할 때는 첫 번째 character에서 특정한 character를 입력으로 주게 되면 학습이 완료된 model에서 나온 예측된 softmax 확률 vector가 결과로 나올 것이고, 거기서 가장 큰 확률을 부여받은 character를 다음에 나타날 character로 예측하게 되고, 이렇게 예측된 character를 그 다음 time step의 RNN에 입력으로 제공해주어 연쇄적으로 최초의 time step에서 character 하나만 입력으로 주면 예측된 sequence를 무한정 만들 수 있게 된다.

이렇게 model의 output이 그 다음 time step의 입력으로 주어지는 방식의 예측 model을 **auto regressive model**이라고 부르며 이러한 model에 입력으로 어떤 등장인물이 특정한 이야기를 하는 문단을 주고 학습을 하게 되면, 첫 번째 character만 입력으로 주면 model은 아래 그림처럼 주어진 문단에 있었던 다양한 패턴으로서 등장인물의 이름을 먼저 생성하고 character sequence로써 문장을 생성해준다. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/46fab6ed-c160-4452-ab37-a597bf4caa12"></p>  

또한, 이러한 모델은 논문이나 수학 수식뿐 아니라 C code와 같은 programming code도 만들 수 있다.

### 실제로 사용하는 RNN 모델

하지만 학습 과정 중에 gradient가 vanishing 되거나 gradient가 기하급수적으로 늘어나는 부작용 등으로 인해 위와 같이 선형 변화 및 tanh를 적용해서 Hidden state를 만드는 original 구조의 RNN 다시 말해서 vanilla RNN 구조를 사용하지 않는다.

따라서 RNN의 기본적인 입출력 setting은 동일하게 유지하되, gradient vanishing이나 exploding문제를 효과적으로 해결한 LSTM(long short term memory)나 GRU(gated recurrent unit)과 같은 특정한 구조의 RNN 모델을 많이 사용한다.

## Seq2Seq Model

Seq2Seq Model은 RNN 기반의 model을 사용해서 sequence를 입력 및 출력으로 처리하며 대표적인 사례로는 챗봇이 있다.


<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/3440ed88-6eb1-4b9c-99fd-e76074aca972"></p>

위의 그림처럼 어떤 고객이 Are you free tommorrow 와 같은 질문을 했을 때 이러한 입력 문장은 단어별로 time step에 해당 vector가 입력으로 주어져 RNN module이 매 time step마다 정보를 잘 처리하고 축적하는 과정을 거치게 되고, 마지막 time step에서 나온 Hidden state vector $h_{t}$는 입력 sequence에 있는 모든 단어들의 정보들을 잘 축적한 역할을 한다.

여기까지의 과정이 입력 신호에서 필요한 정보를 추출한 **encoder**에 해당하는 과정이라고 볼 수 있다.

또한, 챗봇은 질문에 대한 답을 생성해준다. 여기서 답은 질문과 마찬가지로 특정 단어들의 sequence를 output으로 내어주기 위해, 예측 단계에 넘어와서 가장 최초로 등장하는 단어로서 특수문자로서 start of sentence라는 단어를 lauguage modeling에서의 autoregressive setting에서처럼 다음 time step에 입력으로 줘서 그 다음에 나타날 단어를 연쇄적으로 예측한다. 

최종적으로 단어 생성이 끝이 났다는 것을 의미하는 end of sentence token이 예측될 때까지 이러한 예측 과정을 수행하는 단계를 **decoder**라고 부른다.

이러한 encoder와 decoder에 쓰이는 RNN은 서로 parameter가 공육되지 않는 두 개의 별개의 RNN을 사용하는 경우가 일반적이다.

### Atention

Original Seq2Seq model에서는 sequence가 길어질 때 정보를 유실하는 등의 경우들이 많아서 output sequence가 제대로 생성되지 않는 bottleneck 문제들이 발생해서 이를 해결하고자 attention이라는 추가적인 module이 기존 Seq2Seq model에 도입이 되었다.

예를 들어 입력 sequence로서 한자로 된 문장이 주어져 이를 영어로 번역하는 task에서 입력 sequence에 주어지는 각각의 한자 워드를 encoder에서 encoding을 한 후에 decoder에서는 encoder의 마지막 time step의 Hidden-state vector만을 입력으로 받는 것이 아니라 입력과 더불어 decoder의 각 time step에서 encoder에서 나온 여러 단어별로 encoding 된 Hidden-state vector 중에서 유사도를 이용해 우리가 필요로 하는 vector들을 활용할 수 있는 것이 attention model의 아이디어이다.
 
Seq2Seq model에 Atention module을 추가함으로써 정보의 병목현상을 효과적으로 해결하고, 기계 번역 task의 성능을 많이 끌어올리게 되었다.


이번 포스팅을 통해 CNN은 비전과 같은 영상처리에서 사용하고 RNN은 기계번역과 같은 자연어 처리에 사용한다는 것을 알게 되었고 두 알고리즘 모두 Classification Task에서 사용한다는 것을 알게 되었다.



