---
title : "[LG Aimers] 딥러닝(Deep Learning) [3]"
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


## Transformer Model

Transformer Model은 이전 포스팅에서 다룬 RNN을 사용한 Seq2Swq with attention Model에서 여러 time step에 걸쳐서 gradient 정보가 전달되며 정보가 변질되는 문제점(long-term dependency issue)을 개선한 모델이라 생각해볼 수 있다.

기존의 Seq2Swq with attention Model은 크게 encoder와 decoder, 그리고 decoder의 각 time step에서부터 encoder의 Hidden state vector들 중 원하는 정보를 그때그때 가져갈 수 있도록 하는 attention module로 구성되는데 기존 model에서는 RNN기반의 model로 encoder와 decoder가 구성되었는데 이에 반해 Transformal Model구조에서는 추가적인 모듈로만 사용되었던 Attention module이 encoder,docoder에 사용되는 sequence 자체를 잘 encoding하는 역할을 수행할 수 있다.

다시 말해서 Transformal Model은 RNN이나 CNN없이 오로지 Attention module만으로 전체 sequence를 입력 및 출력으로 처리할 수 있다.

## Transformer Model Encoder

정보를 찾고자 하는 주체 혹은 query vector를 정보를 꺼내가려는 재료 vector들과 같은 주체로 생각하면 주어진 seqeunce를 attention model을 통해 encoding 할 수 있다. 이렇듯 정보를 찾고자 하는 주체와 정보를 꺼내가려는 소스 정보들이 동일시된다 해서 seqeunce data를 encoding 할 때 쓰이는 attention module을 **self-attention**이라고 부른다.


<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/3d714f9a-8de8-4e23-90e5-ff15756efab6" ></p>  

위의 그림과 같이 self-attention은 I go home이라는 세 개 단어로 이루어진 seqeunce를 encoding 할 때 먼저 I라는 단어의 vector를 decoder의 특정 time step에서의 Hidden state vector, 다시 말해 query vector로 사용하고, 3개의 vector $x_{1},x_{2},x_{3}$ 들이 정보를 끌어오려고 하는 소스 정보들이라고 생각을 하면, vector들 간에 내적을 통해 유사도를 구하고 거기서 나온 합이 1인 형태의 가중치 vector를 소스 혹은 재료 vector에 해당하는 각각의 vector들의 가중치로 적용해서 소스 vector들의 가중합을 구할 수 있다.

여기서 소스로 사용된 각 단어별 vector들의 가중합이 곧 query로 사용되었던 'I'라는 단어에 대응하는, sequence 전체적인 정보를 잘 버무려 만든 'I'에 대해 encoding 된 vector라고 생각할 수 있다.

비슷하게 두 번째 단어인 'go'를 sequence 전체 상에서 encoding 할 때에는 이 단어를 decoder 상에 두 번째 Hidden state vector 인 것처럼 사용해서 $x_{1}$과 내적하고 자기 자신인 $x_{2}$와 내적하고 $x_{3}$와 내적하여 나온 유사도를 softmax를 취해서 초록색 재료 vector에 가중치로 적용해서 가중합의 vector를 go라는 단어에 대한 sequence상에 encoding 된 vector로서 활용이 가능하다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/b0a4cbe9-41a8-4ecc-ae21-0360ddb46c83"></p>  

정리하자면 하나의 vector set $x_{1},x_{2},x_{3}$에서 한 vector가 query vector 혹은 decoder hidden state vector 즉 $q_{1}$처럼 사용이 되기도 하고, 또 다른 재료 vector로서 query vector와 유사도를 구하게 되는 key vector $k_{1},k_{2},k_{3}$로 사용되기도 하고, query vector와 내적을 통해 유사도를 계산하고 softmax를 통해 나온 가중치를 실제로 적용하게 하는 재료 vector들인 value vector $v_{1},v_{2},v_{3}$ 로 3가지의 다른 용도로서 같은 vector set이 사용된다.

즉, self-attention module에서는 vector들이 세 가지 각각의 용도로 사용될 때 각기 나름의 linear transformation layer가 존재한다고 설계를 했으며 각각의 서로 다른 용도의 vector는 위의 그림과 같이 $W^{q},W^{k},W^{v}$를 곱해주는 선형 변환을 통해 만들어진다. 

또한 각 단어별로 attention module 혹은 self-attention module을 통해서 각 단어를 sequence 상에서 encoding 한 Hidden state vector를 얻는 수식을 위의 그림과 같이 표현 가능하다. 여기서 $d_{k}$는 key vector의 dimension을 의미하는데 만약 $d_{k}$값이 커지면 분산이 커져 softmax를 통해 나오는 전체 확률 분포가 하나의 특정값으로 몰려 학습의 측면에서 gradient가 잘 흐르지 않을 수 있기에 아래 그림과 같은 Scaled Dot-produnct attention을 통해 일정한 분산을 가지게 해준다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/5edbb097-1e30-4745-bdc6-67329844785f"></p>

### Multi-head Attention

주어진 sequence의 각각의 입력 vector들을 encoding할 때 $W^{q},W^{k},W^{v}$를 곱해주면 하나의 정해진 변환으로 attention module을 수행해서 seqeunce를 encoding하게 되는데 **Multi-head Attention**은 세 개의 선형 변환 matrix를 여러 개의 set로 두어서 각각의 set을 통해 변환된 query,key,value vector들을 사용해 attention model에서 seqeunce의 각 단어들을 encoding하고, 각각의 set에 선형 변환을 해서 나온 attention model의 output들을 나중에 concat하는 형태로 정보를 합친다.

각기 다른 기준들로 단어들(예를 들면 thinking, machine)을 encoding 해서 만들어진 결과 vector들을 dimenstion방향으로 concat하여 정보를 합친 후에 추가적인 linear transformation을 거쳐서 thinking, machine 에 해당하는 단어의 dimension을 원하는  dimension으로 설정할 수 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/de6f302d-74d6-43db-97a2-b07671110392"></p>

이와 같은 과정들을 요약하면 위 그림과 같고, 이러한 Multi-head Attention layer 뒤에 Add & Normalization layer가 따라온다.
여기서 Add를 해주는 이유는 residual network에서 skip connection의 용도로 사용하기 위해서이며, vector를 encoding된 Hidden state vector에 더해주고(Add) layer를 normalization 해준다.


### Layer Normalization

skip connection까지 적용(Add 과정)해서 thinking과 machine 각각의 단어에 대해 최종적으로 encoding된 vector가 얻어졌을 때 layer normalization을 수행하기 된다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/cd170d60-7dc1-400d-be7b-4fca9ca3d340"></p>

위의 그림과 같이 이러한 Layer Normalization의 **첫 번째 단계**로는 각 단어마다 vector의 각 원소들을 모아서 한 단어 내에서 평균과 분산을 구하고 **두 번째 단계**로 각 단어마다 평균과 분산이 모두 0과 1이 되도록 정규화를 진행한다. **마지막 단계**로 원하는 평균과 분산을 주입하기 위해 각 단어의 원소별로 affine transformation 즉, y=ax+b 형태의 변환을 수행한다. 

### Positional Encoding

순서를 구분하지 못하는 self-attention model의 단점을 보완하고자 Positional Encoding을 통해 각 단어의 입력 vector에다가 이 단어가 몇 번째 순서로 나타났다 라는 정보를 주입한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/bcb6f51a-cd32-4c57-a588-e5282cbec3da"></p>

지금까지 설명한 self-attention 기반으로 sequence를 encoding하는 Transformer model의 전반적인 과정(Multi-head, Positional encoding, Add&norm 등)과 Transformer model의 전반적인 Decoding과정은 위의 그림을 통해 확인할 수 있다. 


## Transformer Model Decoder

기계 번역의 예시로 입력으로 I go home이라는 단어로 이루어진 sequence가 있다고 하자. self-attention block을 n번 반복해서 얻어낸 최종 encoding된 Hidden state vector를 encoder의 output으로 얻게 된다.

이것을 기계 번역을 했을 때 decoder의 첫 번째 time step의 입력으로 주어지는 단어는 start of sentence token일 것이고, time step에서 예측해야 하는 단어는 입력 영어 문장의 첫 번째 단어인 '나는' 과 다음 단어인 '집에' 와 '간다' 에다가 최종적으로 end of sentence token까지를 예측하는 auto regressive한 형태로 decoder가 동작할 것이다.

decoder에서도 역시 RNN 기반으로 sequence를 encoding하는 대신 self-attention module을 사용해서 sequence를 encoding하게 되는데 위의 그림에서 볼 수 있듯이 encoder와는 다르게 여기서 Masked Multi-head Attention layer가 존재한다.

### Masked Multi-head Attention

Decoder에서의 self-attention block은 다음 단어를 손쉽게 예측하는 문제를 방지하기 위해 masking이라는 연산을 수행한다. $QK^{T}$에 softmax를 취해 얻어진 attention 가중치에다가 각 query 단어를 기준으로 다음 time step들에 나타나는 단어들을 보지 못하도록 해당 단어와 그 이전에 나타난 단어들만을 encoding할 수 있도록 각 query 단어 다음에 나타나는 단어에 해당하는 attention weight 혹은 softmax의 입력으로 주어지는 logit값을 아래 그림처럼 -무한대로 대체해 준다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/0ab3137d-3686-4854-92c3-125699c5fd0c"></p>

(-무한대)라는 logit 값은 확률로 변환했을 때 0이라는 확률 값을 가지게 되어 SOS(start of sentence token)를 query로 사용했을 때 주어진 sequence를 encoding 할 때는 SOS단어만을 보고 encoding하게 되고 '나는'이라는 단어를 encoding할 때에는 SOS와 '나는' 이라는 단어까지 encoding을 한다.

이러한 과정을 통해 $QK^{T}$에 softmax를 취해 얻어진 attention 가중치에 후처리를 적용해준다.

이러한 Masked Multi-head Attention을 수행한 후에는 Add & Layer normalization을 수행하고, Decoder time step 에 encoding 된 Hidden state vector가 encoder Hidden state vector들에서 필요로 하는 정보들을 잘 적절하게 가져갈 수 있도록  하는 Multi-head Attention을 수행하고 다시 Add & Layer normalization을 수행하고, 그 다음엔 각 단어별로 나타난 encoding 된 Hidden state vector의 fully-connected layer(Feed Forward) 및 Add & Layer normalization을 거치고 최종적으로 나타난 decoder의 Hidden state vector들에다가 추가적인 linear layer 및 softmax layer를 통과시켜서 각 time step에서 다음 단어로 나타날 단어를 예측하게 된다.

## 자기 지도 학습

자기 지도 학습(Self-Supervised Learning)은 원시 data 혹은 별도의 추가적인 label이 없는 가려진 입력 data의 일부를 출력 혹은 예측의 대상으로 삼아서 data를 복원하도록 model을 학습하는 방법이다.

### Transfer Learning

대규모 data로 자기 지도 학습을 통해 학습된 model은 특정한 task를 풀기 위한 Transfer Learning의 형태로 활용될 수 있다.

Transfer Learning은 수집된 data를 가지고 impainting(가려진) task를 학습을 하는데 앞쪽 layer에서는 물체를 인식하기 위해 필요로 하는 유의미한 패턴을 추출하도록 학습하고, 뒤쪽 layer로 갈 수록 특정 task에 치중되어 별개의 task에서는 적용이 불가능한 정보들을 위주로 학습이 된다.

따라서 앞쪽 layer에는 다른 task에서도 공통적으로 적용가능한 정보들을 잘 추출해낼 수 있기에 이 앞쪽 layer를 통해 label정보를 얻고 뒤쪽 layer에 특정 task를 위한 output layer를 model에 덧붙여 성능을 높일 수 있따.

## 자기 지도 학습에 기반한 사전 학습 Model

이렇게 대규모로 사전 학습된 model을 다양한 target task에 적용해서 성능을 높이는 사례들은 특히 자연어 처리에서 많은 발전을 이뤘는데 그 중에 대표적인 모델로 **BERT**라는 모델이 있다.

### BERT

BERT는 pre-training of deep bidirectional transformers for language understanding이라는 논문의 제목을 가지고 있고, transformer model에서 encoder를 기반으로 한다. 여기서 bidirectional이라는 말은 masked language modeling(MLM)에 해당하는 의미를 가지고 있으며 추가적인 자기 지도 학습 task로서 next sentence prediction(NSP) task를 가지고 있다. 

따라서 BERT는 masking을 해서 단어를 맞추도록 하는 MLM과 두 개의 문장 간의 관계를 학습해 연속되는 문장인지 파악하는 NSP의 2가지 task로 자기 지도 학습을 수행하게 된다.

### GPT

자기 지도 학습에 기반한 사전 학습 Model에는 BERT 외에도 GPT 시리즈들이 있다. GPT(Generative Pre-trained Transformer)역시 transformer model을 사용했으며 BERT 와는 다르게 decoder를 사용한다.

