---
title : "[Paper Review] Adaptive Mobile Manipulation for Articulated Objects
In the Open World"
excerpt: "Imitation Learning과 Online RL을 활용한 Adaptive MOMA 논문을 읽고 정리해보자"

category :
    - Machine_Learning_Paper_Review
tag :
    - Imitation Learning
    - Reinforcement Learning

toc : true
toc_sticky: true
comments: true

---

Imitation Learning과 Online RL을 활용한 Adaptive MOMA 논문을 읽고 정리해보자

해당 논문은 아래 링크에서 원본을 확인할 수 있다.

- Adaptive Mobile Manipulation for Articulated Objects In the Open World ([ArXiv](https://ar5iv.labs.arxiv.org/html/2401.14403))


## 1. Introduction

기존에 있었던 Mobile Manipulation 관련 연구들은 단순히 Pick and Place 문제들에 초점을 두었다. 
하지만 본 논문에서는 Articulated Objects, 특히 가정과 같은 곳에서 문, 서랍, 냉장고 또는 캐비닛과 같은 조작 가능한 객체들과 관련해서 Task를 수행하는 것에 초점을 맞췄다. 

또한, 다양한 Unseen Object에 대해서도 효과적으로 일반화(Generallization)하는데에도 중점을 두었다.

이전 연구들과 비교했을 때 본 연구가 Advanced된 부분은 다음과 같다.

- **Full-stack Approach** : 하드웨어부터 소프트웨어, 학습 알고리즘에 이르기까지 시스템의 모든 층을 포괄했다.

- **Adaptive Learning Framework** : 소규모 Data Set에서 Imitation Learning을 통해 Policy를 초기화하고, Data Set밖에 있는 새로운 Object들에 대해서는 Online RL을 통해 지속적으로 학습하도록 했다. 

- **Low-cost MOMA hardware Platform** : 약 25,000 USD의 저비용으로 안전하고 자율적인 Online Adaptiion 이 가능한 MOMA 플랫폼을 개발했다.

## 2. Method

본 논문의 Framework는 크게 Action Space 와 Adaptive Learning 2가지로 구성되어 있다.

### Action Space

본 논문에서의 Action Space를 다루기 이전에 Action Space가 무엇인지 설명해보도록 하겠다.

Action Space란 가능한 모든 행동(또는 조작)의 집합을 의미하며, 이산적(Discrete) Action Space, 연속적(Continuous) Action Space 총 2 가지 주요 유형으로 나눌 수 있다.

**이산적(Discrete) Action Space**란 가능한 행동의 수가 제한적이고 명확하게 구분될 수 있는 경우이다. 예를 들어, '앞으로 이동', '뒤로 이동', '멈춤'과 같이 세 가지 행동만 가능한 경우 이산적이라 할 수 있다.

**연속적(Continuous) Action Space**란 행동이 연속적인 값으로 표현될 수 있는 경우이다. 예를 들어, 로봇 팔의 각도를 조절하는 경우, 각도는 연속적인 값으로 설정될 수 있으며 이로 인해 무한히 많은 가능한 행동이 존재할 수 있다.

본 논문에서의 Action Space는 로봇이 Articulated Objects 들과 상호작용할 때 사용하는 파라미터화된 Primitive Action Space를 사용하며, 여기에는 주요 Primitive Action인 Grasping(G) 과 Constrained Mobile-Manipulation(M)이 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/4a681d27-fff4-4ccb-b8a4-a1c3748e9f83" ></p>

각 요소들에 대한 설명은 다음과 같다.

- $I_{s}$ : **Initial observed image**

- $G(g)$ : **파라미터화된 Grasp Primitive**

- $M(C_{i},c_{i})$ : **파라미터화된 Constrained Manipulation Primitive**

- $C_{i}$ : **Discrete parameter**

- $c_{i}$ : **Continuous parameter**

- $I_{f}$ : **Final observed image**

- $R$ : **Reward** for the trajectory


이렇게 **구조화 된 Action Space**는 Full Action Space보다 쉽게 표현할 수 있으며, **적은 Sample을 사용해도 효과적인 Policy를 학습하기에 충분하다.** 

### Adaptive Learning

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/55d2ee27-073f-4880-9736-e64ef4eedc3d" ></p>

다음으로 본 논문에서의 적응형 학습방법에 대해 알아보자.

우선 Initial observed image $I_{s}$ 가 주어지면 Classifier $\pi_{\phi} ((C_{i})_{i=1}^{N}  \vert I)$를 통해 constrained mobile-manipulation을 위한 N개의 Discrete parameter를 예측한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/40d73ae5-778d-46c4-bd02-9eca9682ad4a" ></p>

다음으로 위와 같은 conditional policy network를 사용해 grasping primitive의 continuous parameter와 N개의 Continuous parameter를 생성한다.

이후에 로봇은 파라미터화된 primitives 를 open-loop 방식으로 하나씩 실행한다.

이제 본격적으로 Unseen Object에 대해서도 Task를 잘 수행하는 방법에 대해 알아보자

#### I.Imitation

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/d2564d40-4b9d-4829-9e49-ad795138c1aa" ></p>

우선 Expert로부터 나온 Small Data Set을 Imitation함으로써 policy parameters $\pi_{\theta,\phi}$를 학습한다. 수식은 위와 같다.

#### II. Online RL

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/9dbcd2c7-facb-430f-b0ca-f80ba469f5be" ></p>

다음으로 Behavior Cloning Data에 없는 새로운 articulated object에 대해서도 원활히 Task를 수행하기 위해 로봇이 환경과 상호작용 하며 Reward가 최대가 되도록 Policy를 개선시킨다. 수식은 마찬가지로 위와 같다.

Reward의 경우, 본 연구에서는 Large vision language model 중에서도 CLIP 모델을 사용하여 로봇 실행 후 관찰된 이미지와 두 가지 텍스트 프롬프트 사이의 유사성 점수를 계산했다. 

사용된 두 프롬프트는 "문이 닫혀 있다(door that is closed)"와 "문이 열려 있다(door that is open)" 이다.

최종 관찰된 이미지와 각 프롬프트의 유사성 점수를 계산하고, 이미지가 "문이 열려 있다"는 프롬프트에 더 가까우면 +1의 보상을, 그렇지 않으면 0의 보상을, 만약 안전 보호 장치가 작동하면 -1의 보상을 주었다.

#### III. Overall Finetuning

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/e6335189-4013-4ea2-82da-09c7b9f260b5" ></p>

마지막으로 Online RL을 통해 개선시킨 Policy가 Imitation을 통해 얻어낸 Policy와 너무 달라지지 않게 하기 위하여 위와 같은 Finetuning을 실행한다.

 
## 3. Conclusion


본 논문에서는 4가지 유형의 articulated objects 즉, levers (type A), knobs (type B), revolute joint (type C), prismatic joint (type D)에 대해 각 범주마다 2개의 test object를 통해 실험했다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/96d3f659-b5a5-4924-abaf-565907452a1e" ></p>

그 결과 위 그림과 같이 모든 객체에 대한 평균 성공률을 50%에서 95%로 향상시켰다. 또한, 본 연구에서의 접근방법은 실험 A에서 볼 수 있듯이 초기의 Policy가 Task를 대부분 실패하는 경우에도 객체를 조작하는 방법을 스스로 학습할 수 있는 것을 확인할 수 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/295b36f6-6959-485c-85f4-5b561d5447a4" ></p>

또한, 위 Table에서 확인할 수 있듯이 Reward를 인간이 레이블링한 경우(Adapt-GT)와 본 논문에서 사용한 CLIP 모델을 사용해 **Reward를 자체적으로 학습하는 경우(Adapt-CLIP)** Task를 성공하는 데에 있어서 유사한 성능을 보이는 것을 확인할 수 있다.

** GT = Ground Truth

## Short Comment

본 논문의 경우 초기에 Imitation을 통해 Policy를 초기화하고(Agent에게 줄 Guideline 느낌), Online RL을 통해 Policy를 개선시켜주어 Unseen Object들에 대해서도 Task를 성공적으로 수행했다.

특히, Reward를 선정할 때 인간이 설정하는 것이 아니라 LVLM(Large Vision-Language Model)을 사용하여 Reward를 Agent가 스스로 학습할 수 있게끔 한 점이 흥미로웠다. 






