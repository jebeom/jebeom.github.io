---
title : "[Paper Review] MOMA-Force: Visual-Force Imitation for Real-World Mobile Manipulation"
excerpt: "Imitation Learning을 활용한 MOMA-Force 논문을 읽고 정리해보자"

category :
    - Machine_Learning_Paper_Review
tag :
    - Imitation Learning

toc : true
toc_sticky: true
comments: true

---

Imitation Learning을 활용한 MOMA-Force 논문을 읽고 정리해보자

해당 논문은 아래 링크에서 원본을 확인할 수 있다.

- MOMA-Force: Visual-Force Imitation for Real-World Mobile Manipulation ([ArXiv](https://ar5iv.labs.arxiv.org/html/2308.03624))

## 1. Introduction

해당 논문에서는 로봇의 기초적인 능력인 mobility 와 manipulation을 결합한 Mobile Manupulation와 Imitation Learning을 통해 가정과 같은 contact-rich한 환경에서도 복잡한 Task를 성공적으로 수행하는 것을 보여주는 것을 목표로 한다.

기존에 있던 연구들은 force imitation이 없이 action 만을 imitation해서 복잡한 real world 환경에서의 안정성이 낮았다. 

이와 달리 본 논문에서는 Visual Encoder를 통해 얻어낸 wrench $\mathcal{F} = [m,f] \in R^{6}$ (m is the torque and f is the force)를 모방하여 **real-world에서도 안정적**으로 Task(서랍 열기, 세탁기 문 열기 등)를 수행할 수 있음을 보여준다.

## 2. Method

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/c919f48c-3d6b-4a2d-9f11-4239d3456d8e" ></p>  

Method는 크게 action-wrench prediction module과 admittance whole-body control module의 2가지 파트로 나뉜다. 

우선, 각 module에 대해 설명하기 이전에 각 input요소들이 무엇을 의미하는지 설명하겠다.

- $o_{t}$ : 로봇의 Arm에 장착된 카메라를 통해 얻어낸 **RGB image**

- $a_{t} \in SE(3)$ : **Kinematic Action**

- $g_{t} \in {-1,0,1}$ : **Gripper Action**

- $\hat{\mathcal{F}}_{t+1} \in R^{6}$ : **Target Wrench** for the next time step

- $\mathcal{T_{t}} \in {0,1}$ : **Terminate Flag**

- $\mathcal{P_{t}} = [R_{t},p_{t}]$ : **Current End-effector pose**

- $R_{t}\in SO(3)$ : **Rotation**

- $p_{t} \in R^{3}$ : **Translation** 


또한, 로봇이 모방하고자 하는 Expert Dataset의 경우 각 요소들의 지수에 e가 붙는다. (Ex : $o^{e}$)

만약, $g_{t} = 0$ and $\mathcal{T_{t}} = 0$ 이면, control module은 $a_{t}$, $\hat{\mathcal{F}}_{t+1}$를 input으로 가지고, 

모방하고자 하는 Target Wrench인 $\hat{\mathcal{F}}_{t+1}$ 에서 벗어나지 않게끔 제한하며, 

$a_{t}$을 통해 robot이 target pose를 취하게끔 한다.


또한 만약,  $g_{t} =1$ or $g_{t} = -1$ 이면 robot은 gripper를 open하거나 close하며, $\mathcal{T_{t}} = 1$ 이면 rollout이 중단된다.

### Action-Wrench Prediction

**Action-Wrench Prediction module**의 경우 offline observation encoding(의사코드 1~5번줄)과 online rollout(의사코드 6~17번줄)로 총 2가지 Phase로 구성되어 있다.

Imitation Data를 뽑아내기 위한 **Offline Phase**의 경우 시각적 관찰을 위해 Every Frame D 마다 Pre-trained Vision Encoder를 활용해 관찰 이미지 $o^{e}$를 $z^{e} \in Z^{e}$에 투영시킨다. (해당 논문에서는 ibot: Image bert pre-training with online tokenizer 논문의 self-supervised visual representation model을 사용) 

실제로 우리 로봇이 행하는 **Online Rollout Phase**의 경우 우선 Offline Phase와 같이 매 time step $t$마다 관찰 이미지 $o_{t}$를 동일한 Visual Encoder에 투영시켜 $z_{t}$를 얻어낸다. 다음으로 코사인 유사도를 통해 $z_{t}$와 $Z^{e}$의 유사도를 구한다. (의사코드 6~11번줄)

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/2d4ae525-b9fa-4591-a875-703178220ad3" ></p>

또한 이렇게 구해낸 유사도 중에서 가장 비슷한, 다시 말해서 Top-1 Frame의 index를 얻어낸다.

$$ i^* = \underset{i}{\operatorname{argmax}}  sim(z_{t},Z^{e})$$

얻어낸 index $i^* $ 는 다음으로 취할 kinematic action, gripper action and terminate flag를 예측하고 next time step에서의 Target Wrench를 얻어낼 때 사용된다.  


### Admittance Whole-Body Control

**Admittance Whole-Body Control module**의 경우 Action-Wrench Prediction을 통해 얻어낸 Pose를 실제로 Control한다.

next time step에서의 End-effector의 Target Pose $\hat{\mathcal{P}}_{t+1}$는 의사코드 14번줄과 같이 

$\mathcal{P_{t}} \circ a_{t}$  로 계산되는데, 이는 localization의 불확실성과 action prediction에서의 insufficient accuracy로 인해 정확하지 않을 수 있다.


이러한 이유로 해당 논문에서는 의사코드 15~16번줄 처럼 admittance term $ \triangle \mathcal{P_{t+1}} = (\triangle R_{t+1}, \triangle p_{t+1})$ 을 추가해주었다.

이러한 admittance term은 현재 time step에서의 Wrench와 next time step에서의 Wrench 즉, Target Wrench와의 차이를 보상해준다. 이는 아래 수식과 같이 wrench tracking error로 계산된다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/035c7b47-6ed1-4f3d-a28c-64e9182627c2" ></p>
 
<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/fee41635-6875-46b4-9e8f-a8070a6bb378" ></p>

이렇게 보완된 target pose $\hat{\mathcal{P}}_{t+1}$ 는 Whole-body controller로 보내진다. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/67d51f1c-39be-4797-9231-f0afeb3661bd" ></p>

WBC의 Cost Function과 제약조건은 위와 같다.

여기서 $u$는 base 와 arm의 decision variable vector(which includes velocity control)이며, $Q$는 joint velocity cost를 통합한다. $c = (O_{b},J_{m}^{a})$ 는 arm의 manipulability를 최대화 하기 위한 cost이고, $O_{b}$는 zero vector, $J_{m}^{a}$는 arm의 manipulability Jacobian이다.

또한 s.t. 부분에서 $J$는 base 와 arm의 generalized Jacobian이며, $v_{e} \in R^{6}$는 현재 End-effector Pose와 다음 time step의 Target end-effector Pose로부터 계산된 spatial velocity이다.

A와 B는 joint position constraint 이며, 해당 문제는 QP Problem으로 본 논문에서는 qpOASES를 통해 최적화 문제를 풀었다.

  

### 의사코드

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/01382ed4-d9c4-478a-8507-abc7ba531481" ></p>

지금까지 설명한 내용들을 의사코드로 표현하면 위와 같다.



## 3. Conclusion

본 논문에서는 Open cabinet drawer ,Open drawer, Open left door, Open right door, Rotate tap, Open washing machine 총 6개의 Task를 진행했다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/8b6a57ef-3f27-4077-a138-2eab0f25ab05" ></p> 

위의 Table과 같이 본 논문에서는 Force imitation의 효과를 입증하기 위해 force imitation이 없는 MOMA-Force w/o FC 와 force imitation과 rotation이 없이 translation만 있는 MOMA-Force w/o FC & Rot, 그리고 단순한 BC(Behavior Cloning)에 대해서도 동일한 Task를 수행했으며, 결과는 위 그림에서 알 수 있듯이 Force imitation을 사용한 경우 73.3%의 높은 Task 성공률을 가지는 것을 알 수 있다.
 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/3659994d-e75a-4c67-9376-ea8aeadde65b" ></p>

또한 위 그림과 같이 force imitation을 사용한 경우 다른 Method에 비해 Absolute Force와 Absolute Torque이 작고, Force와 Torque들이 작은 variance를 가져 oscillation이 작아 더 안정적인 것을 확인할 수 있다.

## Short Comment

본 논문에서는 사전에 훈련된 Visual Encoder를 통해 Force를 모방함으로써 높은 Task 성공률과 안정성을 가지는 것에 초점을 맞춘 것 같다. 하지만 Unseen Object에 대해서도 성공적인 Task 수행률을 가질 수 있을 것 같지 않다. 해당 논문에서의 Model과 Unseen Object에 대해서도 Generallized 된 Model을 합치면 좋은 Model을 만들 수 있을 것 같다.


