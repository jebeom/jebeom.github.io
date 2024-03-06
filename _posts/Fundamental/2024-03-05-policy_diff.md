---
title : "[강화학습] On-Policy, Off-Policy, Online, Offline"
excerpt: "On-Policy Method, Off-Policy Method의 차이 와 Online RL, Offline RL의 차이에 대해 정리해보자"

category :
    - Fundamental
tag :
    - Reinforcement

toc : true
toc_sticky: true
comments: true

---
On-Policy Method, Off-Policy Method의 차이 와 Online RL, Offline RL의 차이에 대해 정리해보자


## On-Policy vs Off-Policy

강화학습 알고리즘은 On-Policy 방식과 Off-Policy 방식으로 분류될 수 있다.

On-Policy 방식과 Off-Policy 방식의 주요 차이점은 다음과 같다.


- On-Policy Method : Behavior Policy $=$ Target Policy

- Off-Policy Method : Behavior Policy $\neq$ Target Policy


여기서 Behavior Policy 와 Target Policy 의 개념은 다음과 같다.


- Behavior Policy $(b(a \vert s))$ : Action을 선택하고 Data Sample을 얻을 때 사용되는 Policy

- Target Policy $(\pi(a \vert s))$ : 평가(Evaluate)하고 업데이트(Improve)하고자 하는 Policy

즉, Policy 를 update 하기 위한 $V(s)$나 $Q(s,a)$를 계산하는데는 Target Policy $(\pi(a \vert s))$ 가 사용되며, Agent가 수행한 action에 의해 획득된 Sample들은 Behavior Policy $(b(a \vert s))$를 따른다.

아래와 같이 Behavior Policy의 예시를 들 수 있다.
 
$$ {s_{1},a_{1},r_{1}, ... , s_{T}} ~ b(a \vert s) $$


즉, **On-Policy Method**는 Behavior Policy $=$ Target Policy 이기에 Agent가 직접 행한 내용에 대해서만 학습하여 직접 경험한 내용을 바탕으로 policy를 개선해 나가는 과정이며, 

이와 반대로 **Off-Policy Method**는 이미 알고 있는 내용들을 바탕으로 아직 실행하지 않은 내용까지 예측해서 policy를 개선해 나가는 것이다.

On-Policy Method 의 경우 샘플을 수집되는 순간 policy가 업데이트 되고 그 뒤로는 기존과 다르게 행동하기 때문에 필연적으로 수집한 샘플을 한번 보고 버려야 한다. 또한, Q가 매번 바뀌기 때문에 transition에 대한 전 지식을 활용할 수 없다.

### 각 Method의 대표 알고리즘


On-Policy Method의 대표적 알고리즘은 SARSA이다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/46bf82e7-64e6-420b-8f44-184a7930d1d2" ></p>

SARSA는 behavior policy에 따라 action $a_{t+1}$을 선택하고 그에 해당하는 $Q(s_{t+1},a_{t+1})$을 이용하여 $Q_{new}$를 계산한다. 따라서 SARSA는 behavior policy와 target policy가 $\epsilon$-greedy로 동일하므로 On-Policy 이다.


반면 Off-Policy Method의 대표적 알고리즘은 Q-Learning이다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/a2237fde-9ecc-4058-802d-e2e6ed98dcb2" ></p>

Q-Learning의 경우 SARSA와 달리 behavior policy에 의해 $a_{t+1}$을 선택하여 $Q_{new}$를 계산하는 것이 아니라 $s_{t+1}$에서 가장 높은 Q-value를 가지는 $a_{t+1}$ 를 선택하여 $Q_{new}$를 계산한다. Q-Learning의 경우 target policy 가 $\epsilon$-greedy이다.


### 각 Method의 장단점

On-Policy의 장단점은 다음과 같다.

- 장점(Low bias error) : Behavior policy와 Target policy가 같으므로 일반적으로 bias error를 유발시키지 않아 성능이 안정적이다.

- 단점(Low sample efficieny) : 앞서 말했듯 On-policy의 경우 획득한 sample을 이용해 policy를 업데이트하고 폐기하므로 environment와의 상호작용이 많이 필요하다.


Off-Policy의 장단점은 다음과 같다. 

- 장점(High sample efficieny) : 과거의 Policy로부터 획득한 sample을 현재 policy를 업데이트할 때 재사용이 가능하므로 environment와의 상호작용을 적게 할 수 있다.

- 단점(High bias error) : 과거의 Policy와 현재의 Policy가 많이 달라진 경우, 과거의 sample은 현재의 policy를 업데이트 하기에 좋은 sample이 아닐 수 있다.


## Online RL vs Offline RL


Online 강화학습은 agent가 직접 환경과 상호작용하는 것으로 지속적으로 환경과 상호작용을 가정하는 문제 상황의 경우에 사용한다.

Online + On-Policy (e.g. PPO) / Online + Off-Policy (e.g. DQN, SAC) 알고리즘 모두 존재한다.

반면 Offline 강화학습은 agent가 직접 환경과 상호작용하지 않는다. 즉, 행위 알고리즘이 따로 존재하여 환경과 상호작용 전혀 없이 고정된 데이터만으로 동작하는 문제 상황의 경우에 사용한다.

Offline 강화학습은 policy optimization을 수행하는 동안 추가적인 샘플이 전혀 주어지지 않고, **이미 수집해놓은 데이터(Replay Buffer)**만을 이용해 policy를 수렴할때까지 학습시킨다.(추가적인 데이터 수집(exploration)이 없기 때문에 기본적으로 uncertainty가 높은 action은 피하는게 타당함)

Offline 강화학습의 경우 반드시 Off-Policy 알고리즘이어야 한다. 하지만 모든 Off-Policy 알고리즘이 Offline 상황에서만 동작하는 것은 아니다.(Online RL에서도 사용가능)

### Example

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/6fea1435-ced4-4ff0-a6a8-c20cc5ef433b" ></p>

위 그림과 같이 Off-Policy 강화학습(b)와 Offline 강화학습(c)에서는 Online 강화학습(a)에서 보지 못했던 버퍼를 확인할 수 있다.

예를 들어서 SAC는 Off-Policy 알고리즘이기 때문에 이미 수집된 데이터(Replay Buffer)만으로도 Policy Optimization이 잘 동작해야 할 것 같지만, 실제로 돌려보면 종종 발산한다. 이는 Out-Of-Distribution(OOD) action에 해당하는 Q-value가 과대추정되었을때, Policy는 Q가 (비정상적으로) 높은 action 영역을 샘플링하도록 업데이트되고, 동시에 과대추정된 Q를 bootstrap하는 과정에서 Q의 값이 발산하기 때문이다.

이러한 문제는 Online 상황에서는 데이터가 없는 부분의 Q가 (실제론 좋은 action이 아님에도) 과대추정되는 일이 생기더라도, 환경에서 그 action을 실제로 취해보면서, 사실 해당 action은 안좋았었다는걸 자연스럽게 깨닫고 Q 학습이 보정이 되지만, Offline 상황에선 추가적인 데이터 수집이 안되기 때문에 문제가 발생하는 것이다. 

이러한 이유로 Offline 강화학습 알고리즘들은 단순히 Online 상황을 가정하고 만든 Off-Policy 강화학습 알고리즘과는 좀 다르다.


Offline 강화학습 알고리즘들은 Off-Policy learning을 기본으로 깔고, 여기에 uncertainty가 높은 action을 피하는 매커니즘이 추가된 경우가 많다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/c428e84b-981b-4306-b175-f1d9b23ffc51" ></p>

위의 내용을 표로 정리하면 위와 같다.

여기서  Data Collection using current agent가 Online 강화학습, Fixed Dataset (no additional data collection)이 Offline 강화학습이다.


## Reference

- [이것저것 테크블로그 - 강화학습 이론(On-policy & Off-policy)](https://ai-com.tistory.com/entry/RL-%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5-%EC%9D%B4%EB%A1%A0-On-policy-Off-policy)


- [On-Policy, Off-Policy, Online, Offline 강화학습](https://seungwooham.tistory.com/entry/On-Policy-Off-Policy-Online-Offline-%EA%B0%95%ED%99%94%ED%95%99%EC%8A%B5)
