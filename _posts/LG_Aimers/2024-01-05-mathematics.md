---
title : "[LG Aimers] Mathematics for ML"
excerpt: "행렬 분해, 최적화, 주성분 분석 등 데이터를 다루기 위한 Mathematics들을 알아보자"

category :
    - LG Aimers
tag :
    - mathematics

toc : true
toc_sticky: true
comments: true

---

행렬 분해, 최적화, 주성분 분석 등 데이터를 다루기 위한 Mathematics들을 알아보자

> 본 포스팅은 LG Aimers 수업 내용을 정리한 글로 모든 내용의 출처는 [LG Aimers](https://www.lgaimers.ai)에 있습니다.

## Matrix Decomposition

우선 행렬 분해에 들어가기 전 간단한 개념 정리를 해보자.

Determinant란 어떤 matrix의 역행렬을 구할 때 분모에 위치하는 항으로 Determinant의 중요한 성질 중 하나는 곱셉에 대해 분해가 된다는 점이다.
- $det(AB)=det(A)det(B)$

Trace란 Matrix 의 Diagonal Entry를 다 더한 형태로 Trace의 중요한 성질 중 하나는 덧셈에 대해 분해가 된다는 점이다.
- $tr(A+B)=tr(A)+tr(B)$

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/89b0970a-a7d5-49d8-831f-f1eb2ac0397f" width = "700" ></p>

위 그림에서 $\lambda$는 eigenvalue 이며, $x$는 eigenvector이며 $det(A)$는  eigenvalue 들의 곱셈으로 표현되며 $tr(A)$는 eigenvalue들의 덧셈으로 표현된다.

### i) Cholesky Decomposition

Cholesky Decomposition은 가장 유명한 Matrix Decomposition으로 어떤 Matrix A가 Symmetric(대칭)하고, Positive Semidefinite(모든 eigenvalue 들이 0보다 큼)하면 Matrix $A = LL^T$ 로 표현이 가능하다.

여기서 L은 Lower-triangular Matrix로 Positive한 Diagonal을 기준으로 Upper Entry가 다 0이고, 밑에 Entry에만 값이 존재하며 이러한 L을 A의 Cholesky factor라고 한다.

다음과 같은 Decomposition 과정을 통해 Determinant 계산이 쉬워진다.
- 먼저 Determinant는 곱셈에 대해 분리가 되기에 $det(A)$는 $det(L)det(L^T)$로 표현이 가능하다.
- $det(L^T)$ 는 $det(L)$과 동일하기에 $det(A)=det(L)^2$가 된다.
- $det(L)$ 은 Lower-triangular Matrix의 Determinant이므로 Diagonal entry의 곱셈으로 표현이 가능하다
- 따라서 $det(A)$는 Diagonal entry의 곱셈의 제곱으로 표현이 가능하다.

### ii) Eigen Value Decomposition

EVD(Eigen Value Decomposition)은 Cholesky Decomposition 이상의 Determinant 연산과 여러 행렬 연산들을 간략하게 할 수 있는 Decomposition중 가장 유명한 방법으로 Diagonal Matrix의 개념을 이용한다.

우선 EVD에 대해 설명하기에 앞서 Diagonal Matrix에 대해 알아보자.

Diagonal Matrix란 Diagona entry만 존재하고 나머지 entry는 전부 0인 형태로 위의 그림과 같이 Diagonal Matrix의 지수승도 쉽게 표현이 되고, 역행렬도 Diagonal entry의 역수로 표현이 되고, Determinant 또한, Diagonal enrty의 곱셈으로 쉽게 표현이 가능하다.

만약 어떤 Matrix A가 $D=P^{-1}AP$ 의 형태로 표현된다면 diagonalizable 하다고 할 수 있으며, 여기서 P가 orthogonal 즉, $PP^T=I$ 라면 A는 orthogonally diagonalizable하다고 할 수 있다.(여기서 D는 Diagonal Matrix이다.)
 
또한, Matrix A가 Symmetric한 경우에는 항상 Orthogonally Diagonalizable 즉, $A=PDP^T$ 로 표현이 되는데 여기서 P를 Eigenvector들>을 모아놓은 Matrix, D를 Eigenvalue들을 모아놓은 Matrix라고 정하면 위와 같이 표현이 되는 것을 쉽게 확인이 가능하다.

정리하자면 어떤 Matrix가 A가 Diagonalizable 하게 되면 $A^k = PD^kP^{-1}$ 와 같이 표현될 수 있으며 determinant의 경우 $det(A)=det(D)$ 즉, Diagonal Matrix의 곱이 되므로 이러한 EVD(고유값 분해)를 통해 행렬 곱 연산을 쉽게 할 수 있다..

### iii) Singular Value Decomposition

SVD(Singular Value Decomposition)는 일반적인 matrix에 전부 적용할 수 있는 Decompostion개념으로 A가 Square Matrix도 아니고, Symmetric 하지도 않는 경우에 사용한다. ( Symmetric 한 경우 EVD 사용 )

SVD(Singular Value Decomposition)는 어떤 Matrix A가 주어졌을 때 $A=U$ $\sum$ $V^T$ 형태로 분해하는 꼴로 분해하는 것인데 여기서 $U$와 $V$는 항상 Orthogonal Matrix가 되며 $\sum$는 Diagonal Matrix가 된다. 이러한 $\sum$의 Diagonal entry를 **Singular Value**라고 부르며, $U$와 $V$를 구성하는 vector들을 각각 **Left and Right Singular Vector**라고 부른다.

$S=A^TA$ 형태의 Matrix 에서 $S$는 항상 Symmetric 하고, 항상  Positive Semidefinite( 모든 eigenvalue 들이 0보다 큼) 하므로 $A^TA$에 대해서는 EVD의 적용 즉, $A^TA=VDV^T$로 표현이 가능하다. 따라서 아래와 같은 증명과정을 통해 $A=U$ $\sum$ $V^T$로 표현이 가능하다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/d3364334-0d7e-4f25-85a9-a2c7b9ec2d4a" width = "700" ></p>

따라서 어떤 행렬 $A$의 Singular Value Decomposition 은 행렬 $A^TA$의 Eigenvalue Decomposition과 동일하며 만약 A가 Symmetric 하면 EVD 와 SVD는 동일한 개념이다.

## Optimization

Machine Learning Model을 학습한다고 했을 때 보통 Optimization 문제로 구성이 되며 이러한 문제들은 Model의 좋은 Parameter들을 찾는 과정이다.

### i) Gradient Descent (Unconstrained Optimization)

Unconstrained Optimization 에서 R차원 vector를 Singular R로 Mapping하는 함수 f를 최소화하기 위한 알고리즘은 다음과 같다.

$$x_{k+1}=x_k+r_{k}*d_{k}(k =0,1,2, ...) $$

여기서 $r_{k}$ 는 step-size(Scaler 값), $d_{k}$는 방향성을 나타내는 Direction 이 된다. 만약 Gradient와 내적 값이 0보다 작게 되고 step-size $\alpha$를 잘 정할 수 있다면 업데이트를 했을 때 현재 값보다 더 낮아지는 $\alpha$가 존재한다.

즉, Gradient와 반대 방향으로 방향을 잡고, $r_{k}$ 즉, step-size를 잘 조절이 가능하면 함수 f를 최소화하는 다음 업데이트 값을 구해나가며 어떤 함수가 Local Optima(Optimal Point)로 수렴한다. 

**Steepest Gradient Descent**란 $d_{k}$를 Gradient의 반대 방향, 즉 내적해서 0이 되는 방향으로 선택한다.

보통 Optimization을 Machine Learning에 쓰다 보면 Objective Function이 Data로 부터 정해지는 경우들이 많다. 예를 들어 $L(\theta)$ 라는 Objective Function 에서 $\theta$ 는 모델의 Parameter, $L$ 은 모델의 손실함수가 된다.

최소화 할 Loss Function은 Data point에 대한 Loss의 Summation 형태로 표현되는데 얼만큼의 Data로 정의하느냐에 따라 다음과 같이 분류된다.

- Batch gradient : 모든 데이터 point들을 다 고려하고 계산해서 정확하게 업데이트
- Mini- Batch gradient : 계산의 효율성을 위해 특정 subset을 구해서 그 subset에 있는 gradient만 계산해서 업데이트
- Stochastic gradient :Mini-Batch gradient의 일종으로 subset을 구할 때 subset을 통해 구한 gradient의 expectation이 original full batch gradient와 동일하도록 디자인해서 업데이트

이외에도 $x_{k+1}=x_{k}-r_{i} \nabla f(x_{k})^T$ 에 추가항인 $\alpha(x_{k}-x_{k+1})$를 붙여주어 이전에 업데이트한 값들도 고려한 **Gradient Descent with momentum** 을 통해 수렴화를 더 가속화 시킬 수 있다.

### ii) Lagrange Muliplier (Constrained Optimization)

**Lagrangian Function**는 Original objective($f(x)$) + inequality constrains($g$) 에 $\lambda$ 를 곱해 더한 형태 + equality constrains($h$)에 $\nu$를 곱해 더한 형태이며 Lagrangian 함수를 $\lambda$ 와 $\nu$를 고정했을 때 $x$에 대한 infimum 값을 **Lagrange dual function** 이라 부른다.

여기서 $\lambda$와 $\nu$를 Lagrange Mulipliers(dual variables)라 부르고 $\lambda$는 항상 0보다 커야하며 $\nu$는 제약이 없다.이러한 Lagrange Muliplier 는 constrained optimization을 unconstrained optimization 즉, gradient descent로 풀기 위해 도입되었다.

이러한 Lagrange dual function이 Original Optimal Value의 Lower bound 가 된다. 즉, dual optimal value 를 $d^* $, primal optimal value를 $p^* $ 라 하면 $d^* \leq p^*$ 이다. 따라서 Primal Optimization 이 풀기 힘들더라도 Dual Optimization을 항상 풀 수 있다.

### iii) Convex Sets and Convex Functions

$f(x)$가 Convex Function 이고, subset을 이루는 $x$가 Convex Set이 될 때 Convex Optimization이라고 부른다.

문제가 쉽게 풀리느냐 안풀리느냐는 문제가 선형이냐 비선형이느냐가 아니라 f와 조건들이 convex하냐 아니냐로 나뉘기에 Convex Optimzation은 굉장히 중요하다.

**Convex Set**은 Set안에 있는 2개의 point를 잡고 그 point들을 이었을 때 선분이 항상 set안에 있으면 convex set이다. 아래의 2번째 그림처럼 set안에 위치하지 않는 선분이 있거나 아래의 3번째 그림처럼 set에 빵구가 뚫려있으면 convex set이 아니다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/408e6e48-5d28-4309-9785-4f3e8fa5ea12" width = "400" ></p>

위의 그림에서 제일 왼쪽의 그림은 Convex Set이다.

**Convex Functoin** 은 만약 어떤 함수에 접선을 그었을 때 접선보다 함수가 항상 위에 있으면(볼록함수) Convex function이며 만약 함수에 음수를 취했을 때 즉, -f가 convex하면 Concave function(오목함수) 이라 부른다.

또한 f가 2번 미분 가능할 때 2번 미분한 Hassian Matrix 가 Positive Semidefinite Matrix(eigenvalue >0)인 경우 Convex Function 이다.

이러한 Convex Function에서 함수를 최소화하는 점과 gradient가 0이 되는 점이 동일하다는 성질이 있다. (Global optimum = Local optimum )

참고로 Lagrange dual function의 경우 $\lambda$와 $\nu$ 에 대한 선형함수이므로 Concave 함수가 되기도 하고, Convex 함수가 되기도 한다.

### iV) Convex Optimization

Convex Optimization 이란 Convex Function와 Convex Set에 Constraint을 주고 하는 Optimization이다. 이러한 Convex Optimization에는 strong duality 즉, $d^* = p^*$ 라는 좋은 성질이 있어 풀기가 쉽다.

어떠한 Optimization Problem이든 KKT Condition이 Optimality의 필요조건이 되고, Convex Optimization 의 경우 KKT Condition이 필요충분조건이 된다. 즉, KKT Condition을 만족하는 $x^* $와 $\lambda^*$를 찾을 수 있다면 그것이 결국 Primal 과 Dual Optimum이 된다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/f2047cdd-a0af-47a2-9660-3d0b38f65f4f" width = "600" ></p>

정리하자면 Objective 도 Linear 하고, Constraint 도 Linear 한 Linear Programming 의 경우 Convex Optimization 형태이기에 Primal Soultion과 Dual Solution이 같게 되고 KKT 조건들을 통해 Primal Soultion과 Dual Solution을 구하는 알고리즘을 만들 수 있다.
