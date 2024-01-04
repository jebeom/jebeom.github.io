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


위 그림에서 $\lambda$는 eigenvalue 이며, $x$는 eigenvector이며 $det(A)$는  eigenvalue 들의 곱셈으로 표현되며 $tr(A)$는 eigenvalue들의 덧셈으로 표현된다.

### i) Cholesky Decomposition

Cholesky Decomposition은 가장 유명한 Matrix Decomposition으로 어떤 Matrix A가 Symmetric(대칭)하고, Positive Semidefinite(모든 eigenvalue 들이 0보다 큼)하면 Matrix $A = LL^T$ 로 표현이 가능하다.

여기서 L은 Lower-triangular Matrix로 Positive한 Diagonal을 기준으로 Upper Entry가 다 0이고, 밑에 Entry에만 값이 존재하며 이러한 L을 A의 Cholesky factor라고 한다.

이러한 Decomposition을 통해 Determinant 계산이 쉬워진다.
- 먼저 Determinant는 곱셈에 대해 분리가 되기에 $det(A)$는 $det(L)det(L^T)$로 표현이 가능하다.
- $det(L^T)$ 는 $det(L)$과 동일하기에 $det(A)=det(L)^2$가 된다.
- $det(L)$ 은 Lower-triangular Matrix의 Determinant이므로 Diagonal entry의 곱셈으로 표현이 가능하다
- 따라서 $det(A)$는 Diagonal entry의 곱셈의 제곱으로 표현이 가능하다.

### ii) Eigen Value Decomposition

EVD(Eigen Value Decomposition)은 Cholesky Decomposition 이상의 Determinant 연산과 여러 행렬 연산들을 간략하게 할 수 있는 Decomposition중 가장 유명한 방법이다.

test : $f'(x)$ 랑 f'(x)  
