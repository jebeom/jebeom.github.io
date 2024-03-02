---
title : "LaTex 문법에서 문자 위,아래에 문자 쓰는 방법"
excerpt: "LaTex 문법에서 문자 위,아래에 문자를 쓰는 방법에 대해 알아보자"

category :
    - Tips
tag :
    - Markdown

toc : true
toc_sticky: true
comments: true

---
$\underset{a}{\operatorname{argmax}}$ 와 같이 LaTex 문법에서 문자 위,아래에 문자를 쓰는 방법에 대해 알아보자

## 문자 아래

Markdown은 Latex문법을 지원하는데, 수학 수식을 사용하다 보면 $\underset{a}{\operatorname{argmax}}$ 와 같이 **문자 아래에 숫자나 문자를 써주어야 하는 경우**가 발생한다.

이런 경우 아래와 같이 코드를 작성할 수 있다.

```
$$\underset{a}{\operatorname{argmax}} Q(s,a)$$
```

결과물은 아래와 같다.

$$\underset{a}{\operatorname{argmax}} Q(s,a)$$

다만 Markdown에서는 Latex를 일부 지원하므로 mathrm 대신 operatorname을 사용해주었으며, Overleaf와 같은 곳에서 Latex를 사용할 때는 operatorname 자리에 mathrm을 적어주자.

## 문자 위

이번에는 문자 **위에 숫자나 문자**를 적어보자. 코드는 아래와 같다.

```
$${\overset{x=1}{min{f(x)}}}$$
```
결과물은 아래와 같다.

$${\overset{x=1}{min{f(x)}}}$$

## 문자 아래와 위 둘 다

배운 것들을 활용해서 여러가지 테스트를 해보도록 하자.

### Test 1

```
$$\underset{k=1}{\overset{n}{\prod}} k$$
```
결과물은 아래와 같다.

$$\underset{k=1}{\overset{n}{\prod}} k$$


### Test 2

```
$$\underset{u \subseteq U^{n}}{\overset{x=1, a=2}{sup}} [f(x,a)]$$
```
결과물은 아래와 같다.

$$\underset{u \subseteq U^{n}}{\overset{x=1, a=2}{sup}} [f(x,a)]$$

### 결론

즉, 문자 위 아래 둘 다 문자를 사용해야 한다면 \underset 즉, **아래에 쓸 문자를 먼저 사용**해주고 \overset 즉, 위에 쓸 문자를 사용해주어야 하며, **중괄호({})**를 통해 \overset 안에 $\prod$와 같이 위랑 아래를 표시할 문자를 **묶어**주어야 한다.




