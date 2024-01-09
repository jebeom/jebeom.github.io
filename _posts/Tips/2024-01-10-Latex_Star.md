---
title : "LaTex 문법에서 위첨자에 * 쓰는 방법"
excerpt: "LaTex 문법에서 위첨자에 * 쓰는 방법에 대해 알아보자"

category :
    - Tips
tag :
    - Markdown

toc : true
toc_sticky: true
comments: true

---
LaTex 문법에서 위첨자에 * 쓰는 방법에 대해 알아보자

Markdown은 Latex문법을 지원하는데, 수학 수식을 사용하다 보면 간혹 $a^* $처럼 위첨자에 * 를 써야하는 경우가 있다.

이런 경우 일반적으로는 아래 코드 처럼 작성한다.

```
Sa^{*}$
```
코드에 별 문제는 없어 보이는데 아래 그림 처럼 글자가 깨져버린다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/8ca2348c-fa6a-4a5c-b030-11b20aed42e2"></p>

이러한 문제를 해결하기 위해 위첨자에 *를 써줄 경우에는 아래 코드와 같이 작성해주자.

```
$a^* $
```
그러면 아래 그림과 같이 정상적으로 출력된다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/ef651569-8daf-403d-bf93-d760dcb8ee85"></p>

문제가 생긴 코드와의 차이점은 중괄호를 빼고 공백을 넣어주었다는 점이다. 둘 중에 어느 하나라도 틀리면 글자가 깨져버리니 **꼭 중괄호는 넣지 말고, * 뒤에는 공백을 넣어주자**


이상으로 포스팅을 마치도록 하겠다. 
