---
title : "Markdown 문법정리(수학수식, 특수문자)"
excerpt: "Github Blog에 글을 쓰기 위해 알아야 하는 마크다운 문법 중 수학수식과 특수문자에 대해 알아보자."

category :
    - Tips
tag :
    - Markdown

toc : true
toc_sticky: true
comments: true

---

깃허브 블로그를 만들고 처음 쓰는 글이다.
그 동안 프로젝트를 진행을 함에 있어서 Notion 과 같은 어플에 따로 기록해두었었는데, 이번 기회에 Github와 연동되는 Github Blog를 만들어보았다.

Github 블로그에 글을 작성하기 위해서는 마크다운 문법에 대해 알아야 하는데 이번 포스팅에서는 그 중에서도 필수적 요소인 수학수식과 특수문자에 대해 알아보도록 하겠다.

## 수학수식($)

우선 수학수식을 inline으로 표현하기 위해서는 달러표시를 사용하면 된다. (outline 의 경우 달러표시 2개) 간단하게 사칙연산을 표현해보겠다.

$$1 + 1 = 2 \;,\; 2 - 0 = 2 \;,\; 2 \times 2 = 4 \;,\; 6 \div 3 = 2$$

위와 같이 사칙 연산의 표현이 가능하고 

다음과 같이 코드를 작성하면 된다.

```
$$1 + 1 = 2 \;,\; 2 - 0 = 2 \;,\; 2 \times 2 = 4 \;,\; 6 \div 3 = 2$$
```

## 위첨자, 아래첨자 

위첨자의 경우 ^를, 아래첨자의 경우 _를 사용하면 된다.

$$\lambda^2,2^{2^2},a_1,\lambda_3$$ 

다음과 같이 코드를 작성하면 된다.

```
$$\lambda^2,2^{2^2},a_1,\lambda_3$$
```
참고로 위첨자, 아래첨자의 경우 띄어쓰기가 있으면 인식되지 않고, 일반 텍스트로 인식되기 때문에 꼭 ,뒤에는 띄어쓰기를 제거해야한다.
## 시간에 대한 미분


시간에 대한 미분, 즉, dot 표현은 \과 중괄호를 이용해 표현이 가능하다.

$$\dot{x},\ddot{y}$$

다음과 같이 코드를 작성하면 된다.

```
$$\dot{x},\ddot{y}$$
```
그냥 미분 표현은 키보드에 있는 '를 쓰도록 하자


## 특수문자 

\alpha 와 \beta 와 같은 특수문자들은 아래 표를 통해 사용하도록 하자.



| 이름 | 명령어 | 반환 | | 이름 | 명령어 | 반환|
| ---- | ------ | ---- | -- | --- | --- | --- |
| 알파 | \alpha | $$\alpha$$ | | 크사이  | \xi  | $$\xi$$ |
| 베타 | \beta | $$\beta$$ | |  오미크론 | \o  | $$o$$ |
| 감마 | \gamma | $$\gamma$$ | | 파이 | \pi  | $$\pi$$ |
| 델타 | \delta | $$\delta$$ | | 로 | \rho | $$\rho$$ |
| 엡실론 | \epsilon | $$\epsilon$$ | | 시그마 | \sigma | $$\sigma$$ |
| 제타 | \zeta | $$\zeta$$ | | 타우  | \tau  | $$\tau$$ |
| 에타 | \eta | $$\eta$$ | | 입실론  | \upsilon  | $$\upsilon$$ |
| 세타 | \theta | $$\theta$$ | | 퓌  | \phi | $$\phi$$ |
| 이오타 | \iota | $$\iota$$ | | 카이  | \chi | $$\chi$$ |
| 카파 | \kappa | $$\kappa$$ | | 오메가  | \omega | $$\omega$$ |
| 람다 | \lambda | $$\lambda$$ | | 뉴  | \nu | $$\nu$$ |
| 뮤 | \mu | $$\mu$$ | |   |   | |



**기타**

| 이름 | 명령어 | 반환 | | 이름 | 명령어 | 반환|
| ---- | ------ | ---- | -- | --- | --- | --- |
| hat | \hat{x} | $$\hat{x}$$ | | widehat  | \widehat{x} | $$\widehat{x}$$ |
| 물결 | \tilde{x} | $$\tilde{x}$$ | | wide물결 | \widetilde{x} | $$\widetilde{x}$$ |
| bar | \bar{x} | $$\bar{x}$$ | | overline | \overline{x} | $$\overline{x}$$ |
| check | \check{x} | $$\check{x}$$ | | acute | \acute{x} | $$\acute{x}$$ |
| grave | \grave{x} | $$\grave{x}$$ | | dot | \dot{x} | $$\dot{x}$$ |
| ddot | \ddot{x} | $$\ddot{x}$$ | | breve | \breve{x} | $$\breve{x}$$ |
| vec | \vec{x} | $$\vec{x}$$ | | 델,나블라  | \nabla  | $$\nabla$$ |
| 수직 | \perp | $$\perp$$ | | 평행 | \parallel | $$\parallel$$ |
| 부분집합아님 | \not\subset | $$\not\subset$$ | | 공집합 | \emptyset | $$\emptyset$$ |
| 가운데 점 | \cdot | $$\cdot$$ | | ... | \dots | $$\dots$$ |
| 가운데 점들 | \cdots | $$\cdots$$ | | 세로점들 | \vdots | $$\vdots$$ |
| 나누기 | \div | $$\div$$ | | 물결표 | \sim | $$\sim$$ |
| 플마,마플 | \pm, \mp | $$\pm$$ $$\mp$$ | | 겹물결표 | \approx | $$\approx$$ |
| prime | \prime | $$\prime$$ | | 무한대 | \infty | $$\infty$$ |
| 적분 | \int | $$\int$$ | | 편미분 | \partial | $$\partial$$ |
| 한칸띄어 | x \, y | $$x\,y$$ | | 두칸 | x\;y  | $$x \; y$$ |
| 네칸띄어 | x \quad y | $$x \quad y$$ | | 여덟칸띄어 | x \qquad y  | $$x \qquad y$$ |


## Reference

- [마크다운 수학수식](https://khw11044.github.io/blog/blog-etc/2020-12-21-markdown-tutorial2/#%EC%88%98%ED%95%99-%EA%B3%B5%EC%8B%9D-%EC%88%98%EC%8B%9D-%EB%B2%88%ED%98%B8)

- [마크다운 특수문자](https://jjycjnmath.tistory.com/117)
