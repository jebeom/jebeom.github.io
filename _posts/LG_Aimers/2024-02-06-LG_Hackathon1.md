---
title : "[LG 해커톤] MQL 데이터 기반 B2B 영업기회 창출 예측 모델 개발(Base Code)"
excerpt: "LG Aimers에서 주최한 Hackathon 속 Dataset의 Feature들과 Base코드에 대해 알아보자"

category :
    - LG_Aimers_Hackathon
tag :
    - Hackathon

toc : true
toc_sticky: true
comments: true

---

LG Aimers에서 주최한 Hackathon 속 Dataset의 Feature들과 Base코드에 대해 알아보자

## 해커톤 문제 소개

이번 해커톤을 통해 풀 문제는 고객지수 중 하나인 **영업기회전환지수**이며, 이는 B2B Marketing에서 활용되는 고객지수이다. 

여기서 B2B Marketing은 기업 고객을 대상으로 영업 기회를 발굴해서 지속적으로 매출을 발생시키는 것을 목표로 한다. 자사 제품에 대해 관심을 보이고 구매 가능성이 있는 잠재 고객을 Lead라고 하는데, 이러한 Lead 고객 중에서도 BANT Quatation에 대한 답변을 한 고객을 Marketing Qualified Lead 의 약자로 **MQL 고객** 이라고 정의를 하며 이러한 MQL 고객을 대상으로 영업 사원을 할당하게 되고, 최종적으로 구매까지 이어지게 하기 위해 개인화 Marketing 활동을 진행하게 된다.

하지만 할당할 수 있는 영업사원의 수는 한정적이기에 MQL 고객 정보를 이용해서 영업 전환 성공 여부를 예측하는 AI Model 만든다면 각 고객의 영업 전환 성공 가능성을 지수로 표현할 수 있다. 이렇게 개발된 지수를 **영업기회전환지수**라고 한다.

본 해커톤에서는 약 30개의 Feature를 가진 학습용 Dataset인 train.csv 파일과, 제출용 Test Dataset인 submission 파일, 그리고 Base 코드가 제공된다.

이 대회에서 예측해야 하는 값은 **is_converted**로 각 데이터를 True or False로 분류함을 통해 영업 성공 여부를 예측한다. Feature, 다시 말해서 다른 Column들은 동일하지만 Submission파일에는 고객의 고유한 번호로 고객의 순서가 섞인 경우에도 올바른 채점이 진행되기 위한 값인 id칼럼이 추가적으로 있으며, is_converted은 모두 비어있다.

따라서 우리는 Train data set을 활용해 학습시킨 모델을 이용해 Submission 파일의 is_converted Column을 예측해야 한다. 예측을 하고 제출하면 LG측에서 가지고 있는 정답 파일과 내 예측 파일의 is_converted 값을 비교하여 F1 Score을 제공해준다. 제공해준 Base 코드의 경우 Public score가 대략 0.48 정도 나오는데, 수료를 위해선 예측 모델의 Public Score를 0.52 이상으로 개선해야 한다.


## Data Set

Data Set의 경우 아까 말했던 것 처럼 Train Data Set과 Test Data Set으로 나뉘어져 있다.

여기서 Data Set들의 Feature 종류는 아래와 같다.

- **bant_submit** : MQL 구성 요소들 중 [1]Budget(예산), [2]Title(고객의 직책/직급), [3]Needs(요구사항), [4]Timeline(희망 납기일) 4가지 항목에 대해서 작성된 값의 비율

- **customer_country** : 고객의 국적

- **business_unit** : MQL 요청 상품에 대응되는 사업부

- **com_reg_ver_win_rate** : Vertical Level 1, business unit, region을 기준으로 oppty 비율을 계산

- **customer_idx** : 고객의 회사명

- **customer_type** : 고객 유형

- **enterprise** : Global 기업인지, Small/Medium 규모의 기업인지

- **historical_existing_cnt** : 이전에 Converted(영업 전환) 되었던 횟수

- **id_strategic_ver** : (도메인 지식) 특정 사업부(Business Unit), 특정 사업 영역(Vertical Level1)에 대해 가중치를 부여

- **it_strategic_ver** : (도메인 지식) 특정 사업부(Business Unit), 특정 사업 영역(Vertical Level1)에 대해 가중치를 부여

- **idit_strategic_ver** : id_strategic_ver이나 it_strategic_ver 값 중 하나라도 1의 값을 가지면 1 값으로 표현

- **customer_job** : 고객의 직업군

- **lead_desc_length** : 고객이 작성한 Lead Descriptoin 텍스트 총 길이

- **inquiry_type** : 고객의 문의 유형

- **product_category** : 요청 제품 카테고리

- **product_subcategory** : 요청 제품 하위 카테고리

- **product_modelname** : 요청 제품 모델명

- **customer_country.1** : 담당 자사 법인명 기반의 지역 정보(대륙)

- **customer_position** : 고객의 회사 직책

- **response_corporate** : 담당 자사 법인명

- **expected_timeline** : 고객의 요청한 처리 일정

- **ver_cus** : 특정 Vertical Level 1(사업영역) 이면서 Customer_type(고객 유형)이 소비자(End-user)인 경우에 대한 가중치	

- **ver_pro** : 특정 Vertical Level 1(사업영역) 이면서 특정 Product Category(제품 유형)인 경우에 대한 가중치

- **ver_win_rate_x** : 전체 Lead 중에서 Vertical을 기준으로 Vertical 수 비율과 Vertical 별 Lead 수 대비 영업 전환 성공 비율 값을 곱한 값

- **ver_win_ratio_per_bu** : 특정 Vertical Level1의 Business Unit 별 샘플 수 대비 영업 전환된 샘플 수의 비율을 계산

- **business_area** : 고객의 사업 영역

- **business_subarea** : 고객의 세부 사업 영역

- **lead_owner** : 영업 담당자 이름

- **is_converted** : 영업 성공 여부. True일 시 성공



## Base 코드 설명

### 필수 라이브러리

```
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
```
수치 계산은 위한 numpy와, 표 형식의 데이터나 다양한 형태의 데이터 분석을 위한 numpy를 import 해주고, 머신러닝을 위한 라이브러리인 사이킷런(sklearn)을 import 해주었다.

또한 sklearn에서 평가지표 활용을 위한 라이브러리와 주어진 훈련 데이터를 훈련 데이터셋(Training Set)과 검증 데이터셋(Validation Set)으로 분류하기 위해 train_test_split을 import해주고 마지막으로 의사결정나무 모델을 사용하고 분류 문제이므로 DecisionTreeClassifier를 import해주었다.


### 데이터 전처리

```
df_train = pd.read_csv("train.csv") # 학습용 데이터
df_test = pd.read_csv("submission.csv") # 테스트 데이터(제출파일의 데이터)
```
우선 위와 같은 작업을 통해 데이터 셋들을 읽어왔다.

```
def label_encoding(series: pd.Series) -> pd.Series:
    """범주형 데이터를 시리즈 형태로 받아 숫자형 데이터로 변환합니다."""

    my_dict = {}

    # 모든 요소를 문자열로 변환
    series = series.astype(str)

    for idx, value in enumerate(sorted(series.unique())):
        my_dict[value] = idx
    series = series.map(my_dict)

    return series
```
범주형 데이터의 경우 숫자형 데이터로 변환해주어야 하기에 위와 같은 작업을 통해 레이블을 인코딩해주는 함수를 정의했다.

```
 # 레이블 인코딩할 칼럼들
label_columns = [
    "customer_country",
    "business_subarea",
    "business_area",
    "business_unit",
    "customer_type",
    "enterprise",
    "customer_job",
    "inquiry_type",
    "product_category",
    "product_subcategory",
    "product_modelname",
    "customer_country.1",
    "customer_position",
    "response_corporate",
    "expected_timeline",
]

df_all = pd.concat([df_train[label_columns], df_test[label_columns]])

for col in label_columns:
    df_all[col] = label_encoding(df_all[col])
```

인코딩할 범주형 Column들을 모아두고 pd.concat를 통해 trian data와 test data를 결합해준 후에 선언한 인코딩 함수를 통해 범주형 데이터들을 숫자형 데이터로 변환해주었다.


```
for col in label_columns:  
    df_train[col] = df_all.iloc[: len(df_train)][col]
    df_test[col] = df_all.iloc[len(df_train) :][col]
```

다음으로 위와 같은 작업을 통해 인코딩을 완료한 train data와 test data를 분리해주었다.


```
x_train, x_val, y_train, y_val = train_test_split(
    df_train.drop("is_converted", axis=1),
    df_train["is_converted"],
    test_size=0.2,
    shuffle=True,
    random_state=400,
)
```

한편, Train Set으로 학습시킨 모델을 바로 Test Set으로 학습시키면 해당 모델이 Train Set에만 특화되어 있을 수 있기에 모델의 성능 검증을 위해 꼭 Train Set을 Train Set과 Validation Set으로 분리 해주어야 한다.

따라서 train_test_split을 통해 x(학습 데이터)와 y(예측하고자 하는 Target값)을 분류하기 위해 Drop함수를 이용해 Train Set에서 특정 열("is_converted")을 제거해 주어(학습 데이터에 해당) 인수로 사용하고 is_converted Column만 따로 빼준 것(Target에 해당)도 인수로 사용했다. 여기서 axix=0 이면 행을, axis=1이면 열을 제거한다.

또한 test_size를 0.2로 설정해 Train Set의 20%를 Validation Set으로 설정해주고, Shuffle=True를 통해 데이터를 섞어주었다.

### 모델 학습

```
model = DecisionTreeClassifier()
```

Base 코드에서는 여러 Feature들을 이용해 **is_converted** Column을 분류하기 위해 위와 같이 의사결정나무 모델을 선택했다.

```
model.fit(x_train.fillna(0), y_train)
```

그 후에 위와 같이 모델을 학습 시켜 주었는데 여기서 x_train 즉, **is_converted을 제외한 Feature들**을 모아둔 학습 데이터에는 **결측치**가 존재하므로 fillna함수를 통해 결측치를 0으로 채워주었다.


### 모델 성능 보기
```
def get_clf_eval(y_test, y_pred=None):
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    print("오차행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(F1))
```

앞서 말했듯이 LG 해커톤에서 평가지표로 f1 score를 활용하기에 위와 같이 정확도, 정밀도, 재현율 등을 계산해주었다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/b06b3cad-f2a0-4438-9d54-7ba9cf8abca9" ></p>

f1 score 계산 방법은 위 사진과 같다. 

```
pred = model.predict(x_val.fillna(0))
get_clf_eval(y_val, pred)
```

그 후에 model.predict를 통해 검증 데이터셋의 학습 데이터를 활용한 모델이 **is_converted**를 예측한 것을 pred에 저장하고 앞서 선언한 함수에 pred와 검증 데이터셋의 예측(Target) 데이터를 인수로 주어 f1 score를 계산해주었다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/f516612a-5bf0-47a8-84d5-2801c0f40aa0" ></p> 

계산된 값들은 위와 같다.

### 제출

```
 # 예측에 필요한 데이터 분리
x_test = df_test.drop(["is_converted", "id"], axis=1)

test_pred = model.predict(x_test.fillna(0))
sum(test_pred) # True로 예측된 개수
```

위와 같은 과정을 거치며 만든 모델이 잘 예측한다고 생각하면 위와 같이 Test Data Set에서도 model.predict 함수를 사용해 **is_converted** Column을 True 나 False로 분류하고 sum함수를 통해 True로 예측된 개수를 파악한다. 


<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/7364ec52-e6a5-4dea-a129-bd2ce7ed5f43" ></p> 

True 예측된 개수는 위 사진과 같다.
```
 # 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("submission.csv")
df_sub["is_converted"]

 # 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)
```

마지막으로 제출할 파일인 Submission.csv을 읽어오고 비어 있던 is_converted 열을 우리가 학습시킨 모델을 사용해 예측값을 채워주고 저장한다.


다음 포스팅에서는 내가 데이터 분석을 하기 위해 사용했던 함수들과 수료 기준을 넘긴 코드, 그리고 시도는 해보았지만 사용하지는 않은 방법들에 대해 설명해보겠다.







