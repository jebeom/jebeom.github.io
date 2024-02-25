---
title : "[LG 해커톤] MQL 데이터 기반 B2B 영업기회 창출 예측 모델 개발(Improved Code)"
excerpt: "Hackathon을 진행하며 데이터 분석에 사용한 함수와 성능 개선 코드, 그리고 시도는 해보았지만 사용하지는 않은 방법들에 대해 알아보자"

category :
    - LG_Aimers_Hackathon
tag :
    - Hackathon

toc : true
toc_sticky: true
comments: true

---

Hackathon을 진행하며 데이터 분석에 사용한 함수와 성능 개선 코드, 그리고 시도는 해보았지만 사용하지는 않은 방법들에 대해 알아보자

> 코드 원본은 [여기](https://github.com/jebeom/LG_Aimers_4th/blob/main/code-0.5202.ipynb)에서 확인 가능합니다.

## 데이터 분석에 사용한 함수들 

### describe() 함수 (결측치, 이상치 확인하기)

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/50fa7988-b71f-487c-979d-dccf2a385c46" ></p>  

describe함수의 경우 위의 그림과 같이 데이터프레임의 형태로 몇 개의 Feature들이 있는지 파악이 가능하고, Count를 통해 각 Feature마다 데이터가 몇개 있는지 파악 가능하다. 예를 들어 bant_submit 처럼 결측치가 없는 경우, 59299개의 데이터가 존재하고, com_reg_ver_win_rate 의 경우 44731개의 결측치가 존재해, Count가 14569로 표시된 것을 확인할 수 있다.

또한, std행을 통해 각 Colum별 표준편차를 확인할 수 있으며, customer_country와 같이 std가 너무 큰 값을 가지고 있다면 이상치가 존재한다고 생각해볼 수 있다.

### info() 함수 (결측치, 데이터 타입 확인하기) 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/8b0de239-2b90-4b3b-88dc-10fdf2f9a29c" ></p>

info함수의 경우 위의 그림과 같이 결측치가 있는지 없는지와 데이터 타입을 간단하게 확인 가능하다.


### .value_counts() 함수 (값의 상대 빈도 구하기)

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/e54d7af0-1762-4a48-9bf8-033b18546dbc" ></p>
위 그림과 같이 .value_counts() 함수를 통해 확인 하고자 하는 Column의 각 Value(Class)에 대한 발생횟수를 파악할 수 있다.


### .score() 함수 (모델 성능 측정)

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/8aa38408-9ec1-4604-8908-40ba1af46664" ></p>

위 그림과 같이 .score() 함수를 통해 학습한 모델이 각 데이터 별로 얼마나 잘 맞는지 정확도를 확인할 수 있다.

### Seaborn 라이브러리 (데이터 시각화 하기)

```
# 상관관계 파악 / 상관 관계 높고, 중요도 낮은 특성을 제거해 모델 복잡도를 줄여 일반화 높임
import seaborn as sns

# 상관계수 행렬 계산 (절대값 사용)
corr_matrix = np.abs(df_train.corr())

# plot
plt.figure(figsize=(20, 30))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# 상관계수 임계값 설정
corr_threshold = 0.5

# 상관계수가 임계값 이상인 특성 찾기
to_drop = [column for column in corr_matrix.columns if any(corr_matrix[column][corr_matrix[column] != 1] > corr_threshold)]
print(to_drop)
````

위의 코드와 같이 Seaborn함수를 import 해줌으로써 데이터의 시각화가 가능하며, Seaborn 라이브러리에서 제공해주는 그래프(plot)의 종류는 아래 그림과 같다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/d67e3352-ad73-4781-9ce6-35534b16a1d9" ></p>

필자의 경우 각 feature 마다 상관계수를 파악하기 위해 heatmap을 사용했다. heatmap을 통해 나타낸 각 feature간의 상관계수값은 아래 그림과 같다. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/8cae95f0-241d-45cd-9504-9410d439a092" ></p>  

여기서 모델의 복잡도를 낮추어 일반화 성능을 올리기 위해 상관 관계가 높고, 중요도 낮은 특성을 제거하기 위해 자기 자신을 제외하고 다른 Feature들과 상관계수가 0.5 이상인 Feature들을 찾아 to_drop에 넣고 출력해보았다.


<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/7b795501-a413-4227-a008-26b73d3f851e" ></p>

그 결과 위 그림과 같이 6개의 Feature들이 나타났고, 모델을 학습시킨 후 6개의 Feature 중에서 중요도가 떨어지는 Feature들은 제거하기로 했다.

### feature_importances_ (특성 중요도 구하기)

```
ser = pd.Series(model.feature_importances_,index=df_train.drop("is_converted", axis=1).columns)
 # 내림차순 정렬을 이용
imp = ser.sort_values(ascending=False)
plt.figure(figsize=(8,6))
plt.title('Feature Importances')
sns.barplot(x=imp, y=imp.index)
plt.show()
```
Tree 기반 모델의 경우, 모델을 학습시킨 후 feature_importances_를 통해 각 Feature들의 중요도를 파악이 가능하다. 다시 말해서 feature_importances_ 는 결정트리에서 노드를 분기할 때, 해당 Feature가 분류를 함에 있어서 얼마나 영향을 미쳤는지를 표기하는 척도이며 Normalize된 ndarray를 반환 하기에 0~1 사이의 값을 가진다. 

여기서 pd.Series 함수를 이용해 feature_importances_ 값에 맞는 feature의 name을 index로 주었다. 따라서 imp에는 내림차순으로 정렬된 각 Feature마다의 중요도 값이 들어가고, imp.index에는 각 Feature 중요도 값에 따른 Feature들의 이름이 들어간다.

이러한 Feature 중요도를 위의 코드와 같이 barplot 형태로 그려보면 아래 그림과 같이 각 Feature들의 중요도를 쉽게 파악할 수 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/24dfdd01-798c-4cfc-b168-e76908753155" ></p>

따라서 모델을 일반화 시켜 Public Score를 개선 시키기 위해 상관계수가 높은 6개의 Feature 중에서 해당 Feature와 상관계수가 높은 Feature들이 상대적으로 많고, Feature 중요도가 낮은 **'ver_win_rate_x'** 와 **'ver_win_ratio_per_bu'** Column들을 데이터셋마다 제거해주었다.

## Improved 코드 설명

다음으로 위에서 설명한 함수들을 이용해서 성능을 개선시킨 코드에 대해 설명해보도록 하겠다.

### 필수 라이브러리 및 랜덤 시드 고정
```
import random
import os

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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
my_seed = 400
seed_everything(my_seed) # Seed 고정

```
필수 라이브러리의 경우, 랜덤시드를 400으로 고정해서 모델이 학습 시마다 동일한 값을 출력하기 위해 Base 코드에다 random과 os를 import해주었다.

### 데이터 전처리 

```
df_train = pd.read_csv("train.csv") # 학습용 데이터
df_test = pd.read_csv("submission.csv") # 테스트 데이터(제출파일의 데이터)

def label_encoding(series: pd.Series) -> pd.Series:
    """범주형 데이터를 시리즈 형태로 받아 숫자형 데이터로 변환합니다."""

    my_dict = {}

    # 모든 요소를 문자열로 변환
    series = series.astype(str)

    for idx, value in enumerate(sorted(series.unique())):
        my_dict[value] = idx
    series = series.map(my_dict)

    return series

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

for col in label_columns:  
    df_train[col] = df_all.iloc[: len(df_train)][col]
    df_test[col] = df_all.iloc[len(df_train) :][col]

x_train, x_val, y_train, y_val = train_test_split(
    df_train.drop("is_converted", axis=1),
    df_train["is_converted"],
    test_size=0.2,
    shuffle=True,
    random_state=400,
)
```
Base 코드와 동일하게 데이터 셋들을 읽어온 후에 레이블을 인코딩하고, 학습 데이터를 학습 데이터와 검증 데이터로 분류해주었다.

다음으로 **결측치**를 처리해보겠다.

```
# train

train_nan = x_train.copy()

zero_cols = ['com_reg_ver_win_rate',
             'historical_existing_cnt',
             'id_strategic_ver',
             'it_strategic_ver',
             'idit_strategic_ver']

for col in zero_cols:
    train_nan[col] = train_nan[col].fillna(0)

mean_cols = ['ver_win_rate_x','ver_win_ratio_per_bu']
for col in mean_cols:
    train_nan[col] = train_nan[col].fillna(train_nan[col].mean())
    
x_train = train_nan

# validation

val_nan = x_val.copy()

val_zero_cols = ['com_reg_ver_win_rate',
                 'historical_existing_cnt',
                 'id_strategic_ver',
                 'it_strategic_ver',
                 'idit_strategic_ver']

for col in val_zero_cols:
    val_nan[col] = val_nan[col].fillna(0)

val_mean_cols = ['ver_win_rate_x','ver_win_ratio_per_bu']
for col in val_mean_cols:
    val_nan[col] = val_nan[col].fillna(train_nan[col].mean())

x_val = val_nan

# test

test_nan = df_test.copy()

test_zero_cols = ['com_reg_ver_win_rate',
                  'historical_existing_cnt',
                  'id_strategic_ver',
                  'it_strategic_ver',
                  'idit_strategic_ver']

for col in test_zero_cols:
    test_nan[col] = test_nan[col].fillna(0)
    
test_mean_cols = ['ver_win_rate_x','ver_win_ratio_per_bu']
for col in test_mean_cols:
    test_nan[col] = test_nan[col].fillna(train_nan[col].mean())

df_test = test_nan
```
Base 코드의 경우, 결측치 값들을 fillna(0)를 통해 전부 0으로 처리해주었지만, 성능의 개선을 위해 Data Set의 Feature 설명을 보며 결측치를 Feature 별로 논리적으로 채워주었다. 따라서 위의 코드와 같이 **'com_reg_ver_win_rate','historical_existing_cnt','id_strategic_ver','it_strategic_ver','idit_strategic_ver'** Column들은 0으로, **'ver_win_rate_x','ver_win_ratio_per_bu'** Column들은 평균값으로 채워 주었다.(하지만 평균값으로 채워준 Column들은 삭제하기에 Base코드와 동일하게 결측치 처리한 것이긴 하다.)

또한 validation set과 test set에서 평균값을 사용할 때는 **데이터 유출의 우려가 있으니 훈련 데이터에서 계산한 평균값을 사용해야함**에 유의하자.
 
이외에도 fillna(method = 'pad')를 통해 **결측치 바로 이전 값**으로 채우거나, fillna(method = 'bfill')을 통해 **결측치 바로 이후 값**으로 채우거나, 혹은 **최빈값**으로 결측치를 채워주는 방법들이 있다.


```
x_train = x_train.drop(['ver_win_rate_x','ver_win_ratio_per_bu'],axis=1)
x_val = x_val.drop(['ver_win_rate_x','ver_win_ratio_per_bu'],axis=1)
df_test = df_test.drop(['ver_win_rate_x','ver_win_ratio_per_bu'],axis=1)
```

다음으로는 위의 코드와 같이 모델의 일반화 성능을 높여 오버피팅을 방지하기 위해 앞서 찾아낸 상관계수가 높고 중요도가 낮은 Feature들을 삭제해주었다.

데이터 전처리에는 결측치 처리, Feature 삭제 외에도, 이상치 제거 혹은 정규화나 표준화와 같은 Scaling 작업 등이 있다. 하지만 의사결정나무나 랜덤 포레스트와 같은 트리 기반 모델은 결정 구조가 데이터의 특정 범위나 이상치에 영향을 덜 받고, 데이터를 분할할 때 스케일의 영향을 받지 않으므로 이상치 제거, 정규화, 표준화 등의 과정이 필요없기에 제외했다

### 모델 학습 및 성능보기


```
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

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

pred = model.predict(x_val.fillna(0))
get_clf_eval(y_val, pred)
```

모델 학습과 성능은 Base코드와 동일하게 진행했다. Hyperparameter를 조절하거나 모델을 바꾸지 않은 이유는 뒤에서 기술하도록 하겠다.

결과적으로 아래 그림에서도 볼 수 있듯이 Base 코드에 비해 F1 score가 약간 개선된 것을 확인할 수 있다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/256e517d-4b21-4390-9f99-049fd43ee476"></p>  

마지막으로 제출 방법 또한 Base 코드와 동일하니 이는 생략하도록 하겠다.

이와 같은 방법으로 improved한 코드를 제출한 결과 Public Score가 0.5202 점이 나왔다. 

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/9a286d8d-5f3f-4437-b71c-d78d3def16f4" ></p>  

애초에 다음학기에는 현장실습이 예정되어 있어 이번 해커톤의 목표는 수료였다. 따라서 도메인 지식을 활용해 새로운 Feature를 만드는 등의 방법으로 모델의 성능을 더 개선시키기 보다는 현장실습에서 원활한 연구를 위해 앞으로는 ROS와 Mobile Manipulator에 대해 공부할 예정이다. 하지만 이따금씩 해당 예측 모델의 성능 개선에 있어서 좋은 아이디어가 생각난다면 간간히 테스트해볼 예정이다.  

## 시도는 해보았지만 사용하지 않은 방법

### 이상치 제거, Data Scaling

데이터 전처리 과정에서는 이상치 제거나 정규화나 표준화 등의 Data Scailing 작업을 해보았는데, 앞서 언급했듯이 트리 기반의 모델에서는 별 효과가 없어 사용하지 않았다. 하지만 Tree 기반의 모델 말고 다른 모델에서는 위와 같은 작업을 꼭 해주어야 한다.

<p align="center"><img src="https://github.com/jebeom/jebeom.github.io/assets/107978090/fe6931f5-2e6c-46e2-b724-bf2ac7a7e841"></p>  

정규화와 표준화의 차이는 위 그림과 같으며, 추후 Data Scaling 작업을 해야할 때는 [여기](https://resultofeffort.tistory.com/10)를 참고해서 진행해보자.	


### 모델 변화

의사결정나무의 경우 과적합이 발생할 수 있다는 단점이 있어 DecisionTreeClassifier 대신 앙상블 모델인 RandomForestClassifier 모델을 사용해보았는데, 이상하게도 Public Score가 눈에 띄게 떨어졌다. 이는 랜덤포레스트 모델을 사용했을 때, 의사결정나무 보다 더 과적합이 일어난 것임을 알 수 있다. 이러한 일이 발생한 이유가 무엇 때문인지 정확히는 모르겠다만, 현재로써는 해커톤에서의 Task가 의사결정나무에 더 적합했기 때문이라고 생각 중에 있다.

시도해보지는 않았지만 랜덤포레스트 외에도 GBM(Gradient Boosting Machine)와 GBM을 기반으로 한 XGBoost, LGBM, CATBOOST 등의 모델들이 있는데 추후에 머신러닝과 관련된 해커톤을 진행한다면 해당 모델들도 활용해봐야겠다.
 
### Hyperparameter 조절

```
params = {'max_depth': range(20, 40, 1),
          'min_samples_leaf' : range(1,15,1),
          'min_samples_split': range(2, 15, 1),
          }

# min_impurity_decrease : 최소 불순도
# min_impurity_split : 나무 성장을 멈추기 위한 임계치
gs = GridSearchCV(DecisionTreeClassifier(random_state=41), params, scoring = 'f1', cv=5, n_jobs=-1)
gs.fit(x_train, y_train)

print(gs.best_params_)

dt = gs.best_estimator_ # best_estimator_ : 검증점수가 가장 높은 모델이 저장됨
print(gs.best_score_)  # f1점수 계산
print(dt)              # val 정확도 계산
print(dt.score(x_val, y_val))
```

사이킷런에서 제공하는 GridSearchCV를 이용하면 하이퍼파라미터 탐색과 교차 검증을 한 번에 수행할 수 있다. 여기서 GridSearchCV 에 들어가는 인수 중 cv에 대입하는 값은 K-폴드 교차 검증에서 K의 값과 동일하다.

위와 같은 코드에서 GridSearchCV을 이용해 더욱 일반화된 성능을 가진 모델의 하이퍼파라미터를 찾을 수 있을 것이라고 기대했는데 GridSearchCV 방법을 통해 찾아낸 매개변수를 입력하고 제출했더니 오히려 Public Score가 떨어졌다. 이유가 무었인지 이 역시 정확히 모르겠지만 아마 test data set에는 train set에는 없는 특성이 있어서 그런 것이라고 생각된다.

참고로 의사결정나무에서 주로 조절하는 Hyperparameter들은 다음과 같다.


- **max_depth** : 결정 트리의 최대 깊이를 나타내며 default는 None값으로 완벽히 클래스 결정값이 될때 까지 깊이를 계속 키우거나 노드가 가지는 데이터 개수가 min_samples_split보다 작아질 때 까지 계속 분할한다. **해당 값을 감소시킬수록 깊이가 얕아지므로 모델이 덜 복잡해져** 과대적합이 일어나지 않을 가능성이 높아진다.


- **min_samples_split** : 내부 노드를 분할하기 위해 필요한 최소 샘플 수로 과적합 제어에 사용한다.만약 이 값이 10이면 노드가 분할하기 위해서는 해당 노드에 최소 10개의 샘플이 있어야 한다는 것을 의미한다. default값은 2이며, **해당 값을 키울수록 모델이 덜 복잡해진다.**

- **min_samples_leaf** : 리프(나무의 말단) 노드가 가지고 있어야 하는 최소한의 샘플 수로 과적합 제어에 사용한다.만약 이 값이 5이고, 어떤 분할이 5개 미만의 샘플을 가진 리프 노트를 만든다면 분할은 수행되지 않는다. default값은 1이며, min_samples_split과 동일하게 **해당 값을 키울수록 모델이 덜 복잡해진다.**


- **max_leaf_nodes**  : 리프 노드의 최대 개수로 default값은 None이며, **해당 값을 감소 시킬수록 모델이 덜 복잡해진다.**

- **max_features** : 각 노드에서 분할에 사용할 특성의 최대 개수로 default값은 None으로 데이터 세트의 모든 feature을 사용하여 분할한다. **해당 값을 감소시킬수록 모델이 덜 복잡해진다.** 하지만 너무 낮게 설정하면, 알고리즘이 특성의 일부만을 고려하여 정보를 놓칠 수 있으니 주의하자.

- **min_impurity_decrease** : 노드를 분할하는 결정을 내릴 때 사용되는 값으로 불순도 감소가 이 값 이상일 때만 노드를 분할하도록 제한한다. default는 0이며, 예를 들어, 'min_impurity_decrease'를 0.01로 설정하면, 노드를 분할하여 얻을 수 있는 불순도 감소가 0.01 이상일 때만 분할이 이루어진다. 이 경우, 불순도 감소가 0.01 미만인 분할은 수행되지 않아 트리의 크기가 작아질 수 있다.



## 간단한 후기

이번에 진행한 해커톤은 그동안 머신러닝과 관련해서 이론적인 내용들만 학습하다, 처음으로 실제 머신러닝 라이브러리(sklearn)를 사용해 예측 모델을 만들어보았다. sklearn 라이브러리를 사용해보지 않아 어떤 내장 함수가 있는지, 데이터 분석을 어떻게 하는지에 대해 공부 하느라 꽤나 힘들었지만, 비교적 빠른 시간 내에 목표를 달성한 것 같아 기쁘다.

이론적인 내용만 배웠을 땐 체감이 잘 안되었지만, 실제로 모델의 성능 향상을 위해 데이터 전처리부터 모델 학습 및 검증까지의 전 과정을 구성해봄으로써 머신러닝의 기본적인 Process가 어떤 식으로 진행되고, 어느 부분에서 성능 향상의 가능성이 있는지에 대해 깨달을 수 있었던 좋은 기회였던 것 같다.



