```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

# Pycaret을 사용한 모델링! 
 - 이 노트의 목적
그 전까지는 사이킷런으로 lightGBM, Catboost 모델링 한 것을 kaggle에 summit 했었다.
하지만 pycaret을 사용하여 lightGBM과 Catboost를 자동으로 하이퍼 파라미터 튜닝 하여 기존의 점수와 비교해보려 한다.

-결과 
*사이킷런으로 스스로 하이퍼파라미터 튜닝한 것 보다, pycaret tune_model로 하이퍼 파라미터 튜닝한 것이 kaggle 점수가 높게 나왔다. *




```python
!pip install scikit-learn==0.23.2
```

    Collecting scikit-learn==0.23.2
      Downloading scikit_learn-0.23.2-cp37-cp37m-manylinux1_x86_64.whl (6.8 MB)
    [K     |████████████████████████████████| 6.8 MB 5.4 MB/s 
    [?25hRequirement already satisfied: scipy>=0.19.1 in /opt/conda/lib/python3.7/site-packages (from scikit-learn==0.23.2) (1.5.4)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from scikit-learn==0.23.2) (2.1.0)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/lib/python3.7/site-packages (from scikit-learn==0.23.2) (1.0.1)
    Requirement already satisfied: numpy>=1.13.3 in /opt/conda/lib/python3.7/site-packages (from scikit-learn==0.23.2) (1.19.5)
    Installing collected packages: scikit-learn
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 0.24.1
        Uninstalling scikit-learn-0.24.1:
          Successfully uninstalled scikit-learn-0.24.1
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    pyldavis 3.3.1 requires numpy>=1.20.0, but you have numpy 1.19.5 which is incompatible.
    pdpbox 0.2.1 requires matplotlib==3.1.1, but you have matplotlib 3.4.0 which is incompatible.
    imbalanced-learn 0.8.0 requires scikit-learn>=0.24, but you have scikit-learn 0.23.2 which is incompatible.[0m
    Successfully installed scikit-learn-0.23.2
    

https://www.kaggle.com/udbhavpangotra/tps-apr21-eda-model


https://www.kaggle.com/hiro5299834/tps-apr-2021-voting-pseudo-labeling

# KAGGLE 스터디 


```python
import pandas as pd
import numpy as np
import random
import os

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import lightgbm as lgb
import catboost as ctb
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')
```


<style type='text/css'>
.datatable table.frame { margin-bottom: 0; }
.datatable table.frame thead { border-bottom: none; }
.datatable table.frame tr.coltypes td {  color: #FFFFFF;  line-height: 6px;  padding: 0 0.5em;}
.datatable .bool    { background: #DDDD99; }
.datatable .object  { background: #565656; }
.datatable .int     { background: #5D9E5D; }
.datatable .float   { background: #4040CC; }
.datatable .str     { background: #CC4040; }
.datatable .row_index {  background: var(--jp-border-color3);  border-right: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  font-size: 9px;}
.datatable .frame tr.coltypes .row_index {  background: var(--jp-border-color0);}
.datatable th:nth-child(2) { padding-left: 12px; }
.datatable .hellipsis {  color: var(--jp-cell-editor-border-color);}
.datatable .vellipsis {  background: var(--jp-layout-color0);  color: var(--jp-cell-editor-border-color);}
.datatable .na {  color: var(--jp-cell-editor-border-color);  font-size: 80%;}
.datatable .footer { font-size: 9px; }
.datatable .frame_dimensions {  background: var(--jp-border-color3);  border-top: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  display: inline-block;  opacity: 0.6;  padding: 1px 10px 1px 5px;}
</style>




```python
TARGET = 'Survived'

N_ESTIMATORS = 1000
N_SPLITS = 10
SEED = 2021
EARLY_STOPPING_ROUNDS = 100
VERBOSE = 100
```


```python
# #랜덤 시드 생성
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
set_seed(SEED)
```

## 데이터 전처리

### lode data


```python
train_df = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
submission = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')
#test_df['Survived'] = pd.read_csv("../input/submission-merged3/submission_merged3.csv")['Survived']

all_df = pd.concat([train_df, test_df]).reset_index(drop=True)
#reset_index : 인덱스를 세팅한다. drop=True를 하면 인덱스를 세팅한걸 삭제함. 

```


```python
print('Rows and Columns in train dataset:', train_df.shape)
print('Rows and Columns in test dataset:', test_df.shape)
```

    Rows and Columns in train dataset: (100000, 12)
    Rows and Columns in test dataset: (100000, 11)
    

### 결측치 갯수 출력


```python
print('Missing values per columns in train dataset')
for col in train_df.columns:
    temp_col = train_df[col].isnull().sum()
    print(f'{col}: {temp_col}')
print()
print('Missing values per columns in test dataset')
for col in test_df.columns:
    temp_col = test_df[col].isnull().sum()
    print(f'{col}: {temp_col}')
```

    Missing values per columns in train dataset
    PassengerId: 0
    Survived: 0
    Pclass: 0
    Name: 0
    Sex: 0
    Age: 3292
    SibSp: 0
    Parch: 0
    Ticket: 4623
    Fare: 134
    Cabin: 67866
    Embarked: 250
    
    Missing values per columns in test dataset
    PassengerId: 0
    Pclass: 0
    Name: 0
    Sex: 0
    Age: 3487
    SibSp: 0
    Parch: 0
    Ticket: 5181
    Fare: 133
    Cabin: 70831
    Embarked: 277
    

### Filling missing values


```python
#나이는 나이의 평균치로 채운다.
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].mean())

#cabin은 문자열을 분할하고, 제일 첫번째 글자를 따와서 넣는다. 결측치엔 X를 넣는다.
#strip() : 양쪽 공백을 지운다. 여기서느 x[0]외엔 다 지우는듯. 
all_df['Cabin'] = all_df['Cabin'].fillna('X').map(lambda x: x[0].strip())


#print(all_df['Ticket'].head(10))
#Ticket, fillna with 'X', split string and take first split 
#split() : 문자열 나누기. 디폴트는 ' '이고, 문자를 가진 데이터들이 전부 띄워쓰기로 구분되어있기때문에 가능. 
all_df['Ticket'] = all_df['Ticket'].fillna('X').map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')

#pclass에 따른 Fare의 평균을 구해서 dictionary형태로 만든다. 
fare_map = all_df[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
#fare의 결측치에 본인 행의 pclass 값을 넣고, 그 값을 fare 평균에 맵핑시킨다.  
all_df['Fare'] = all_df['Fare'].fillna(all_df['Pclass'].map(fare_map['Fare']))
#유독 높은 가격이나 낮은 가격이 있기때문에, 이상치의 영향을 줄이기 위해서 Fare에 log를 취해준다.
all_df['Fare'] = np.log1p(all_df['Fare'])


#항구의 결측치를 X로 채운다. 
all_df['Embarked'] = all_df['Embarked'].fillna('X')

#이름은 성만 사용한다.
all_df['Name'] = all_df['Name'].map(lambda x: x.split(',')[0])

```


```python
data_1=all_df.loc[all_df['Pclass']==1].groupby('Ticket')['Ticket'].count().sort_values(ascending=False)
print(data_1)
print()
data_2=all_df.loc[all_df['Pclass']==2].groupby('Ticket')['Ticket'].count().sort_values(ascending=False)
print(data_2)
print()
data_3=all_df.loc[all_df['Pclass']==3].groupby('Ticket')['Ticket'].count().sort_values(ascending=False)
print(data_3)
print()
```

    Ticket
    X             36336
    PC            16814
    C.A.            338
    SC/Paris        334
    SC/PARIS        260
    W./C.           206
    S.O.C.          192
    S.C./PARIS      191
    PP              186
    F.C.            183
    SC/AH           178
    F.C.C.          167
    STON/O          163
    CA.             161
    SOTON/O.Q.      123
    A/4             115
    A/5.            108
    W.E.P.           94
    WE/P             92
    SOTON/OQ         87
    CA               81
    STON/O2.         81
    A/5              70
    C                67
    A/4.             66
    P/PP             66
    SC               59
    SOTON/O2         48
    A./5.            46
    S.O./P.P.        40
    A.5.             33
    AQ/4             27
    A/S              23
    SCO/W            19
    S.P.             17
    SC/A4            16
    SW/PP            16
    SC/A.3           15
    S.O.P.           15
    C.A./SOTON       14
    A.               14
    SO/C             14
    S.C./A.4.        14
    STON/OQ.         13
    W/C              13
    LP               11
    S.W./PP          11
    AQ/3.             8
    Fa                7
    A4.               6
    Name: Ticket, dtype: int64
    
    Ticket
    X             31337
    A.              997
    C.A.            717
    SC/PARIS        470
    STON/O          387
    PC              330
    S.O.C.          313
    PP              308
    SC/AH           284
    W./C.           259
    SOTON/O.Q.      219
    F.C.C.          203
    A/5.            200
    A/4             152
    SC/Paris        135
    S.C./PARIS      119
    SOTON/O2        112
    CA.             107
    STON/O2.        106
    C               104
    F.C.            100
    WE/P             92
    SOTON/OQ         86
    A/5              82
    CA               66
    W.E.P.           60
    A./5.            60
    S.O./P.P.        54
    P/PP             50
    A/4.             46
    SCO/W            36
    SC               33
    A.5.             29
    AQ/4             29
    LP               25
    SC/A.3           20
    C.A./SOTON       19
    A/S              19
    SC/A4            17
    Fa               15
    S.W./PP          13
    SO/C             13
    S.C./A.4.        13
    STON/OQ.         12
    W/C              11
    S.P.             10
    SW/PP             9
    S.O.P.            9
    A4.               7
    AQ/3.             6
    Name: Ticket, dtype: int64
    
    Ticket
    X             84781
    A.             6420
    C.A.           2615
    STON/O         1508
    A/5.            918
    SOTON/O.Q.      719
    PP              679
    SC/PARIS        642
    W./C.           623
    PC              595
    F.C.C.          541
    A/5             420
    CA.             368
    STON/O2.        363
    SC/AH           331
    A/4             268
    SOTON/O2        264
    S.O.C.          231
    C               227
    SC/Paris        177
    S.O./P.P.       177
    SOTON/OQ        172
    CA              172
    W.E.P.          154
    F.C.            131
    S.C./PARIS      127
    A./5.           122
    WE/P            121
    SC              106
    A/4.            104
    SCO/W            74
    A.5.             72
    P/PP             68
    SC/A4            67
    AQ/4             56
    LP               41
    Fa               37
    STON/OQ.         37
    S.W./PP          32
    SC/A.3           31
    C.A./SOTON       31
    SW/PP            30
    A/S              28
    SO/C             28
    AQ/3.            26
    S.P.             24
    S.C./A.4.        23
    S.O.P.           21
    W/C              20
    A4.              20
    Name: Ticket, dtype: int64
    
    

## 인코딩 

변수별로 인코딩을 다르게 해준다. 


```python
label_cols = ['Name', 'Ticket', 'Sex','Pclass','Embarked']
onehot_cols = [ 'Cabin',]
numerical_cols = [ 'Age', 'SibSp', 'Parch', 'Fare']
```


```python
#라벨 인코딩 함수. c라는 매개변수를 받아서 맞게 트렌스폼 해준다. 
def label_encoder(c):
    le = LabelEncoder()
    return le.fit_transform(c)
```


```python

#StandardScaler(): 평균을 제거하고 데이터를 단위 분산으로 조정한다. 
#그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 되는 함수
scaler = StandardScaler()

onehot_encoded_df = pd.get_dummies(all_df[onehot_cols])
label_encoded_df = all_df[label_cols].apply(label_encoder)
numerical_df = pd.DataFrame(scaler.fit_transform(all_df[numerical_cols]), columns=numerical_cols)
target_df = all_df[TARGET]

all_df = pd.concat([numerical_df, label_encoded_df,onehot_encoded_df, target_df], axis=1)
#all_df = pd.concat([numerical_df, label_encoded_df, target_df], axis=1)
```

## 모델링


```python
drop_list=['Survived','Parch']
```

## not pseudo


```python
train = all_df.iloc[:100000, :]#0개~100000개
test = all_df.iloc[100000:, :] #100000개~ 
#iloc은 정수형 인덱싱
test = test.drop('Survived', axis=1) #test에서 종속변수를 드랍한다. 
model_results = pd.DataFrame()
folds = 5
```


```python

```


```python
y= train.loc[:,'Survived']
X= train.drop(drop_list,axis=1)
```

## pycarot


```python
caret_train=train.drop('Parch',axis=1)
```


```python


!pip install pycaret==2.2.3
```

    Collecting pycaret==2.2.3
      Downloading pycaret-2.2.3-py3-none-any.whl (249 kB)
    [K     |████████████████████████████████| 249 kB 867 kB/s 
    [?25hRequirement already satisfied: scikit-plot in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (0.3.7)
    Requirement already satisfied: xgboost>=1.1.0 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (1.3.3)
    Requirement already satisfied: scikit-learn==0.23.2 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (0.23.2)
    Requirement already satisfied: textblob in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (0.15.3)
    Requirement already satisfied: wordcloud in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (1.8.1)
    Requirement already satisfied: pyLDAvis in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (3.3.1)
    Requirement already satisfied: joblib in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (1.0.1)
    Requirement already satisfied: numpy>=1.17 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (1.19.5)
    Requirement already satisfied: plotly>=4.4.1 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (4.14.3)
    Requirement already satisfied: mlxtend in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (0.18.0)
    Requirement already satisfied: IPython in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (7.20.0)
    Requirement already satisfied: gensim in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (4.0.0)
    Requirement already satisfied: ipywidgets in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (7.6.3)
    Requirement already satisfied: imbalanced-learn>=0.7.0 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (0.8.0)
    Requirement already satisfied: nltk in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (3.2.4)
    Requirement already satisfied: catboost>=0.23.2 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (0.25)
    Requirement already satisfied: pandas-profiling>=2.8.0 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (2.8.0)
    Requirement already satisfied: cufflinks>=0.17.0 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (0.17.3)
    Requirement already satisfied: matplotlib in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (3.4.0)
    Requirement already satisfied: umap-learn in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (0.5.1)
    Collecting pyod
      Downloading pyod-0.8.8.tar.gz (102 kB)
    [K     |████████████████████████████████| 102 kB 4.0 MB/s 
    [?25hRequirement already satisfied: kmodes>=0.10.1 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (0.11.0)
    Collecting mlflow
      Downloading mlflow-1.16.0-py3-none-any.whl (14.2 MB)
    [K     |████████████████████████████████| 14.2 MB 7.2 MB/s 
    [?25hRequirement already satisfied: seaborn in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (0.11.1)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (1.2.2)
    Requirement already satisfied: yellowbrick>=1.0.1 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (1.3.post1)
    Requirement already satisfied: spacy in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (2.3.5)
    Requirement already satisfied: lightgbm>=2.3.1 in /opt/conda/lib/python3.7/site-packages (from pycaret==2.2.3) (3.1.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from scikit-learn==0.23.2->pycaret==2.2.3) (2.1.0)
    Requirement already satisfied: scipy>=0.19.1 in /opt/conda/lib/python3.7/site-packages (from scikit-learn==0.23.2->pycaret==2.2.3) (1.5.4)
    Requirement already satisfied: graphviz in /opt/conda/lib/python3.7/site-packages (from catboost>=0.23.2->pycaret==2.2.3) (0.8.4)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from catboost>=0.23.2->pycaret==2.2.3) (1.15.0)
    Requirement already satisfied: setuptools>=34.4.1 in /opt/conda/lib/python3.7/site-packages (from cufflinks>=0.17.0->pycaret==2.2.3) (49.6.0.post20210108)
    Requirement already satisfied: colorlover>=0.2.1 in /opt/conda/lib/python3.7/site-packages (from cufflinks>=0.17.0->pycaret==2.2.3) (0.3.0)
    Collecting imbalanced-learn>=0.7.0
      Downloading imbalanced_learn-0.7.0-py3-none-any.whl (167 kB)
    [K     |████████████████████████████████| 167 kB 14.9 MB/s 
    [?25hRequirement already satisfied: pickleshare in /opt/conda/lib/python3.7/site-packages (from IPython->pycaret==2.2.3) (0.7.5)
    Requirement already satisfied: pygments in /opt/conda/lib/python3.7/site-packages (from IPython->pycaret==2.2.3) (2.8.0)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.7/site-packages (from IPython->pycaret==2.2.3) (4.4.2)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from IPython->pycaret==2.2.3) (3.0.16)
    Requirement already satisfied: backcall in /opt/conda/lib/python3.7/site-packages (from IPython->pycaret==2.2.3) (0.2.0)
    Requirement already satisfied: traitlets>=4.2 in /opt/conda/lib/python3.7/site-packages (from IPython->pycaret==2.2.3) (5.0.5)
    Requirement already satisfied: pexpect>4.3 in /opt/conda/lib/python3.7/site-packages (from IPython->pycaret==2.2.3) (4.8.0)
    Requirement already satisfied: jedi>=0.16 in /opt/conda/lib/python3.7/site-packages (from IPython->pycaret==2.2.3) (0.17.2)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /opt/conda/lib/python3.7/site-packages (from ipywidgets->pycaret==2.2.3) (1.0.0)
    Requirement already satisfied: widgetsnbextension~=3.5.0 in /opt/conda/lib/python3.7/site-packages (from ipywidgets->pycaret==2.2.3) (3.5.1)
    Requirement already satisfied: ipykernel>=4.5.1 in /opt/conda/lib/python3.7/site-packages (from ipywidgets->pycaret==2.2.3) (5.1.1)
    Requirement already satisfied: nbformat>=4.2.0 in /opt/conda/lib/python3.7/site-packages (from ipywidgets->pycaret==2.2.3) (5.1.2)
    Requirement already satisfied: jupyter-client in /opt/conda/lib/python3.7/site-packages (from ipykernel>=4.5.1->ipywidgets->pycaret==2.2.3) (6.1.11)
    Requirement already satisfied: tornado>=4.2 in /opt/conda/lib/python3.7/site-packages (from ipykernel>=4.5.1->ipywidgets->pycaret==2.2.3) (5.0.2)
    Requirement already satisfied: parso<0.8.0,>=0.7.0 in /opt/conda/lib/python3.7/site-packages (from jedi>=0.16->IPython->pycaret==2.2.3) (0.7.1)
    Requirement already satisfied: wheel in /opt/conda/lib/python3.7/site-packages (from lightgbm>=2.3.1->pycaret==2.2.3) (0.36.2)
    Requirement already satisfied: ipython-genutils in /opt/conda/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets->pycaret==2.2.3) (0.2.0)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /opt/conda/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets->pycaret==2.2.3) (3.2.0)
    Requirement already satisfied: jupyter-core in /opt/conda/lib/python3.7/site-packages (from nbformat>=4.2.0->ipywidgets->pycaret==2.2.3) (4.7.1)
    Requirement already satisfied: pyrsistent>=0.14.0 in /opt/conda/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets->pycaret==2.2.3) (0.17.3)
    Requirement already satisfied: importlib-metadata in /opt/conda/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets->pycaret==2.2.3) (3.4.0)
    Requirement already satisfied: attrs>=17.4.0 in /opt/conda/lib/python3.7/site-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets->pycaret==2.2.3) (20.3.0)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/lib/python3.7/site-packages (from pandas->pycaret==2.2.3) (2.8.1)
    Requirement already satisfied: pytz>=2017.3 in /opt/conda/lib/python3.7/site-packages (from pandas->pycaret==2.2.3) (2021.1)
    Requirement already satisfied: visions[type_image_path]==0.4.4 in /opt/conda/lib/python3.7/site-packages (from pandas-profiling>=2.8.0->pycaret==2.2.3) (0.4.4)
    Requirement already satisfied: astropy>=4.0 in /opt/conda/lib/python3.7/site-packages (from pandas-profiling>=2.8.0->pycaret==2.2.3) (4.2)
    Requirement already satisfied: tqdm>=4.43.0 in /opt/conda/lib/python3.7/site-packages (from pandas-profiling>=2.8.0->pycaret==2.2.3) (4.56.2)
    Requirement already satisfied: confuse>=1.0.0 in /opt/conda/lib/python3.7/site-packages (from pandas-profiling>=2.8.0->pycaret==2.2.3) (1.4.0)
    Requirement already satisfied: htmlmin>=0.1.12 in /opt/conda/lib/python3.7/site-packages (from pandas-profiling>=2.8.0->pycaret==2.2.3) (0.1.12)
    Requirement already satisfied: jinja2>=2.11.1 in /opt/conda/lib/python3.7/site-packages (from pandas-profiling>=2.8.0->pycaret==2.2.3) (2.11.3)
    Requirement already satisfied: missingno>=0.4.2 in /opt/conda/lib/python3.7/site-packages (from pandas-profiling>=2.8.0->pycaret==2.2.3) (0.4.2)
    Requirement already satisfied: requests>=2.23.0 in /opt/conda/lib/python3.7/site-packages (from pandas-profiling>=2.8.0->pycaret==2.2.3) (2.25.1)
    Requirement already satisfied: phik>=0.9.10 in /opt/conda/lib/python3.7/site-packages (from pandas-profiling>=2.8.0->pycaret==2.2.3) (0.10.0)
    Requirement already satisfied: tangled-up-in-unicode>=0.0.6 in /opt/conda/lib/python3.7/site-packages (from pandas-profiling>=2.8.0->pycaret==2.2.3) (0.0.6)
    Requirement already satisfied: networkx>=2.4 in /opt/conda/lib/python3.7/site-packages (from visions[type_image_path]==0.4.4->pandas-profiling>=2.8.0->pycaret==2.2.3) (2.5)
    Requirement already satisfied: imagehash in /opt/conda/lib/python3.7/site-packages (from visions[type_image_path]==0.4.4->pandas-profiling>=2.8.0->pycaret==2.2.3) (4.2.0)
    Requirement already satisfied: Pillow in /opt/conda/lib/python3.7/site-packages (from visions[type_image_path]==0.4.4->pandas-profiling>=2.8.0->pycaret==2.2.3) (7.2.0)
    Requirement already satisfied: pyerfa in /opt/conda/lib/python3.7/site-packages (from astropy>=4.0->pandas-profiling>=2.8.0->pycaret==2.2.3) (1.7.1.1)
    Requirement already satisfied: pyyaml in /opt/conda/lib/python3.7/site-packages (from confuse>=1.0.0->pandas-profiling>=2.8.0->pycaret==2.2.3) (5.3.1)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/lib/python3.7/site-packages (from jinja2>=2.11.1->pandas-profiling>=2.8.0->pycaret==2.2.3) (1.1.1)
    Requirement already satisfied: pyparsing>=2.2.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pycaret==2.2.3) (2.4.7)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pycaret==2.2.3) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib->pycaret==2.2.3) (1.3.1)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/conda/lib/python3.7/site-packages (from pexpect>4.3->IPython->pycaret==2.2.3) (0.7.0)
    Requirement already satisfied: numba>=0.38.1 in /opt/conda/lib/python3.7/site-packages (from phik>=0.9.10->pandas-profiling>=2.8.0->pycaret==2.2.3) (0.52.0)
    Requirement already satisfied: llvmlite<0.36,>=0.35.0 in /opt/conda/lib/python3.7/site-packages (from numba>=0.38.1->phik>=0.9.10->pandas-profiling>=2.8.0->pycaret==2.2.3) (0.35.0)
    Requirement already satisfied: retrying>=1.3.3 in /opt/conda/lib/python3.7/site-packages (from plotly>=4.4.1->pycaret==2.2.3) (1.3.3)
    Requirement already satisfied: wcwidth in /opt/conda/lib/python3.7/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->IPython->pycaret==2.2.3) (0.2.5)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.7/site-packages (from requests>=2.23.0->pandas-profiling>=2.8.0->pycaret==2.2.3) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.7/site-packages (from requests>=2.23.0->pandas-profiling>=2.8.0->pycaret==2.2.3) (2020.12.5)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.7/site-packages (from requests>=2.23.0->pandas-profiling>=2.8.0->pycaret==2.2.3) (1.26.3)
    Requirement already satisfied: chardet<5,>=3.0.2 in /opt/conda/lib/python3.7/site-packages (from requests>=2.23.0->pandas-profiling>=2.8.0->pycaret==2.2.3) (3.0.4)
    Requirement already satisfied: notebook>=4.4.1 in /opt/conda/lib/python3.7/site-packages (from widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (5.5.0)
    Requirement already satisfied: pyzmq>=17 in /opt/conda/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (22.0.3)
    Requirement already satisfied: terminado>=0.8.1 in /opt/conda/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (0.9.2)
    Requirement already satisfied: nbconvert in /opt/conda/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (6.0.7)
    Requirement already satisfied: Send2Trash in /opt/conda/lib/python3.7/site-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (1.5.0)
    Requirement already satisfied: smart-open>=1.8.1 in /opt/conda/lib/python3.7/site-packages (from gensim->pycaret==2.2.3) (5.0.0)
    Requirement already satisfied: PyWavelets in /opt/conda/lib/python3.7/site-packages (from imagehash->visions[type_image_path]==0.4.4->pandas-profiling>=2.8.0->pycaret==2.2.3) (1.1.1)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/lib/python3.7/site-packages (from importlib-metadata->jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets->pycaret==2.2.3) (3.4.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /opt/conda/lib/python3.7/site-packages (from importlib-metadata->jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets->pycaret==2.2.3) (3.7.4.3)
    Requirement already satisfied: sqlparse>=0.3.1 in /opt/conda/lib/python3.7/site-packages (from mlflow->pycaret==2.2.3) (0.4.1)
    Collecting databricks-cli>=0.8.7
      Downloading databricks-cli-0.14.3.tar.gz (54 kB)
    [K     |████████████████████████████████| 54 kB 1.7 MB/s 
    [?25hRequirement already satisfied: sqlalchemy in /opt/conda/lib/python3.7/site-packages (from mlflow->pycaret==2.2.3) (1.3.23)
    Requirement already satisfied: click>=7.0 in /opt/conda/lib/python3.7/site-packages (from mlflow->pycaret==2.2.3) (7.1.2)
    Requirement already satisfied: cloudpickle in /opt/conda/lib/python3.7/site-packages (from mlflow->pycaret==2.2.3) (1.6.0)
    Requirement already satisfied: entrypoints in /opt/conda/lib/python3.7/site-packages (from mlflow->pycaret==2.2.3) (0.3)
    Requirement already satisfied: gitpython>=2.1.0 in /opt/conda/lib/python3.7/site-packages (from mlflow->pycaret==2.2.3) (3.1.13)
    Collecting querystring-parser
      Downloading querystring_parser-1.2.4-py2.py3-none-any.whl (7.9 kB)
    Collecting gunicorn
      Downloading gunicorn-20.1.0.tar.gz (370 kB)
    [K     |████████████████████████████████| 370 kB 14.5 MB/s 
    [?25hRequirement already satisfied: Flask in /opt/conda/lib/python3.7/site-packages (from mlflow->pycaret==2.2.3) (1.1.2)
    Collecting alembic<=1.4.1
      Downloading alembic-1.4.1.tar.gz (1.1 MB)
    [K     |████████████████████████████████| 1.1 MB 15.3 MB/s 
    [?25hRequirement already satisfied: protobuf>=3.6.0 in /opt/conda/lib/python3.7/site-packages (from mlflow->pycaret==2.2.3) (3.15.6)
    Requirement already satisfied: docker>=4.0.0 in /opt/conda/lib/python3.7/site-packages (from mlflow->pycaret==2.2.3) (4.4.1)
    Collecting prometheus-flask-exporter
      Downloading prometheus_flask_exporter-0.18.1.tar.gz (21 kB)
    Requirement already satisfied: Mako in /opt/conda/lib/python3.7/site-packages (from alembic<=1.4.1->mlflow->pycaret==2.2.3) (1.1.4)
    Requirement already satisfied: python-editor>=0.3 in /opt/conda/lib/python3.7/site-packages (from alembic<=1.4.1->mlflow->pycaret==2.2.3) (1.0.4)
    Requirement already satisfied: tabulate>=0.7.7 in /opt/conda/lib/python3.7/site-packages (from databricks-cli>=0.8.7->mlflow->pycaret==2.2.3) (0.8.9)
    Requirement already satisfied: websocket-client>=0.32.0 in /opt/conda/lib/python3.7/site-packages (from docker>=4.0.0->mlflow->pycaret==2.2.3) (0.57.0)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/lib/python3.7/site-packages (from gitpython>=2.1.0->mlflow->pycaret==2.2.3) (4.0.5)
    Requirement already satisfied: smmap<4,>=3.0.1 in /opt/conda/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython>=2.1.0->mlflow->pycaret==2.2.3) (3.0.5)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/lib/python3.7/site-packages (from Flask->mlflow->pycaret==2.2.3) (1.0.1)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/lib/python3.7/site-packages (from Flask->mlflow->pycaret==2.2.3) (1.1.0)
    Requirement already satisfied: defusedxml in /opt/conda/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (0.6.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in /opt/conda/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (1.4.2)
    Requirement already satisfied: bleach in /opt/conda/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (3.3.0)
    Requirement already satisfied: jupyterlab-pygments in /opt/conda/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (0.1.2)
    Requirement already satisfied: mistune<2,>=0.8.1 in /opt/conda/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (0.8.4)
    Requirement already satisfied: nbclient<0.6.0,>=0.5.0 in /opt/conda/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (0.5.2)
    Requirement already satisfied: testpath in /opt/conda/lib/python3.7/site-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (0.4.4)
    Requirement already satisfied: nest-asyncio in /opt/conda/lib/python3.7/site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (1.4.3)
    Requirement already satisfied: async-generator in /opt/conda/lib/python3.7/site-packages (from nbclient<0.6.0,>=0.5.0->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (1.10)
    Requirement already satisfied: webencodings in /opt/conda/lib/python3.7/site-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (0.5.1)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.7/site-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets->pycaret==2.2.3) (20.9)
    Requirement already satisfied: prometheus_client in /opt/conda/lib/python3.7/site-packages (from prometheus-flask-exporter->mlflow->pycaret==2.2.3) (0.9.0)
    Requirement already satisfied: future in /opt/conda/lib/python3.7/site-packages (from pyLDAvis->pycaret==2.2.3) (0.18.2)
    Requirement already satisfied: funcy in /opt/conda/lib/python3.7/site-packages (from pyLDAvis->pycaret==2.2.3) (1.15)
    Requirement already satisfied: numexpr in /opt/conda/lib/python3.7/site-packages (from pyLDAvis->pycaret==2.2.3) (2.7.3)
    Requirement already satisfied: sklearn in /opt/conda/lib/python3.7/site-packages (from pyLDAvis->pycaret==2.2.3) (0.0)
    Collecting pyLDAvis
      Downloading pyLDAvis-3.3.0.tar.gz (1.7 MB)
    [K     |████████████████████████████████| 1.7 MB 18.5 MB/s 
    [?25h  Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Installing backend dependencies ... [?25ldone
    [?25h    Preparing wheel metadata ... [?25ldone
    [?25h  Downloading pyLDAvis-3.2.2.tar.gz (1.7 MB)
    [K     |████████████████████████████████| 1.7 MB 18.5 MB/s 
    [?25hRequirement already satisfied: statsmodels in /opt/conda/lib/python3.7/site-packages (from pyod->pycaret==2.2.3) (0.12.2)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/conda/lib/python3.7/site-packages (from spacy->pycaret==2.2.3) (1.0.0)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/conda/lib/python3.7/site-packages (from spacy->pycaret==2.2.3) (1.1.3)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/conda/lib/python3.7/site-packages (from spacy->pycaret==2.2.3) (2.0.5)
    Requirement already satisfied: blis<0.8.0,>=0.4.0 in /opt/conda/lib/python3.7/site-packages (from spacy->pycaret==2.2.3) (0.7.4)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/conda/lib/python3.7/site-packages (from spacy->pycaret==2.2.3) (1.0.5)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/conda/lib/python3.7/site-packages (from spacy->pycaret==2.2.3) (0.8.2)
    Requirement already satisfied: thinc<7.5.0,>=7.4.1 in /opt/conda/lib/python3.7/site-packages (from spacy->pycaret==2.2.3) (7.4.5)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/conda/lib/python3.7/site-packages (from spacy->pycaret==2.2.3) (1.0.5)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/conda/lib/python3.7/site-packages (from spacy->pycaret==2.2.3) (3.0.5)
    Requirement already satisfied: patsy>=0.5 in /opt/conda/lib/python3.7/site-packages (from statsmodels->pyod->pycaret==2.2.3) (0.5.1)
    Requirement already satisfied: pynndescent>=0.5 in /opt/conda/lib/python3.7/site-packages (from umap-learn->pycaret==2.2.3) (0.5.2)
    Building wheels for collected packages: alembic, databricks-cli, gunicorn, prometheus-flask-exporter, pyLDAvis, pyod
      Building wheel for alembic (setup.py) ... [?25ldone
    [?25h  Created wheel for alembic: filename=alembic-1.4.1-py2.py3-none-any.whl size=158155 sha256=d8f392ff192a5f199c46190c11c01ebd9ee89c41e962f8c53ca200a8f722820d
      Stored in directory: /root/.cache/pip/wheels/be/5d/0a/9e13f53f4f5dfb67cd8d245bb7cdffe12f135846f491a283e3
      Building wheel for databricks-cli (setup.py) ... [?25ldone
    [?25h  Created wheel for databricks-cli: filename=databricks_cli-0.14.3-py3-none-any.whl size=100555 sha256=9a7dcf0f0828c4d66f92d9a9aa6a9648bf13c7e106e6f2a319b7f4e0f8b869d3
      Stored in directory: /root/.cache/pip/wheels/3b/60/14/6930445b08959fbdf4e3029bac7e1f2cccb2e94df8afa00b29
      Building wheel for gunicorn (setup.py) ... [?25ldone
    [?25h  Created wheel for gunicorn: filename=gunicorn-20.1.0-py3-none-any.whl size=78917 sha256=27823621640949d8ef7678db5b24d401299123365598476d18f7734668b6a362
      Stored in directory: /root/.cache/pip/wheels/48/64/50/67e9a3524590218acb6a0c0f94038c0d60815866c52a667d57
      Building wheel for prometheus-flask-exporter (setup.py) ... [?25ldone
    [?25h  Created wheel for prometheus-flask-exporter: filename=prometheus_flask_exporter-0.18.1-py3-none-any.whl size=17158 sha256=b30adb8b36411d5ac696a62d1f72fd5f3f0a6071d76737d1901d929229e1120e
      Stored in directory: /root/.cache/pip/wheels/c4/b6/b5/e76659f3b2a3a226565e27f0a7eb7a3ac93c3f4d68acfbe617
      Building wheel for pyLDAvis (setup.py) ... [?25ldone
    [?25h  Created wheel for pyLDAvis: filename=pyLDAvis-3.2.2-py2.py3-none-any.whl size=135593 sha256=f2c6a61232597db4377bb34194186a0c15694c4ec4bf0451684c68495c1ac812
      Stored in directory: /root/.cache/pip/wheels/f8/b1/9b/560ac1931796b7303f7b517b949d2d31a4fbc512aad3b9f284
      Building wheel for pyod (setup.py) ... [?25ldone
    [?25h  Created wheel for pyod: filename=pyod-0.8.8-py3-none-any.whl size=116965 sha256=4bff0abc63f27aaf26ccb9c6f4ee9af5c336d94730dd88558cd8bc79aadc58af
      Stored in directory: /root/.cache/pip/wheels/77/59/4c/18e7ef198e2c737674b0bd8b6fa0fb1163c83ecc4e622fbda4
    Successfully built alembic databricks-cli gunicorn prometheus-flask-exporter pyLDAvis pyod
    Installing collected packages: querystring-parser, prometheus-flask-exporter, gunicorn, databricks-cli, alembic, pyod, pyLDAvis, mlflow, imbalanced-learn, pycaret
      Attempting uninstall: alembic
        Found existing installation: alembic 1.5.8
        Uninstalling alembic-1.5.8:
          Successfully uninstalled alembic-1.5.8
      Attempting uninstall: pyLDAvis
        Found existing installation: pyLDAvis 3.3.1
        Uninstalling pyLDAvis-3.3.1:
          Successfully uninstalled pyLDAvis-3.3.1
      Attempting uninstall: imbalanced-learn
        Found existing installation: imbalanced-learn 0.8.0
        Uninstalling imbalanced-learn-0.8.0:
          Successfully uninstalled imbalanced-learn-0.8.0
    Successfully installed alembic-1.4.1 databricks-cli-0.14.3 gunicorn-20.1.0 imbalanced-learn-0.7.0 mlflow-1.16.0 prometheus-flask-exporter-0.18.1 pyLDAvis-3.2.2 pycaret-2.2.3 pyod-0.8.8 querystring-parser-1.2.4
    


```python
from pycaret.utils import version
import sklearn
print("pycaret version:", version())
print("sklearn version:", sklearn.__version__)
```

    pycaret version: 2.2.3
    sklearn version: 0.23.2
    


```python
import pycaret.classification 
# from pycaret.regression import *
# reg1 = setup(data = train,target = 'Survived')
from pycaret.classification import *
```


```python
caret_train.loc[:,'Embarked'].unique()
caret_train.loc[:,'Pclass'].unique()
caret_train.loc[:,'Sex'].unique()

```




    array([1, 0])




```python
category_caret={'Sex':['0','1'],'Pclass':['0','1','2'], 'Embarked':['0','1','2','3']}
```


```python
setup(data = caret_train, 
      target = 'Survived',
      ordinal_features= category_caret,
      #numeric_imputation = 'Age','SibSp','Name','Ticket','Fare',
      fold=5,
      silent = True,
      session_id=1,
      #data_split_shuffle=True
      fold_shuffle=True

     )
```


<style  type="text/css" >
#T_4e1bb_row0_col1,#T_4e1bb_row8_col1{
            background-color:  lightgreen;
        }</style><table id="T_4e1bb_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Description</th>        <th class="col_heading level0 col1" >Value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_4e1bb_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_4e1bb_row0_col0" class="data row0 col0" >session_id</td>
                        <td id="T_4e1bb_row0_col1" class="data row0 col1" >1</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_4e1bb_row1_col0" class="data row1 col0" >Target</td>
                        <td id="T_4e1bb_row1_col1" class="data row1 col1" >Survived</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_4e1bb_row2_col0" class="data row2 col0" >Target Type</td>
                        <td id="T_4e1bb_row2_col1" class="data row2 col1" >Binary</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_4e1bb_row3_col0" class="data row3 col0" >Label Encoded</td>
                        <td id="T_4e1bb_row3_col1" class="data row3 col1" >0.0: 0, 1.0: 1</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_4e1bb_row4_col0" class="data row4 col0" >Original Data</td>
                        <td id="T_4e1bb_row4_col1" class="data row4 col1" >(100000, 18)</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_4e1bb_row5_col0" class="data row5 col0" >Missing Values</td>
                        <td id="T_4e1bb_row5_col1" class="data row5 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_4e1bb_row6_col0" class="data row6 col0" >Numeric Features</td>
                        <td id="T_4e1bb_row6_col1" class="data row6 col1" >14</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_4e1bb_row7_col0" class="data row7 col0" >Categorical Features</td>
                        <td id="T_4e1bb_row7_col1" class="data row7 col1" >3</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_4e1bb_row8_col0" class="data row8 col0" >Ordinal Features</td>
                        <td id="T_4e1bb_row8_col1" class="data row8 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_4e1bb_row9_col0" class="data row9 col0" >High Cardinality Features</td>
                        <td id="T_4e1bb_row9_col1" class="data row9 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_4e1bb_row10_col0" class="data row10 col0" >High Cardinality Method</td>
                        <td id="T_4e1bb_row10_col1" class="data row10 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_4e1bb_row11_col0" class="data row11 col0" >Transformed Train Set</td>
                        <td id="T_4e1bb_row11_col1" class="data row11 col1" >(69999, 17)</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_4e1bb_row12_col0" class="data row12 col0" >Transformed Test Set</td>
                        <td id="T_4e1bb_row12_col1" class="data row12 col1" >(30001, 17)</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_4e1bb_row13_col0" class="data row13 col0" >Shuffle Train-Test</td>
                        <td id="T_4e1bb_row13_col1" class="data row13 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_4e1bb_row14_col0" class="data row14 col0" >Stratify Train-Test</td>
                        <td id="T_4e1bb_row14_col1" class="data row14 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_4e1bb_row15_col0" class="data row15 col0" >Fold Generator</td>
                        <td id="T_4e1bb_row15_col1" class="data row15 col1" >StratifiedKFold</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_4e1bb_row16_col0" class="data row16 col0" >Fold Number</td>
                        <td id="T_4e1bb_row16_col1" class="data row16 col1" >5</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_4e1bb_row17_col0" class="data row17 col0" >CPU Jobs</td>
                        <td id="T_4e1bb_row17_col1" class="data row17 col1" >-1</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_4e1bb_row18_col0" class="data row18 col0" >Use GPU</td>
                        <td id="T_4e1bb_row18_col1" class="data row18 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_4e1bb_row19_col0" class="data row19 col0" >Log Experiment</td>
                        <td id="T_4e1bb_row19_col1" class="data row19 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_4e1bb_row20_col0" class="data row20 col0" >Experiment Name</td>
                        <td id="T_4e1bb_row20_col1" class="data row20 col1" >clf-default-name</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_4e1bb_row21_col0" class="data row21 col0" >USI</td>
                        <td id="T_4e1bb_row21_col1" class="data row21 col1" >9220</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_4e1bb_row22_col0" class="data row22 col0" >Imputation Type</td>
                        <td id="T_4e1bb_row22_col1" class="data row22 col1" >simple</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_4e1bb_row23_col0" class="data row23 col0" >Iterative Imputation Iteration</td>
                        <td id="T_4e1bb_row23_col1" class="data row23 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_4e1bb_row24_col0" class="data row24 col0" >Numeric Imputer</td>
                        <td id="T_4e1bb_row24_col1" class="data row24 col1" >mean</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_4e1bb_row25_col0" class="data row25 col0" >Iterative Imputation Numeric Model</td>
                        <td id="T_4e1bb_row25_col1" class="data row25 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_4e1bb_row26_col0" class="data row26 col0" >Categorical Imputer</td>
                        <td id="T_4e1bb_row26_col1" class="data row26 col1" >constant</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_4e1bb_row27_col0" class="data row27 col0" >Iterative Imputation Categorical Model</td>
                        <td id="T_4e1bb_row27_col1" class="data row27 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_4e1bb_row28_col0" class="data row28 col0" >Unknown Categoricals Handling</td>
                        <td id="T_4e1bb_row28_col1" class="data row28 col1" >least_frequent</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_4e1bb_row29_col0" class="data row29 col0" >Normalize</td>
                        <td id="T_4e1bb_row29_col1" class="data row29 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_4e1bb_row30_col0" class="data row30 col0" >Normalize Method</td>
                        <td id="T_4e1bb_row30_col1" class="data row30 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_4e1bb_row31_col0" class="data row31 col0" >Transformation</td>
                        <td id="T_4e1bb_row31_col1" class="data row31 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_4e1bb_row32_col0" class="data row32 col0" >Transformation Method</td>
                        <td id="T_4e1bb_row32_col1" class="data row32 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_4e1bb_row33_col0" class="data row33 col0" >PCA</td>
                        <td id="T_4e1bb_row33_col1" class="data row33 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_4e1bb_row34_col0" class="data row34 col0" >PCA Method</td>
                        <td id="T_4e1bb_row34_col1" class="data row34 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_4e1bb_row35_col0" class="data row35 col0" >PCA Components</td>
                        <td id="T_4e1bb_row35_col1" class="data row35 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_4e1bb_row36_col0" class="data row36 col0" >Ignore Low Variance</td>
                        <td id="T_4e1bb_row36_col1" class="data row36 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_4e1bb_row37_col0" class="data row37 col0" >Combine Rare Levels</td>
                        <td id="T_4e1bb_row37_col1" class="data row37 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_4e1bb_row38_col0" class="data row38 col0" >Rare Level Threshold</td>
                        <td id="T_4e1bb_row38_col1" class="data row38 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_4e1bb_row39_col0" class="data row39 col0" >Numeric Binning</td>
                        <td id="T_4e1bb_row39_col1" class="data row39 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_4e1bb_row40_col0" class="data row40 col0" >Remove Outliers</td>
                        <td id="T_4e1bb_row40_col1" class="data row40 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_4e1bb_row41_col0" class="data row41 col0" >Outliers Threshold</td>
                        <td id="T_4e1bb_row41_col1" class="data row41 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row42" class="row_heading level0 row42" >42</th>
                        <td id="T_4e1bb_row42_col0" class="data row42 col0" >Remove Multicollinearity</td>
                        <td id="T_4e1bb_row42_col1" class="data row42 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row43" class="row_heading level0 row43" >43</th>
                        <td id="T_4e1bb_row43_col0" class="data row43 col0" >Multicollinearity Threshold</td>
                        <td id="T_4e1bb_row43_col1" class="data row43 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row44" class="row_heading level0 row44" >44</th>
                        <td id="T_4e1bb_row44_col0" class="data row44 col0" >Clustering</td>
                        <td id="T_4e1bb_row44_col1" class="data row44 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row45" class="row_heading level0 row45" >45</th>
                        <td id="T_4e1bb_row45_col0" class="data row45 col0" >Clustering Iteration</td>
                        <td id="T_4e1bb_row45_col1" class="data row45 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row46" class="row_heading level0 row46" >46</th>
                        <td id="T_4e1bb_row46_col0" class="data row46 col0" >Polynomial Features</td>
                        <td id="T_4e1bb_row46_col1" class="data row46 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row47" class="row_heading level0 row47" >47</th>
                        <td id="T_4e1bb_row47_col0" class="data row47 col0" >Polynomial Degree</td>
                        <td id="T_4e1bb_row47_col1" class="data row47 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row48" class="row_heading level0 row48" >48</th>
                        <td id="T_4e1bb_row48_col0" class="data row48 col0" >Trignometry Features</td>
                        <td id="T_4e1bb_row48_col1" class="data row48 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row49" class="row_heading level0 row49" >49</th>
                        <td id="T_4e1bb_row49_col0" class="data row49 col0" >Polynomial Threshold</td>
                        <td id="T_4e1bb_row49_col1" class="data row49 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row50" class="row_heading level0 row50" >50</th>
                        <td id="T_4e1bb_row50_col0" class="data row50 col0" >Group Features</td>
                        <td id="T_4e1bb_row50_col1" class="data row50 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row51" class="row_heading level0 row51" >51</th>
                        <td id="T_4e1bb_row51_col0" class="data row51 col0" >Feature Selection</td>
                        <td id="T_4e1bb_row51_col1" class="data row51 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row52" class="row_heading level0 row52" >52</th>
                        <td id="T_4e1bb_row52_col0" class="data row52 col0" >Features Selection Threshold</td>
                        <td id="T_4e1bb_row52_col1" class="data row52 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row53" class="row_heading level0 row53" >53</th>
                        <td id="T_4e1bb_row53_col0" class="data row53 col0" >Feature Interaction</td>
                        <td id="T_4e1bb_row53_col1" class="data row53 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row54" class="row_heading level0 row54" >54</th>
                        <td id="T_4e1bb_row54_col0" class="data row54 col0" >Feature Ratio</td>
                        <td id="T_4e1bb_row54_col1" class="data row54 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row55" class="row_heading level0 row55" >55</th>
                        <td id="T_4e1bb_row55_col0" class="data row55 col0" >Interaction Threshold</td>
                        <td id="T_4e1bb_row55_col1" class="data row55 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row56" class="row_heading level0 row56" >56</th>
                        <td id="T_4e1bb_row56_col0" class="data row56 col0" >Fix Imbalance</td>
                        <td id="T_4e1bb_row56_col1" class="data row56 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_4e1bb_level0_row57" class="row_heading level0 row57" >57</th>
                        <td id="T_4e1bb_row57_col0" class="data row57 col0" >Fix Imbalance Method</td>
                        <td id="T_4e1bb_row57_col1" class="data row57 col1" >SMOTE</td>
            </tr>
    </tbody></table>





    (1,
     {'lr': <pycaret.containers.models.classification.LogisticRegressionClassifierContainer at 0x7fdbc0f458d0>,
      'knn': <pycaret.containers.models.classification.KNeighborsClassifierContainer at 0x7fdbc0f45950>,
      'nb': <pycaret.containers.models.classification.GaussianNBClassifierContainer at 0x7fdbc0f45750>,
      'dt': <pycaret.containers.models.classification.DecisionTreeClassifierContainer at 0x7fdbc0f45790>,
      'svm': <pycaret.containers.models.classification.SGDClassifierContainer at 0x7fdbd7c5a2d0>,
      'rbfsvm': <pycaret.containers.models.classification.SVCClassifierContainer at 0x7fdbc188d650>,
      'gpc': <pycaret.containers.models.classification.GaussianProcessClassifierContainer at 0x7fdbc188d8d0>,
      'mlp': <pycaret.containers.models.classification.MLPClassifierContainer at 0x7fdbc188d490>,
      'ridge': <pycaret.containers.models.classification.RidgeClassifierContainer at 0x7fdbc188d810>,
      'rf': <pycaret.containers.models.classification.RandomForestClassifierContainer at 0x7fdbc0f45c90>,
      'qda': <pycaret.containers.models.classification.QuadraticDiscriminantAnalysisContainer at 0x7fdbc0857350>,
      'ada': <pycaret.containers.models.classification.AdaBoostClassifierContainer at 0x7fdbc0857410>,
      'gbc': <pycaret.containers.models.classification.GradientBoostingClassifierContainer at 0x7fdbc076e2d0>,
      'lda': <pycaret.containers.models.classification.LinearDiscriminantAnalysisContainer at 0x7fdbc076e250>,
      'et': <pycaret.containers.models.classification.ExtraTreesClassifierContainer at 0x7fdbc07621d0>,
      'xgboost': <pycaret.containers.models.classification.XGBClassifierContainer at 0x7fdbc07620d0>,
      'lightgbm': <pycaret.containers.models.classification.LGBMClassifierContainer at 0x7fdbc0762850>,
      'catboost': <pycaret.containers.models.classification.CatBoostClassifierContainer at 0x7fdbc0762890>},
     False,
     False,
     'clf-default-name',
     True,
     [<pandas.io.formats.style.Styler at 0x7fdbc0f33cd0>],
     {'acc': <pycaret.containers.metrics.classification.AccuracyMetricContainer at 0x7fdbc0f33590>,
      'auc': <pycaret.containers.metrics.classification.ROCAUCMetricContainer at 0x7fdbc0f335d0>,
      'recall': <pycaret.containers.metrics.classification.RecallMetricContainer at 0x7fdbc0f33650>,
      'precision': <pycaret.containers.metrics.classification.PrecisionMetricContainer at 0x7fdbc0f337d0>,
      'f1': <pycaret.containers.metrics.classification.F1MetricContainer at 0x7fdbc0f33950>,
      'kappa': <pycaret.containers.metrics.classification.KappaMetricContainer at 0x7fdbc0f33b10>,
      'mcc': <pycaret.containers.metrics.classification.MCCMetricContainer at 0x7fdbc0f33b90>},
     StratifiedKFold(n_splits=5, random_state=1, shuffle=True),
     {'USI',
      'X',
      'X_test',
      'X_train',
      '_all_metrics',
      '_all_models',
      '_all_models_internal',
      '_available_plots',
      '_gpu_n_jobs_param',
      '_internal_pipeline',
      '_ml_usecase',
      'create_model_container',
      'data_before_preprocess',
      'display_container',
      'exp_name_log',
      'experiment__',
      'fix_imbalance_method_param',
      'fix_imbalance_param',
      'fold_generator',
      'fold_groups_param',
      'fold_param',
      'fold_shuffle_param',
      'gpu_param',
      'html_param',
      'imputation_classifier',
      'imputation_regressor',
      'iterative_imputation_iters_param',
      'log_plots_param',
      'logging_param',
      'master_model_container',
      'n_jobs_param',
      'prep_pipe',
      'pycaret_globals',
      'seed',
      'stratify_param',
      'target_param',
      'transform_target_method_param',
      'transform_target_param',
      'y',
      'y_test',
      'y_train'},
     'Survived',
     True,
     -1,
                 Age     SibSp      Fare     Name  Ticket  Sex  Pclass  Embarked  \
     62017 -1.786066 -0.539572 -0.425227  23058.0    38.0  0.0     1.0       2.0   
     5005   0.274926 -0.539572  0.041264   9979.0    49.0  1.0     1.0       2.0   
     56849 -1.361744 -0.539572  0.215883  14486.0    49.0  0.0     1.0       2.0   
     42434  1.426657 -0.539572  1.209948   8350.0    21.0  0.0     0.0       0.0   
     54712 -1.725448 -0.539572 -0.906172  18642.0     0.0  0.0     2.0       2.0   
     ...         ...       ...       ...      ...     ...  ...     ...       ...   
     50057  1.608510 -0.539572  1.938114   4224.0    49.0  1.0     0.0       2.0   
     98047 -0.149396  0.680848 -0.914255  21294.0    49.0  1.0     2.0       2.0   
     5192  -0.634335 -0.539572 -1.037225  24329.0    49.0  1.0     1.0       2.0   
     77708  0.820483 -0.539572  2.156857   5150.0    49.0  0.0     0.0       0.0   
     98539  0.396161 -0.539572  2.729733  25222.0    49.0  1.0     0.0       1.0   
     
            Cabin_A  Cabin_B  Cabin_C  Cabin_D  Cabin_E  Cabin_F  Cabin_G  Cabin_T  \
     62017        0        0        0        0        0        0        0        0   
     5005         0        0        0        0        0        0        0        0   
     56849        0        0        0        0        0        0        0        0   
     42434        0        0        1        0        0        0        0        0   
     54712        0        0        0        0        0        0        0        0   
     ...        ...      ...      ...      ...      ...      ...      ...      ...   
     50057        1        0        0        0        0        0        0        0   
     98047        0        0        0        0        0        0        0        0   
     5192         0        0        0        0        0        0        0        0   
     77708        0        1        0        0        0        0        0        0   
     98539        1        0        0        0        0        0        0        0   
     
            Cabin_X  
     62017        1  
     5005         1  
     56849        1  
     42434        0  
     54712        1  
     ...        ...  
     50057        0  
     98047        1  
     5192         1  
     77708        0  
     98539        0  
     
     [69999 rows x 17 columns],
     Pipeline(memory=None, steps=[('empty_step', 'passthrough')], verbose=False),
     'lightgbm',
     False,
     Pipeline(memory=None,
              steps=[('dtypes',
                      DataTypes_Auto_infer(categorical_features=[],
                                           display_types=False, features_todrop=[],
                                           id_columns=[],
                                           ml_usecase='classification',
                                           numerical_features=[], target='Survived',
                                           time_features=[])),
                     ('imputer',
                      Simple_Imputer(categorical_strategy='not_available',
                                     fill_value_categorical=None,
                                     fill_value_numerical=None,
                                     numeric_st...
                     ('scaling', 'passthrough'), ('P_transform', 'passthrough'),
                     ('binn', 'passthrough'), ('rem_outliers', 'passthrough'),
                     ('cluster_all', 'passthrough'),
                     ('dummy', Dummify(target='Survived')),
                     ('fix_perfect', Remove_100(target='Survived')),
                     ('clean_names', Clean_Colum_Names()),
                     ('feature_select', 'passthrough'), ('fix_multi', 'passthrough'),
                     ('dfs', 'passthrough'), ('pca', 'passthrough')],
              verbose=False),
     [],
     '9220',
     0        1
     1        0
     2        0
     3        0
     4        1
             ..
     99995    1
     99996    0
     99997    0
     99998    0
     99999    0
     Name: Survived, Length: 100000, dtype: int64,
     False,
     [],
     'box-cox',
     {'lr': <pycaret.containers.models.classification.LogisticRegressionClassifierContainer at 0x7fdbc078af90>,
      'knn': <pycaret.containers.models.classification.KNeighborsClassifierContainer at 0x7fdbc078ae50>,
      'nb': <pycaret.containers.models.classification.GaussianNBClassifierContainer at 0x7fdbc078ad10>,
      'dt': <pycaret.containers.models.classification.DecisionTreeClassifierContainer at 0x7fdbc078ae90>,
      'svm': <pycaret.containers.models.classification.SGDClassifierContainer at 0x7fdbc078aa10>,
      'rbfsvm': <pycaret.containers.models.classification.SVCClassifierContainer at 0x7fdbc078a710>,
      'gpc': <pycaret.containers.models.classification.GaussianProcessClassifierContainer at 0x7fdbc078a8d0>,
      'mlp': <pycaret.containers.models.classification.MLPClassifierContainer at 0x7fdbc078a550>,
      'ridge': <pycaret.containers.models.classification.RidgeClassifierContainer at 0x7fdbc078a2d0>,
      'rf': <pycaret.containers.models.classification.RandomForestClassifierContainer at 0x7fdbc078a990>,
      'qda': <pycaret.containers.models.classification.QuadraticDiscriminantAnalysisContainer at 0x7fdbc078b210>,
      'ada': <pycaret.containers.models.classification.AdaBoostClassifierContainer at 0x7fdbc078b250>,
      'gbc': <pycaret.containers.models.classification.GradientBoostingClassifierContainer at 0x7fdbc078b4d0>,
      'lda': <pycaret.containers.models.classification.LinearDiscriminantAnalysisContainer at 0x7fdbc078b7d0>,
      'et': <pycaret.containers.models.classification.ExtraTreesClassifierContainer at 0x7fdbc078b8d0>,
      'xgboost': <pycaret.containers.models.classification.XGBClassifierContainer at 0x7fdbc078bc10>,
      'lightgbm': <pycaret.containers.models.classification.LGBMClassifierContainer at 0x7fdbc0f33090>,
      'catboost': <pycaret.containers.models.classification.CatBoostClassifierContainer at 0x7fdbc078afd0>,
      'Bagging': <pycaret.containers.models.classification.BaggingClassifierContainer at 0x7fdbc0762ed0>,
      'Stacking': <pycaret.containers.models.classification.StackingClassifierContainer at 0x7fdbc0762f10>,
      'Voting': <pycaret.containers.models.classification.VotingClassifierContainer at 0x7fdbc0f33190>,
      'CalibratedCV': <pycaret.containers.models.classification.CalibratedClassifierCVContainer at 0x7fdbc0f334d0>},
                     Age     SibSp      Fare   Name  Ticket  Sex  Pclass  Embarked  \
     0     -8.614253e-16  1.901268  0.134351  17441      49    1       0         2   
     1     -8.614253e-16 -0.539572 -0.533837   3063      49    1       2         2   
     2     -2.069149e+00  0.680848  1.070483  17798      14    1       2         2   
     3     -9.374220e-01 -0.539572 -0.555506  12742       0    1       2         2   
     4     -5.737175e-01 -0.539572 -1.023540   2335      49    1       2         2   
     ...             ...       ...       ...    ...     ...  ...     ...       ...   
     99995  1.669127e+00 -0.539572 -0.434567   1590      21    0       1         0   
     99996  1.911597e+00 -0.539572 -0.698959   2992      49    1       1         2   
     99997  1.536915e-01 -0.539572 -0.802137   4219      49    1       2         2   
     99998  1.002335e+00 -0.539572  0.259408   3941      49    1       2         2   
     99999  1.244805e+00 -0.539572 -0.492531   7055      49    1       2         2   
     
            Cabin_A  Cabin_B  Cabin_C  Cabin_D  Cabin_E  Cabin_F  Cabin_G  Cabin_T  \
     0            0        0        1        0        0        0        0        0   
     1            0        0        0        0        0        0        0        0   
     2            0        0        0        0        0        0        0        0   
     3            0        0        0        0        0        0        0        0   
     4            0        0        0        0        0        0        0        0   
     ...        ...      ...      ...      ...      ...      ...      ...      ...   
     99995        0        0        0        1        0        0        0        0   
     99996        0        0        0        0        0        0        0        0   
     99997        0        0        0        0        0        0        0        0   
     99998        0        0        0        0        0        0        0        0   
     99999        0        0        0        0        0        0        0        0   
     
            Cabin_X  Survived  
     0            0       1.0  
     1            1       0.0  
     2            1       0.0  
     3            1       0.0  
     4            1       1.0  
     ...        ...       ...  
     99995        0       1.0  
     99996        1       0.0  
     99997        1       0.0  
     99998        1       0.0  
     99999        1       0.0  
     
     [100000 rows x 18 columns],
     5,
     [('Setup Config',
                                      Description             Value
       0                               session_id                 1
       1                                   Target          Survived
       2                              Target Type            Binary
       3                            Label Encoded    0.0: 0, 1.0: 1
       4                            Original Data      (100000, 18)
       5                           Missing Values             False
       6                         Numeric Features                14
       7                     Categorical Features                 3
       8                         Ordinal Features              True
       9                High Cardinality Features             False
       10                 High Cardinality Method              None
       11                   Transformed Train Set       (69999, 17)
       12                    Transformed Test Set       (30001, 17)
       13                      Shuffle Train-Test              True
       14                     Stratify Train-Test             False
       15                          Fold Generator   StratifiedKFold
       16                             Fold Number                 5
       17                                CPU Jobs                -1
       18                                 Use GPU             False
       19                          Log Experiment             False
       20                         Experiment Name  clf-default-name
       21                                     USI              9220
       22                         Imputation Type            simple
       23          Iterative Imputation Iteration              None
       24                         Numeric Imputer              mean
       25      Iterative Imputation Numeric Model              None
       26                     Categorical Imputer          constant
       27  Iterative Imputation Categorical Model              None
       28           Unknown Categoricals Handling    least_frequent
       29                               Normalize             False
       30                        Normalize Method              None
       31                          Transformation             False
       32                   Transformation Method              None
       33                                     PCA             False
       34                              PCA Method              None
       35                          PCA Components              None
       36                     Ignore Low Variance             False
       37                     Combine Rare Levels             False
       38                    Rare Level Threshold              None
       39                         Numeric Binning             False
       40                         Remove Outliers             False
       41                      Outliers Threshold              None
       42                Remove Multicollinearity             False
       43             Multicollinearity Threshold              None
       44                              Clustering             False
       45                    Clustering Iteration              None
       46                     Polynomial Features             False
       47                       Polynomial Degree              None
       48                    Trignometry Features             False
       49                    Polynomial Threshold              None
       50                          Group Features             False
       51                       Feature Selection             False
       52            Features Selection Threshold              None
       53                     Feature Interaction             False
       54                           Feature Ratio             False
       55                   Interaction Threshold              None
       56                           Fix Imbalance             False
       57                    Fix Imbalance Method             SMOTE),
      ('X_training Set',
                   Age     SibSp      Fare     Name  Ticket  Sex  Pclass  Embarked  \
       62017 -1.786066 -0.539572 -0.425227  23058.0    38.0  0.0     1.0       2.0   
       5005   0.274926 -0.539572  0.041264   9979.0    49.0  1.0     1.0       2.0   
       56849 -1.361744 -0.539572  0.215883  14486.0    49.0  0.0     1.0       2.0   
       42434  1.426657 -0.539572  1.209948   8350.0    21.0  0.0     0.0       0.0   
       54712 -1.725448 -0.539572 -0.906172  18642.0     0.0  0.0     2.0       2.0   
       ...         ...       ...       ...      ...     ...  ...     ...       ...   
       50057  1.608510 -0.539572  1.938114   4224.0    49.0  1.0     0.0       2.0   
       98047 -0.149396  0.680848 -0.914255  21294.0    49.0  1.0     2.0       2.0   
       5192  -0.634335 -0.539572 -1.037225  24329.0    49.0  1.0     1.0       2.0   
       77708  0.820483 -0.539572  2.156857   5150.0    49.0  0.0     0.0       0.0   
       98539  0.396161 -0.539572  2.729733  25222.0    49.0  1.0     0.0       1.0   
       
              Cabin_A  Cabin_B  Cabin_C  Cabin_D  Cabin_E  Cabin_F  Cabin_G  Cabin_T  \
       62017        0        0        0        0        0        0        0        0   
       5005         0        0        0        0        0        0        0        0   
       56849        0        0        0        0        0        0        0        0   
       42434        0        0        1        0        0        0        0        0   
       54712        0        0        0        0        0        0        0        0   
       ...        ...      ...      ...      ...      ...      ...      ...      ...   
       50057        1        0        0        0        0        0        0        0   
       98047        0        0        0        0        0        0        0        0   
       5192         0        0        0        0        0        0        0        0   
       77708        0        1        0        0        0        0        0        0   
       98539        1        0        0        0        0        0        0        0   
       
              Cabin_X  
       62017        1  
       5005         1  
       56849        1  
       42434        0  
       54712        1  
       ...        ...  
       50057        0  
       98047        1  
       5192         1  
       77708        0  
       98539        0  
       
       [69999 rows x 17 columns]),
      ('y_training Set',
       62017    0
       5005     0
       56849    1
       42434    1
       54712    0
               ..
       50057    1
       98047    0
       5192     0
       77708    0
       98539    1
       Name: Survived, Length: 69999, dtype: int64),
      ('X_test Set',
                   Age     SibSp      Fare     Name  Ticket  Sex  Pclass  Embarked  \
       43660  1.244805 -0.539572  0.820679  25313.0    49.0  0.0     0.0       2.0   
       87278 -0.210013 -0.539572 -0.555506   3156.0    49.0  1.0     0.0       0.0   
       14317  0.941718  0.680848  1.561942  11588.0    49.0  1.0     0.0       2.0   
       81932  0.638631 -0.539572  0.050516  16175.0    21.0  1.0     0.0       0.0   
       95321 -0.694952  0.680848  0.063476  10196.0    49.0  1.0     0.0       2.0   
       ...         ...       ...       ...      ...     ...  ...     ...       ...   
       42287 -1.179892 -0.539572 -0.463126  11629.0    49.0  1.0     2.0       2.0   
       4967  -0.452483 -0.539572  0.265297   3394.0    49.0  1.0     1.0       0.0   
       47725  1.244805 -0.539572 -0.364529  19040.0    49.0  1.0     1.0       2.0   
       42348 -0.694952 -0.539572 -1.197666  15816.0    49.0  1.0     2.0       2.0   
       80630 -0.452483 -0.539572 -0.900153  24505.0    49.0  1.0     2.0       2.0   
       
              Cabin_A  Cabin_B  Cabin_C  Cabin_D  Cabin_E  Cabin_F  Cabin_G  Cabin_T  \
       43660        0        0        1        0        0        0        0        0   
       87278        0        0        1        0        0        0        0        0   
       14317        1        0        0        0        0        0        0        0   
       81932        1        0        0        0        0        0        0        0   
       95321        0        0        1        0        0        0        0        0   
       ...        ...      ...      ...      ...      ...      ...      ...      ...   
       42287        0        0        0        0        0        0        0        0   
       4967         0        0        0        0        0        0        0        0   
       47725        0        0        0        0        0        0        0        0   
       42348        0        0        0        0        0        0        0        0   
       80630        0        0        0        0        0        0        0        0   
       
              Cabin_X  
       43660        0  
       87278        0  
       14317        0  
       81932        0  
       95321        0  
       ...        ...  
       42287        1  
       4967         1  
       47725        1  
       42348        1  
       80630        1  
       
       [30001 rows x 17 columns]),
      ('y_test Set',
       43660    1
       87278    0
       14317    0
       81932    1
       95321    0
               ..
       42287    0
       4967     1
       47725    0
       42348    0
       80630    0
       Name: Survived, Length: 30001, dtype: int64),
      ('Transformation Pipeline',
       Pipeline(memory=None,
                steps=[('dtypes',
                        DataTypes_Auto_infer(categorical_features=[],
                                             display_types=False, features_todrop=[],
                                             id_columns=[],
                                             ml_usecase='classification',
                                             numerical_features=[], target='Survived',
                                             time_features=[])),
                       ('imputer',
                        Simple_Imputer(categorical_strategy='not_available',
                                       fill_value_categorical=None,
                                       fill_value_numerical=None,
                                       numeric_st...
                       ('scaling', 'passthrough'), ('P_transform', 'passthrough'),
                       ('binn', 'passthrough'), ('rem_outliers', 'passthrough'),
                       ('cluster_all', 'passthrough'),
                       ('dummy', Dummify(target='Survived')),
                       ('fix_perfect', Remove_100(target='Survived')),
                       ('clean_names', Clean_Colum_Names()),
                       ('feature_select', 'passthrough'), ('fix_multi', 'passthrough'),
                       ('dfs', 'passthrough'), ('pca', 'passthrough')],
                verbose=False))],
     'lightgbm',
     False,
                 Age     SibSp      Fare     Name  Ticket  Sex  Pclass  Embarked  \
     43660  1.244805 -0.539572  0.820679  25313.0    49.0  0.0     0.0       2.0   
     87278 -0.210013 -0.539572 -0.555506   3156.0    49.0  1.0     0.0       0.0   
     14317  0.941718  0.680848  1.561942  11588.0    49.0  1.0     0.0       2.0   
     81932  0.638631 -0.539572  0.050516  16175.0    21.0  1.0     0.0       0.0   
     95321 -0.694952  0.680848  0.063476  10196.0    49.0  1.0     0.0       2.0   
     ...         ...       ...       ...      ...     ...  ...     ...       ...   
     42287 -1.179892 -0.539572 -0.463126  11629.0    49.0  1.0     2.0       2.0   
     4967  -0.452483 -0.539572  0.265297   3394.0    49.0  1.0     1.0       0.0   
     47725  1.244805 -0.539572 -0.364529  19040.0    49.0  1.0     1.0       2.0   
     42348 -0.694952 -0.539572 -1.197666  15816.0    49.0  1.0     2.0       2.0   
     80630 -0.452483 -0.539572 -0.900153  24505.0    49.0  1.0     2.0       2.0   
     
            Cabin_A  Cabin_B  Cabin_C  Cabin_D  Cabin_E  Cabin_F  Cabin_G  Cabin_T  \
     43660        0        0        1        0        0        0        0        0   
     87278        0        0        1        0        0        0        0        0   
     14317        1        0        0        0        0        0        0        0   
     81932        1        0        0        0        0        0        0        0   
     95321        0        0        1        0        0        0        0        0   
     ...        ...      ...      ...      ...      ...      ...      ...      ...   
     42287        0        0        0        0        0        0        0        0   
     4967         0        0        0        0        0        0        0        0   
     47725        0        0        0        0        0        0        0        0   
     42348        0        0        0        0        0        0        0        0   
     80630        0        0        0        0        0        0        0        0   
     
            Cabin_X  
     43660        0  
     87278        0  
     14317        0  
     81932        0  
     95321        0  
     ...        ...  
     42287        1  
     4967         1  
     47725        1  
     42348        1  
     80630        1  
     
     [30001 rows x 17 columns],
     62017    0
     5005     0
     56849    1
     42434    1
     54712    0
             ..
     50057    1
     98047    0
     5192     0
     77708    0
     98539    1
     Name: Survived, Length: 69999, dtype: int64,
     {'parameter': 'Hyperparameters',
      'auc': 'AUC',
      'confusion_matrix': 'Confusion Matrix',
      'threshold': 'Threshold',
      'pr': 'Precision Recall',
      'error': 'Prediction Error',
      'class_report': 'Class Report',
      'rfe': 'Feature Selection',
      'learning': 'Learning Curve',
      'manifold': 'Manifold Learning',
      'calibration': 'Calibration Curve',
      'vc': 'Validation Curve',
      'dimension': 'Dimensions',
      'feature': 'Feature Importance',
      'feature_all': 'Feature Importance (All)',
      'boundary': 'Decision Boundary',
      'lift': 'Lift Chart',
      'gain': 'Gain Chart',
      'tree': 'Decision Tree'},
     None,
     <MLUsecase.CLASSIFICATION: 1>,
     -1,
     43660    1
     87278    0
     14317    0
     81932    1
     95321    0
             ..
     42287    0
     4967     1
     47725    0
     42348    0
     80630    0
     Name: Survived, Length: 30001, dtype: int64,
                     Age     SibSp      Fare     Name  Ticket  Sex  Pclass  \
     0     -8.614253e-16  1.901268  0.134351  17441.0    49.0  1.0     0.0   
     1     -8.614253e-16 -0.539572 -0.533837   3063.0    49.0  1.0     2.0   
     2     -2.069149e+00  0.680848  1.070483  17798.0    14.0  1.0     2.0   
     3     -9.374220e-01 -0.539572 -0.555506  12742.0     0.0  1.0     2.0   
     4     -5.737175e-01 -0.539572 -1.023540   2335.0    49.0  1.0     2.0   
     ...             ...       ...       ...      ...     ...  ...     ...   
     99995  1.669127e+00 -0.539572 -0.434567   1590.0    21.0  0.0     1.0   
     99996  1.911597e+00 -0.539572 -0.698959   2992.0    49.0  1.0     1.0   
     99997  1.536915e-01 -0.539572 -0.802137   4219.0    49.0  1.0     2.0   
     99998  1.002335e+00 -0.539572  0.259408   3941.0    49.0  1.0     2.0   
     99999  1.244805e+00 -0.539572 -0.492531   7055.0    49.0  1.0     2.0   
     
            Embarked  Cabin_A  Cabin_B  Cabin_C  Cabin_D  Cabin_E  Cabin_F  \
     0           2.0        0        0        1        0        0        0   
     1           2.0        0        0        0        0        0        0   
     2           2.0        0        0        0        0        0        0   
     3           2.0        0        0        0        0        0        0   
     4           2.0        0        0        0        0        0        0   
     ...         ...      ...      ...      ...      ...      ...      ...   
     99995       0.0        0        0        0        1        0        0   
     99996       2.0        0        0        0        0        0        0   
     99997       2.0        0        0        0        0        0        0   
     99998       2.0        0        0        0        0        0        0   
     99999       2.0        0        0        0        0        0        0   
     
            Cabin_G  Cabin_T  Cabin_X  
     0            0        0        0  
     1            0        0        1  
     2            0        0        1  
     3            0        0        1  
     4            0        0        1  
     ...        ...      ...      ...  
     99995        0        0        0  
     99996        0        0        1  
     99997        0        0        1  
     99998        0        0        1  
     99999        0        0        1  
     
     [100000 rows x 17 columns],
     5,
     None,
     False)




```python

```


```python
#best_model = compare_models(sort = 'Accuracy', n_select = 4)
```


```python
# gbc=create_model('gbc')
```


```python
# gbc=tune_model(gbc)
```


```python
# print(gbc)
```


```python
lightgbm=create_model('lightgbm'

                     )
```


<style  type="text/css" >
#T_222f1_row5_col0,#T_222f1_row5_col1,#T_222f1_row5_col2,#T_222f1_row5_col3,#T_222f1_row5_col4,#T_222f1_row5_col5,#T_222f1_row5_col6{
            background:  yellow;
        }</style><table id="T_222f1_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_222f1_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_222f1_row0_col0" class="data row0 col0" >0.7825</td>
                        <td id="T_222f1_row0_col1" class="data row0 col1" >0.8494</td>
                        <td id="T_222f1_row0_col2" class="data row0 col2" >0.7359</td>
                        <td id="T_222f1_row0_col3" class="data row0 col3" >0.7513</td>
                        <td id="T_222f1_row0_col4" class="data row0 col4" >0.7435</td>
                        <td id="T_222f1_row0_col5" class="data row0 col5" >0.5547</td>
                        <td id="T_222f1_row0_col6" class="data row0 col6" >0.5548</td>
            </tr>
            <tr>
                        <th id="T_222f1_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_222f1_row1_col0" class="data row1 col0" >0.7801</td>
                        <td id="T_222f1_row1_col1" class="data row1 col1" >0.8490</td>
                        <td id="T_222f1_row1_col2" class="data row1 col2" >0.7437</td>
                        <td id="T_222f1_row1_col3" class="data row1 col3" >0.7431</td>
                        <td id="T_222f1_row1_col4" class="data row1 col4" >0.7434</td>
                        <td id="T_222f1_row1_col5" class="data row1 col5" >0.5510</td>
                        <td id="T_222f1_row1_col6" class="data row1 col6" >0.5510</td>
            </tr>
            <tr>
                        <th id="T_222f1_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_222f1_row2_col0" class="data row2 col0" >0.7863</td>
                        <td id="T_222f1_row2_col1" class="data row2 col1" >0.8547</td>
                        <td id="T_222f1_row2_col2" class="data row2 col2" >0.7467</td>
                        <td id="T_222f1_row2_col3" class="data row2 col3" >0.7525</td>
                        <td id="T_222f1_row2_col4" class="data row2 col4" >0.7496</td>
                        <td id="T_222f1_row2_col5" class="data row2 col5" >0.5632</td>
                        <td id="T_222f1_row2_col6" class="data row2 col6" >0.5632</td>
            </tr>
            <tr>
                        <th id="T_222f1_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_222f1_row3_col0" class="data row3 col0" >0.7796</td>
                        <td id="T_222f1_row3_col1" class="data row3 col1" >0.8490</td>
                        <td id="T_222f1_row3_col2" class="data row3 col2" >0.7391</td>
                        <td id="T_222f1_row3_col3" class="data row3 col3" >0.7447</td>
                        <td id="T_222f1_row3_col4" class="data row3 col4" >0.7419</td>
                        <td id="T_222f1_row3_col5" class="data row3 col5" >0.5496</td>
                        <td id="T_222f1_row3_col6" class="data row3 col6" >0.5497</td>
            </tr>
            <tr>
                        <th id="T_222f1_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_222f1_row4_col0" class="data row4 col0" >0.7833</td>
                        <td id="T_222f1_row4_col1" class="data row4 col1" >0.8502</td>
                        <td id="T_222f1_row4_col2" class="data row4 col2" >0.7475</td>
                        <td id="T_222f1_row4_col3" class="data row4 col3" >0.7469</td>
                        <td id="T_222f1_row4_col4" class="data row4 col4" >0.7472</td>
                        <td id="T_222f1_row4_col5" class="data row4 col5" >0.5577</td>
                        <td id="T_222f1_row4_col6" class="data row4 col6" >0.5577</td>
            </tr>
            <tr>
                        <th id="T_222f1_level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_222f1_row5_col0" class="data row5 col0" >0.7824</td>
                        <td id="T_222f1_row5_col1" class="data row5 col1" >0.8505</td>
                        <td id="T_222f1_row5_col2" class="data row5 col2" >0.7426</td>
                        <td id="T_222f1_row5_col3" class="data row5 col3" >0.7477</td>
                        <td id="T_222f1_row5_col4" class="data row5 col4" >0.7451</td>
                        <td id="T_222f1_row5_col5" class="data row5 col5" >0.5552</td>
                        <td id="T_222f1_row5_col6" class="data row5 col6" >0.5553</td>
            </tr>
            <tr>
                        <th id="T_222f1_level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_222f1_row6_col0" class="data row6 col0" >0.0024</td>
                        <td id="T_222f1_row6_col1" class="data row6 col1" >0.0022</td>
                        <td id="T_222f1_row6_col2" class="data row6 col2" >0.0045</td>
                        <td id="T_222f1_row6_col3" class="data row6 col3" >0.0037</td>
                        <td id="T_222f1_row6_col4" class="data row6 col4" >0.0029</td>
                        <td id="T_222f1_row6_col5" class="data row6 col5" >0.0049</td>
                        <td id="T_222f1_row6_col6" class="data row6 col6" >0.0049</td>
            </tr>
    </tbody></table>



```python
lightgbm = tune_model(lightgbm
                     , optimize='AUC' 
                     )
```


<style  type="text/css" >
#T_9e81c_row5_col0,#T_9e81c_row5_col1,#T_9e81c_row5_col2,#T_9e81c_row5_col3,#T_9e81c_row5_col4,#T_9e81c_row5_col5,#T_9e81c_row5_col6{
            background:  yellow;
        }</style><table id="T_9e81c_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_9e81c_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_9e81c_row0_col0" class="data row0 col0" >0.7793</td>
                        <td id="T_9e81c_row0_col1" class="data row0 col1" >0.8496</td>
                        <td id="T_9e81c_row0_col2" class="data row0 col2" >0.7382</td>
                        <td id="T_9e81c_row0_col3" class="data row0 col3" >0.7444</td>
                        <td id="T_9e81c_row0_col4" class="data row0 col4" >0.7413</td>
                        <td id="T_9e81c_row0_col5" class="data row0 col5" >0.5488</td>
                        <td id="T_9e81c_row0_col6" class="data row0 col6" >0.5489</td>
            </tr>
            <tr>
                        <th id="T_9e81c_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_9e81c_row1_col0" class="data row1 col0" >0.7797</td>
                        <td id="T_9e81c_row1_col1" class="data row1 col1" >0.8492</td>
                        <td id="T_9e81c_row1_col2" class="data row1 col2" >0.7439</td>
                        <td id="T_9e81c_row1_col3" class="data row1 col3" >0.7424</td>
                        <td id="T_9e81c_row1_col4" class="data row1 col4" >0.7431</td>
                        <td id="T_9e81c_row1_col5" class="data row1 col5" >0.5503</td>
                        <td id="T_9e81c_row1_col6" class="data row1 col6" >0.5503</td>
            </tr>
            <tr>
                        <th id="T_9e81c_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_9e81c_row2_col0" class="data row2 col0" >0.7858</td>
                        <td id="T_9e81c_row2_col1" class="data row2 col1" >0.8547</td>
                        <td id="T_9e81c_row2_col2" class="data row2 col2" >0.7489</td>
                        <td id="T_9e81c_row2_col3" class="data row2 col3" >0.7505</td>
                        <td id="T_9e81c_row2_col4" class="data row2 col4" >0.7497</td>
                        <td id="T_9e81c_row2_col5" class="data row2 col5" >0.5625</td>
                        <td id="T_9e81c_row2_col6" class="data row2 col6" >0.5625</td>
            </tr>
            <tr>
                        <th id="T_9e81c_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_9e81c_row3_col0" class="data row3 col0" >0.7777</td>
                        <td id="T_9e81c_row3_col1" class="data row3 col1" >0.8493</td>
                        <td id="T_9e81c_row3_col2" class="data row3 col2" >0.7467</td>
                        <td id="T_9e81c_row3_col3" class="data row3 col3" >0.7376</td>
                        <td id="T_9e81c_row3_col4" class="data row3 col4" >0.7422</td>
                        <td id="T_9e81c_row3_col5" class="data row3 col5" >0.5468</td>
                        <td id="T_9e81c_row3_col6" class="data row3 col6" >0.5469</td>
            </tr>
            <tr>
                        <th id="T_9e81c_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_9e81c_row4_col0" class="data row4 col0" >0.7825</td>
                        <td id="T_9e81c_row4_col1" class="data row4 col1" >0.8508</td>
                        <td id="T_9e81c_row4_col2" class="data row4 col2" >0.7429</td>
                        <td id="T_9e81c_row4_col3" class="data row4 col3" >0.7477</td>
                        <td id="T_9e81c_row4_col4" class="data row4 col4" >0.7453</td>
                        <td id="T_9e81c_row4_col5" class="data row4 col5" >0.5555</td>
                        <td id="T_9e81c_row4_col6" class="data row4 col6" >0.5555</td>
            </tr>
            <tr>
                        <th id="T_9e81c_level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_9e81c_row5_col0" class="data row5 col0" >0.7810</td>
                        <td id="T_9e81c_row5_col1" class="data row5 col1" >0.8507</td>
                        <td id="T_9e81c_row5_col2" class="data row5 col2" >0.7441</td>
                        <td id="T_9e81c_row5_col3" class="data row5 col3" >0.7445</td>
                        <td id="T_9e81c_row5_col4" class="data row5 col4" >0.7443</td>
                        <td id="T_9e81c_row5_col5" class="data row5 col5" >0.5528</td>
                        <td id="T_9e81c_row5_col6" class="data row5 col6" >0.5528</td>
            </tr>
            <tr>
                        <th id="T_9e81c_level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_9e81c_row6_col0" class="data row6 col0" >0.0028</td>
                        <td id="T_9e81c_row6_col1" class="data row6 col1" >0.0021</td>
                        <td id="T_9e81c_row6_col2" class="data row6 col2" >0.0036</td>
                        <td id="T_9e81c_row6_col3" class="data row6 col3" >0.0044</td>
                        <td id="T_9e81c_row6_col4" class="data row6 col4" >0.0030</td>
                        <td id="T_9e81c_row6_col5" class="data row6 col5" >0.0056</td>
                        <td id="T_9e81c_row6_col6" class="data row6 col6" >0.0056</td>
            </tr>
    </tbody></table>



```python
print(lightgbm)
```

    LGBMClassifier(bagging_fraction=0.8, bagging_freq=5, boosting_type='gbdt',
                   class_weight=None, colsample_bytree=1.0, feature_fraction=0.9,
                   importance_type='split', learning_rate=0.103, max_depth=-1,
                   min_child_samples=30, min_child_weight=0.001, min_split_gain=0.4,
                   n_estimators=40, n_jobs=-1, num_leaves=30, objective=None,
                   random_state=1, reg_alpha=2, reg_lambda=0.2, silent=True,
                   subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
    


```python
catboost=create_model('catboost'

                     )
```


<style  type="text/css" >
#T_cd7ef_row5_col0,#T_cd7ef_row5_col1,#T_cd7ef_row5_col2,#T_cd7ef_row5_col3,#T_cd7ef_row5_col4,#T_cd7ef_row5_col5,#T_cd7ef_row5_col6{
            background:  yellow;
        }</style><table id="T_cd7ef_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_cd7ef_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_cd7ef_row0_col0" class="data row0 col0" >0.7786</td>
                        <td id="T_cd7ef_row0_col1" class="data row0 col1" >0.8480</td>
                        <td id="T_cd7ef_row0_col2" class="data row0 col2" >0.7287</td>
                        <td id="T_cd7ef_row0_col3" class="data row0 col3" >0.7479</td>
                        <td id="T_cd7ef_row0_col4" class="data row0 col4" >0.7382</td>
                        <td id="T_cd7ef_row0_col5" class="data row0 col5" >0.5464</td>
                        <td id="T_cd7ef_row0_col6" class="data row0 col6" >0.5465</td>
            </tr>
            <tr>
                        <th id="T_cd7ef_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_cd7ef_row1_col0" class="data row1 col0" >0.7774</td>
                        <td id="T_cd7ef_row1_col1" class="data row1 col1" >0.8474</td>
                        <td id="T_cd7ef_row1_col2" class="data row1 col2" >0.7394</td>
                        <td id="T_cd7ef_row1_col3" class="data row1 col3" >0.7405</td>
                        <td id="T_cd7ef_row1_col4" class="data row1 col4" >0.7399</td>
                        <td id="T_cd7ef_row1_col5" class="data row1 col5" >0.5453</td>
                        <td id="T_cd7ef_row1_col6" class="data row1 col6" >0.5453</td>
            </tr>
            <tr>
                        <th id="T_cd7ef_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_cd7ef_row2_col0" class="data row2 col0" >0.7826</td>
                        <td id="T_cd7ef_row2_col1" class="data row2 col1" >0.8532</td>
                        <td id="T_cd7ef_row2_col2" class="data row2 col2" >0.7417</td>
                        <td id="T_cd7ef_row2_col3" class="data row2 col3" >0.7486</td>
                        <td id="T_cd7ef_row2_col4" class="data row2 col4" >0.7452</td>
                        <td id="T_cd7ef_row2_col5" class="data row2 col5" >0.5557</td>
                        <td id="T_cd7ef_row2_col6" class="data row2 col6" >0.5557</td>
            </tr>
            <tr>
                        <th id="T_cd7ef_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_cd7ef_row3_col0" class="data row3 col0" >0.7761</td>
                        <td id="T_cd7ef_row3_col1" class="data row3 col1" >0.8477</td>
                        <td id="T_cd7ef_row3_col2" class="data row3 col2" >0.7354</td>
                        <td id="T_cd7ef_row3_col3" class="data row3 col3" >0.7403</td>
                        <td id="T_cd7ef_row3_col4" class="data row3 col4" >0.7379</td>
                        <td id="T_cd7ef_row3_col5" class="data row3 col5" >0.5425</td>
                        <td id="T_cd7ef_row3_col6" class="data row3 col6" >0.5425</td>
            </tr>
            <tr>
                        <th id="T_cd7ef_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_cd7ef_row4_col0" class="data row4 col0" >0.7835</td>
                        <td id="T_cd7ef_row4_col1" class="data row4 col1" >0.8499</td>
                        <td id="T_cd7ef_row4_col2" class="data row4 col2" >0.7410</td>
                        <td id="T_cd7ef_row4_col3" class="data row4 col3" >0.7504</td>
                        <td id="T_cd7ef_row4_col4" class="data row4 col4" >0.7457</td>
                        <td id="T_cd7ef_row4_col5" class="data row4 col5" >0.5572</td>
                        <td id="T_cd7ef_row4_col6" class="data row4 col6" >0.5572</td>
            </tr>
            <tr>
                        <th id="T_cd7ef_level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_cd7ef_row5_col0" class="data row5 col0" >0.7796</td>
                        <td id="T_cd7ef_row5_col1" class="data row5 col1" >0.8492</td>
                        <td id="T_cd7ef_row5_col2" class="data row5 col2" >0.7373</td>
                        <td id="T_cd7ef_row5_col3" class="data row5 col3" >0.7456</td>
                        <td id="T_cd7ef_row5_col4" class="data row5 col4" >0.7414</td>
                        <td id="T_cd7ef_row5_col5" class="data row5 col5" >0.5494</td>
                        <td id="T_cd7ef_row5_col6" class="data row5 col6" >0.5495</td>
            </tr>
            <tr>
                        <th id="T_cd7ef_level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_cd7ef_row6_col0" class="data row6 col0" >0.0029</td>
                        <td id="T_cd7ef_row6_col1" class="data row6 col1" >0.0022</td>
                        <td id="T_cd7ef_row6_col2" class="data row6 col2" >0.0048</td>
                        <td id="T_cd7ef_row6_col3" class="data row6 col3" >0.0043</td>
                        <td id="T_cd7ef_row6_col4" class="data row6 col4" >0.0034</td>
                        <td id="T_cd7ef_row6_col5" class="data row6 col5" >0.0059</td>
                        <td id="T_cd7ef_row6_col6" class="data row6 col6" >0.0059</td>
            </tr>
    </tbody></table>



```python
catboost = tune_model(catboost
                     , optimize='AUC' 
                     )
```


<style  type="text/css" >
#T_27911_row5_col0,#T_27911_row5_col1,#T_27911_row5_col2,#T_27911_row5_col3,#T_27911_row5_col4,#T_27911_row5_col5,#T_27911_row5_col6{
            background:  yellow;
        }</style><table id="T_27911_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_27911_level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_27911_row0_col0" class="data row0 col0" >0.7815</td>
                        <td id="T_27911_row0_col1" class="data row0 col1" >0.8501</td>
                        <td id="T_27911_row0_col2" class="data row0 col2" >0.7455</td>
                        <td id="T_27911_row0_col3" class="data row0 col3" >0.7447</td>
                        <td id="T_27911_row0_col4" class="data row0 col4" >0.7451</td>
                        <td id="T_27911_row0_col5" class="data row0 col5" >0.5539</td>
                        <td id="T_27911_row0_col6" class="data row0 col6" >0.5539</td>
            </tr>
            <tr>
                        <th id="T_27911_level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_27911_row1_col0" class="data row1 col0" >0.7802</td>
                        <td id="T_27911_row1_col1" class="data row1 col1" >0.8490</td>
                        <td id="T_27911_row1_col2" class="data row1 col2" >0.7485</td>
                        <td id="T_27911_row1_col3" class="data row1 col3" >0.7410</td>
                        <td id="T_27911_row1_col4" class="data row1 col4" >0.7448</td>
                        <td id="T_27911_row1_col5" class="data row1 col5" >0.5518</td>
                        <td id="T_27911_row1_col6" class="data row1 col6" >0.5518</td>
            </tr>
            <tr>
                        <th id="T_27911_level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_27911_row2_col0" class="data row2 col0" >0.7861</td>
                        <td id="T_27911_row2_col1" class="data row2 col1" >0.8554</td>
                        <td id="T_27911_row2_col2" class="data row2 col2" >0.7539</td>
                        <td id="T_27911_row2_col3" class="data row2 col3" >0.7486</td>
                        <td id="T_27911_row2_col4" class="data row2 col4" >0.7512</td>
                        <td id="T_27911_row2_col5" class="data row2 col5" >0.5636</td>
                        <td id="T_27911_row2_col6" class="data row2 col6" >0.5636</td>
            </tr>
            <tr>
                        <th id="T_27911_level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_27911_row3_col0" class="data row3 col0" >0.7786</td>
                        <td id="T_27911_row3_col1" class="data row3 col1" >0.8494</td>
                        <td id="T_27911_row3_col2" class="data row3 col2" >0.7462</td>
                        <td id="T_27911_row3_col3" class="data row3 col3" >0.7395</td>
                        <td id="T_27911_row3_col4" class="data row3 col4" >0.7428</td>
                        <td id="T_27911_row3_col5" class="data row3 col5" >0.5485</td>
                        <td id="T_27911_row3_col6" class="data row3 col6" >0.5486</td>
            </tr>
            <tr>
                        <th id="T_27911_level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_27911_row4_col0" class="data row4 col0" >0.7830</td>
                        <td id="T_27911_row4_col1" class="data row4 col1" >0.8509</td>
                        <td id="T_27911_row4_col2" class="data row4 col2" >0.7482</td>
                        <td id="T_27911_row4_col3" class="data row4 col3" >0.7460</td>
                        <td id="T_27911_row4_col4" class="data row4 col4" >0.7471</td>
                        <td id="T_27911_row4_col5" class="data row4 col5" >0.5570</td>
                        <td id="T_27911_row4_col6" class="data row4 col6" >0.5570</td>
            </tr>
            <tr>
                        <th id="T_27911_level0_row5" class="row_heading level0 row5" >Mean</th>
                        <td id="T_27911_row5_col0" class="data row5 col0" >0.7819</td>
                        <td id="T_27911_row5_col1" class="data row5 col1" >0.8510</td>
                        <td id="T_27911_row5_col2" class="data row5 col2" >0.7485</td>
                        <td id="T_27911_row5_col3" class="data row5 col3" >0.7439</td>
                        <td id="T_27911_row5_col4" class="data row5 col4" >0.7462</td>
                        <td id="T_27911_row5_col5" class="data row5 col5" >0.5550</td>
                        <td id="T_27911_row5_col6" class="data row5 col6" >0.5550</td>
            </tr>
            <tr>
                        <th id="T_27911_level0_row6" class="row_heading level0 row6" >SD</th>
                        <td id="T_27911_row6_col0" class="data row6 col0" >0.0025</td>
                        <td id="T_27911_row6_col1" class="data row6 col1" >0.0023</td>
                        <td id="T_27911_row6_col2" class="data row6 col2" >0.0029</td>
                        <td id="T_27911_row6_col3" class="data row6 col3" >0.0033</td>
                        <td id="T_27911_row6_col4" class="data row6 col4" >0.0028</td>
                        <td id="T_27911_row6_col5" class="data row6 col5" >0.0051</td>
                        <td id="T_27911_row6_col6" class="data row6 col6" >0.0051</td>
            </tr>
    </tbody></table>



```python
# ada=create_model('ada')
```


```python
# ada=tune_model(ada,optimize='AUC')
```


```python
def create_submission(model, test, test_passenger_id, model_name):
    y_pred_test = model.predict_proba(test)[:, 1]
    submission = pd.DataFrame(
        {
            'PassengerId': test_passenger_id, 
            'Survived': (y_pred_test >= 0.5).astype(int),
        }
    )
    submission.to_csv(f"submission_{model_name}.csv", index=False)
    
    return y_pred_test
```


```python
test = all_df.iloc[100000:, :] #100000개~ 
X_test=test.drop(drop_list,axis=1)
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Ticket</th>
      <th>Sex</th>
      <th>Pclass</th>
      <th>Embarked</th>
      <th>Cabin_A</th>
      <th>Cabin_B</th>
      <th>Cabin_C</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100000</th>
      <td>-0.937422</td>
      <td>-0.539572</td>
      <td>0.949786</td>
      <td>10830</td>
      <td>49</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100001</th>
      <td>1.123570</td>
      <td>-0.539572</td>
      <td>-1.273379</td>
      <td>17134</td>
      <td>49</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>-0.937422</td>
      <td>-0.539572</td>
      <td>0.481059</td>
      <td>9978</td>
      <td>49</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>-0.573717</td>
      <td>-0.539572</td>
      <td>-0.563310</td>
      <td>13303</td>
      <td>49</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>-1.058657</td>
      <td>-0.539572</td>
      <td>0.125497</td>
      <td>4406</td>
      <td>49</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_pred_lightgbm = create_submission(
    lightgbm, X_test, test_df["PassengerId"], "lightgbm"
)
# test_pred_ada = create_submission(
#     ada, X_test, test_df["PassengerId"], "ada"
# )
# test_pred_gbc = create_submission(
#     ada, X_test, test_df["PassengerId"], "gbc"
# )
test_pred_catboost = create_submission(
    catboost, X_test, test_df["PassengerId"], "catboost"
)
```


```python

```


```python
test_pred_merged = (

    test_pred_lightgbm + 
    test_pred_catboost
#     test_pred_ada +
#     test_pred_gbc
)

test_pred_merged = np.round(test_pred_merged / 2)

```


```python
submission = pd.DataFrame(
    {
        'PassengerId': test_df["PassengerId"], 
        'Survived': test_pred_merged.astype(int),
    }
submission.to_csv(f"submission_merged.csv", index=False)
```


      File "<ipython-input-39-f1cf7d72db78>", line 6
        submission.to_csv(f"submission_merged.csv", index=False)
                 ^
    SyntaxError: invalid syntax
    



```python

```


```python

```


```python

```

# 
