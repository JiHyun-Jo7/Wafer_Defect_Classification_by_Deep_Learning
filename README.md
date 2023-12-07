# 🌑🔍Wafer_Defect_Classification_by_Deep_Learning
### 딥러닝을 활용한 웨이퍼 자동 결함 시스템
- Gahee Jung: 데이터 학습 모델
- Jihyun Jo: 데이터 전처리
---
## 🖥️ 개발 환경(IDE)
- Win 10 & 11
- Pycharm 2023.2
- Python 3.8.1
---
## Preprocessing 
### 1. [Check Missing Value & Make New Pickle](01_preprocessing.py)

<details>
	<summary>Result</summary>
  	<div markdown="1">

```
df = pd.read_pickle("./datasets/LSWMD.pkl")
df.info()
---
<class 'pandas.core.frame.DataFrame'>  
RangeIndex: 811457 entries, 0 to 811456  
Data columns (total 6 columns):  
waferMap          811457 non-null object    # 0: none / 1: wafer / 2: defect 
dieSize           811457 non-null float64
lotName           811457 non-null object    # one lot has 25 wafers
waferIndex        811457 non-null float64   # 1 ~ 25
trianTestLabel    811457 non-null object
failureType       811457 non-null object    # 9 Types of defects + NaN
dtypes: float64(2), object(4)  
memory usage: 37.1+ MB
```
- [Load Pickle](https://www.kaggle.com/code/cchou1217/wm-811k-wafermap) and check the data Remove unused data that interferes with progress
- [Pickle](https://www.kaggle.com/code/cchou1217/wm-811k-wafermap)을 불러오고 데이터를 확인한다 사용하지 않을 데이터는 진행에 방해가 되므로 제거한다

```
def replace_value(defect):
    if defect == [['none']]:
        defect = 'Normal'
    else:
        defect = defect[0][0]
    return defect

df['failureType'] = df['failureType'].apply(replace_value)
```
- The 'failureType' column in the DataFrame,   
represented as a two-dimensional list, will be simplified for easier access  
Additionally, the label 'none' will be renamed for greater convenience in future use
- df['failureType']는 2차원 리스트 데이터들로 구성되어있다    
이 값들에 쉽게 접급하기 위해 2차원 리스트를 제거한다  
또한 이후 사용할 'none' 데이터의 이름을 구별하기 쉽도록 변경한다  

```
df['failureNum'] = df.failureType
df['trainTestNum'] = df.trianTestLabel

mapping_type = {'Normal': 0, 'Center': 1, 'Donut': 2, 'Edge-Loc': 3, 'Edge-Ring': 4, 'Loc': 5, 'Random': 6, 'Scratch': 7,
                'Near-full': 8}
mapping_traintest = {'Training': 0, 'Test': 1}
df = df.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})
df = df.drop(['trianTestLabel'], axis=1)
```
- Label 'failureType', 'trianTestLabel' and remove 'trianTestLabel'
- 'failureType', 'trianTestLabel' 에 라벨링을 진행한 후 사용하지 않을 'trianTestLabel'는 제거한다

```
def find_dim(x):
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return dim0, dim1

df['waferMapDim'] = df.waferMap.apply(find_dim)
```
- Retrieve the wafer size from the Wafer Map and incorporate it into the data frame (≠ Die Size)
- Wafer Map을 이용하여 Wafer의 크기를 구하고 데이터 프레임에 추가한다 (≠ Die Size)

```
sorted_list_X = sorted(df.waferMapDim, key=lambda x: x[0], reverse=False)
sorted_list_Y = sorted(df.waferMapDim, key=lambda x: x[1], reverse=False)

ordered_set_X = list(OrderedDict.fromkeys(sorted_list_X))
ordered_set_Y = list(OrderedDict.fromkeys(sorted_list_Y))

topX_values = ordered_set_X[:10]
topY_values = ordered_set_Y[:10]

index_Num = df.index[(df['waferMapDim'] == (15, 3)) | (df['waferMapDim'] == (18, 4)) |
                        (df['waferMapDim'] == (18, 44)) | (df['waferMapDim'] == (24, 13)) |
                        (df['waferMapDim'] == (27, 15)) | (df['waferMapDim'] == (24, 18))]

index_list = index_Num.tolist()
```
- Filter out specific sizes to eliminate errors  
Identify the index corresponding to the wafer size
- 불량 데이터를 제거하기 위해 특정 크기의 Wafer를 걸러낸다  
Wafer 크기를 사용하여 인덱스를 특정한다
```
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))
ax = ax.ravel(order='C')

for i in range(6):
    idx = index_list[i]
    img = df.waferMap[idx]
    ax[i].imshow(img)
    ax[i].set_title(df.failureType[idx], fontsize=10)
    ax[i].set_xlabel(df.index[idx], fontsize=8)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()
```

![error](https://github.com/JiHyun-Jo7/CV2/assets/141097551/20cfc3b1-f463-446b-bb9d-09bd9c912b81)

```
df = df[~df.index.isin(index_list)]
df = df.reset_index()
```
- Remove the wafer from the DataFrame using .isin()  
Utilize reset_index() to address issues caused by the removed indexes
- .isin()로 해당 Wafer를 DateFrame에서 제거한다  
제거한 인덱스는 이후 과정에서 문제가 되므로 .reset_index() 과정을 거쳐준다

```
df_withlabel = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 8)]
df_withlabel = df_withlabel.drop("level_0", axis=1).reset_index(drop=True)

df_withpattern = df[(df['failureNum'] >= 1) & (df['failureNum'] <= 8)]
df_withpattern = df_withpattern.drop("level_0", axis=1).reset_index(drop=True)

df_nonpattern = df[(df['failureNum'] == 0)]
df_nonpattern = df_nonpattern.drop("level_0", axis=1).reset_index(drop=True)
```
- Arrange the labels to examine wafer images
- 웨이퍼 이미지를 살펴보기 위해 라벨을 정렬한다
```
fig, ax = plt.subplots(nrows=4, ncols=10, figsize=(10, 10))
ax = ax.ravel(order='C')
for i in range(0, 40):
    img = df_withpattern.waferMap[i]
    ax[i].imshow(img)
    print(df_withpattern.failureType[i])
    ax[i].set_title(df_withpattern.failureType[i], fontsize=10)
    ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()
```

![defect](https://github.com/JiHyun-Jo7/CV2/assets/141097551/f8efd5ad-9b96-49c9-b7d3-368ed93b3b62)

```
with open('./datasets/LSWMD_CleanData.pickle', 'wb') as f:
    pickle.dump(df, f)
```
- Afterwards, save the data frame with irrelevant data removed as a new pickle file for more efficient processing
- 불필요한 데이터가 제거된 데이터 프레임은 이후 빠른 작업을 위해 새로운 Pickle로 저장한다

   </div>
</details>

---
### 2. [Data Preprocessing](02_preprocessing.py)

#### 1. Title