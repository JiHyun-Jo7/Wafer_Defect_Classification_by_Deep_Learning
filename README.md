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
1. [Check Missing Value & Make New Pickle](01_preprocessing.py)

#### 1. [Load Pickle](https://www.kaggle.com/code/cchou1217/wm-811k-wafermap)
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
failureType       811457 non-null object    # 9 Types of defects
dtypes: float64(2), object(4)  
memory usage: 37.1+ MB
```
- eng
- Pickle을 불러온 후 데이터를 확인한다 사용하지 않을 데이터는 진행에 방해가 되므로 제거한다

```
def replace_value(defect):

    if defect == [['none']]:
        defect = [['Normal']]
    else: pass
    return defect

df['failureType'] = df['failureType'].apply(replace_value)
```
- eng
- 'none' 을 사용하기 위해 이름을 변경한다  
값이 2차원 행렬로 되어있어 .replace()가 적용되지 않아 함수를 만들어 사용했다

```
df['failureNum'] = df.failureType
df['trainTestNum'] = df.trianTestLabel

mapping_type = {'Normal': 0, 'Center': 1, 'Donut': 2, 'Edge-Loc': 3, 'Edge-Ring': 4, 'Loc': 5, 'Random': 6, 'Scratch': 7,
                'Near-full': 8}  # , "Nan": 9
mapping_traintest = {'Training': 0, 'Test': 1}
df = df.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})

df = df.drop(['trianTestLabel'], axis=1)
```
- Eng
- 'failureType', 'trianTestLabel' 에 라벨링을 진행한 후 사용하지 않을 'trianTestLabel'는 제거한다

```
def find_dim(x):
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return dim0, dim1

df['waferMapDim'] = df.waferMap.apply(find_dim)
```
- eng
- Wafer Map을 이용하여 Wafer의 크기를 구하고 데이터 프레임에 추가한다 (≠ Die Size)

```
print(max(df_nonpattern.waferMapDim), min(df_nonpattern.waferMapDim))  # df.waferMapDim 최대 최소값 확인
sorted_list_X = sorted(df_nonpattern.waferMapDim, key=lambda x: x[0], reverse=False)
sorted_list_Y = sorted(df_nonpattern.waferMapDim, key=lambda x: x[1], reverse=False)

ordered_set_X = list(OrderedDict.fromkeys(sorted_list_X))
ordered_set_Y = list(OrderedDict.fromkeys(sorted_list_Y))
topX_values = ordered_set_X[:10]
topY_values = ordered_set_Y[:10]
```
- Eng
- 불량 데이터를 제거하기 위해 Wafer의 크기를 확인한다  

```
index_Num_df = df.index[(df['waferMapDim'] == (15, 3)) | (df['waferMapDim'] == (18, 4)) |
                        (df['waferMapDim'] == (18, 44)) | (df['waferMapDim'] == (24, 13)) |
                        (df['waferMapDim'] == (27, 15)) | (df['waferMapDim'] == (24, 18))]

index_list_np = index_Num_np.tolist()
index_list_df = index_Num_df.tolist()
print(len(index_list_np), len(index_list_df))

df = df[~df.index.isin(index_list_df)]

df = df.reset_index()
```
- Eng
- 불량 Wafer의 크기를 사용하여 Index를 구한 뒤, .isin()로 해당 Wafer를 DateFrame에서 제거한다  
제거된 Index는 이후 과정에서 문제가 되므로 .reset_index() 과정을 거쳐준다

```
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(10, 10))
ax = ax.ravel(order='C')
for i in range(0, 20):
    img = df_withpattern.waferMap[i]
    ax[i].imshow(img)
    print(df_withpattern.failureType[i])
    ax[i].set_title(df_withpattern.failureType[i][0][0], fontsize=10)
    ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()
```
- Image
```
with open('./datasets/LSWMD_CleanData.pickle', 'wb') as f:
    pickle.dump(df, f)
```
- Eng
- 불필요한 데이터가 제거된 데이터 프레임은 이후 빠른 작업을 위해 새로운 Pickle 파일로 저장한다
---


2. [Data Preprocessing](02_preprocessing.py)

#### 1. Title