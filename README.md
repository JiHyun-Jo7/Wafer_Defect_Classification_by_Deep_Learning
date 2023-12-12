# 🌑🔍Wafer_Defect_Classification_by_Deep_Learning
### 딥러닝을 활용한 웨이퍼 자동 결함 시스템
- Gahee Jung: 데이터 학습 모델
- Jihyun Jo: 데이터 전처리
---
## 🖥️ 개발 환경(IDE)
- Win 10 & 11
- Pycharm 2023.2
- Python 3.8.1
- Kaggle
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
- Load [Pickle](https://www.kaggle.com/code/cchou1217/wm-811k-wafermap/input) and check the data Remove unused data that interferes with progress
- [Pickle](https://www.kaggle.com/code/cchou1217/wm-811k-wafermap/input)을 불러오고 데이터를 확인한다 사용하지 않을 데이터는 진행에 방해가 되므로 제거한다

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
- 'failureType', 'trianTestLabel' 에 라벨링한 후 사용하지 않을 'trianTestLabel'는 제거한다

```
def find_dim(x):
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return dim0, dim1

df['waferMapDim'] = df.waferMap.apply(find_dim)
```
- Retrieve the wafer size from the Wafer Map and incorporate it into the data frame (≠ Die Size)
- Wafer Map을 이용하여 웨이퍼퍼의 크기를 구하고 데이터 프레임에 추가한다 (≠ Die Size)

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
Identify the index corresponding to the waferMapDim
- 불량 데이터를 제거하기 위해 특정 크기의 웨이퍼를 걸러낸다  
waferMapDim를 사용하여 해당 웨이퍼의 인덱스를 특정한다
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
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
```
- Remove the wafer from the DataFrame using .isin()  
Utilize reset_index() to address issues caused by the removed indexes
- .isin()로 해당 웨이퍼를 DateFrame에서 제거한다  
제거한 인덱스는 이후 과정에서 문제가 되므로 .reset_index()를 사용한다

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

![defect](https://github.com/JiHyun-Jo7/CV2/assets/141097551/03526192-1c9c-45a7-8572-99dff0114115)

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

<details>
	<summary>Result</summary>
  	<div markdown="1">

```
x = []	 # x = [115, 9477, 20256, 6550, 6874, 6666, 7138, 8364, 109228]
labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
for label in labels:
    idx = df[df['failureType'] == label].index
    x.append(idx[0])
```
- 
- 결함에 따른 특징을 구하기 위해 결함 별 웨이퍼의 인덱스를 구하고 그 값을 리스트에 저장한다
```
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
ax = ax.ravel(order='C')
for i in range(9):
    img = df.waferMap[x[i]]
    ax[i].imshow(img)
    ax[i].set_title(df.failureType[x[i]], fontsize=10)
    ax[i].set_xlabel(df.index[x[i]], fontsize=8)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()
```

![00](https://github.com/JiHyun-Jo7/CV2/assets/141097551/eb596771-6f76-4537-87e8-c736bbd52708)

```
def cal_den(x):
    return 100 * (np.sum(x == 2) / np.size(x))

def find_regions(x):
    rows = np.size(x, axis=0)
    cols = np.size(x, axis=1)

    ind1 = np.arange(0, rows, rows // 5)
    ind2 = np.arange(0, cols, cols // 5)

    reg1 = x[ind1[0]:ind1[1], :]
    reg3 = x[ind1[4]:, :]
    reg4 = x[:, ind2[0]:ind2[1]]
    reg2 = x[:, ind2[4]:]

    reg5 = x[ind1[1]:ind1[2], ind2[1]:ind2[2]]
    reg6 = x[ind1[1]:ind1[2], ind2[2]:ind2[3]]
    reg7 = x[ind1[1]:ind1[2], ind2[3]:ind2[4]]
    reg8 = x[ind1[2]:ind1[3], ind2[1]:ind2[2]]
    reg9 = x[ind1[2]:ind1[3], ind2[2]:ind2[3]]
    reg10 = x[ind1[2]:ind1[3], ind2[3]:ind2[4]]
    reg11 = x[ind1[3]:ind1[4], ind2[1]:ind2[2]]
    reg12 = x[ind1[3]:ind1[4], ind2[2]:ind2[3]]
    reg13 = x[ind1[3]:ind1[4], ind2[3]:ind2[4]]

    fea_reg_den = [cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4), cal_den(reg5), cal_den(reg6),
                   cal_den(reg7), cal_den(reg8), cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12),
                   cal_den(reg13)]
    return fea_reg_den

df['fea_reg'] = df.waferMap.apply(find_regions)
```
-
- 웨이퍼에 구역을 나누고 구역 별 결함 밀도를 구한다
```
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
ax = ax.ravel(order='C')
for i in range(9):
    ax[i].bar(np.linspace(1, 13, 13), df.fea_reg[x[i]])
    ax[i].set_title(df.failureType[x[i]], fontsize=15)
    ax[i].set_xticklabels(labels)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()
```
![05](https://github.com/JiHyun-Jo7/CV2/assets/141097551/0ea58264-fe50-4850-b6d9-b6f979e27d50)
```
def change_val(img):
    img[img==1] =0  
    return img

df_copy = df.copy()
df_copy['new_waferMap'] =df_copy.waferMap.apply(change_val)
```
-
- df를 복제한 뒤, waferMap 데이터에서 웨이퍼 기판에 해당하는 1을 0으로 변환한다
```
fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize=(15, 15))
ax = ax.ravel(order='C')
for i in range(9):
    img = df_copy.waferMap[x[i]]
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)    
      
    ax[i].imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
    ax[i].set_title(df_copy.failureType[x[i]],fontsize=15)
    ax[i].set_xticks([])
plt.tight_layout()
plt.show() 
```
![01](https://github.com/JiHyun-Jo7/CV2/assets/141097551/bda1081d-d22e-4d54-9e5c-6a5bd347017f)
```
def cubic_inter_mean(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis = 1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    y = xMean_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew

df_copy['fea_cub_mean'] =df_copy.waferMap.apply(cubic_inter_mean)
```
-
- waferMap의 라돈 변환(sinogram)을 생성한다  
sinogram의 각 투영에 대한 평균값을 보간하여 부드러운 곡선을 얻고 이를 100으로 나누어서 새로운 특성 열을 생성한다
```
fig, ax = plt.subplots(nrows = 3, ncols = 3,figsize=(15, 15))
ax = ax.ravel(order='C')
for i in range(9):
    ax[i].bar(np.linspace(1,20,20),df_copy.fea_cub_mean[x[i]])
    ax[i].set_title(df_copy.failureType[x[i]],fontsize=10)
    ax[i].set_xticks([])
    ax[i].set_xlim([0,21])   
    ax[i].set_ylim([0,1])
plt.tight_layout()
plt.show()
```
- fea_cub_mean
![02](https://github.com/JiHyun-Jo7/CV2/assets/141097551/ffa372de-fe9c-478e-b100-e400ed9eed42)  

- fea_cub_std
![03](https://github.com/JiHyun-Jo7/CV2/assets/141097551/8825c5b1-7d3f-4b6f-852a-fb5c04b16116)
```
fig, ax = plt.subplots(nrows = 3, ncols = 3,figsize=(15, 15))
ax = ax.ravel(order='C')
for i in range(9):
    img = df_copy.waferMap[x[i]]
    zero_img = np.zeros(img.shape)
    img_labels = measure.label(img, connectivity=1, background=0)
    img_labels = img_labels -1
    if img_labels.max()==0:
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels>-1], axis = None)
        no_region = info_region[0]
    
    zero_img[np.where(img_labels==no_region)] = 2
    ax[i].imshow(zero_img)
    ax[i].set_title(df_copy.failureType[x[i]],fontsize=10)
    ax[i].set_xticks([])
plt.tight_layout()
plt.show() 
```
![04](https://github.com/JiHyun-Jo7/CV2/assets/141097551/d5deadac-06b7-4610-995a-48a0a3b22a11)
```
def cal_dist(img,x,y):
    dim0=np.size(img,axis=0)    
    dim1=np.size(img,axis=1)
    dist = np.sqrt((x-dim0/2)**2+(y-dim1/2)**2)
    return dist  

def fea_geom(img):
    norm_area=img.shape[0]*img.shape[1]
    norm_perimeter=np.sqrt((img.shape[0])**2+(img.shape[1])**2)
    
    img_labels = measure.label(img, connectivity=1, background=0)

    if img_labels.max()==0:
        img_labels[img_labels==0]=1
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels>0], axis = None)
        no_region = info_region[0][0]-1       
    
    prop = measure.regionprops(img_labels)
    prop_area = prop[no_region].area/norm_area
    prop_perimeter = prop[no_region].perimeter/norm_perimeter 
    
    prop_cent = prop[no_region].local_centroid 
    prop_cent = cal_dist(img,prop_cent[0],prop_cent[1])
    
    prop_majaxis = prop[no_region].major_axis_length/norm_perimeter 
    prop_minaxis = prop[no_region].minor_axis_length/norm_perimeter  
    prop_ecc = prop[no_region].eccentricity  
    prop_solidity = prop[no_region].solidity  
    
    return prop_area,prop_perimeter,prop_majaxis,prop_minaxis,prop_ecc,prop_solidity

df_copy['fea_geom'] =df_copy.waferMap.apply(fea_geom)
```

   </div>
</details>
