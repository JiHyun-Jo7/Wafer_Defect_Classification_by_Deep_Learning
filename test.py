from sklearn.preprocessing import StandardScaler
from matplotlib import gridspec
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import warnings
import sys

warnings.filterwarnings("ignore")  # 경고문 출력 제거
np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
pd.set_option('display.max_columns', None)
df = pd.read_pickle("./datasets/LSWMD.pkl")  # 피클에 있는 데이터 프레임 read

# df.info()
# waferMap          웨이퍼 사진 (0 : 공백 | 1 : 웨이퍼 | 2 : 결함)
# dieSize           다이싱 사이즈
# lotName           웨이퍼 1묶음의 번호
# waferIndex        웨이퍼 1묶음 내 번호
# trainTestLabel    훈련, 테스트 라벨
# failureType       웨이퍼 결함 타입

# 사용하지 않는 데이터 제거
df = df.drop(['waferIndex', 'dieSize', 'lotName'], axis=1)
# df.info()

def find_dim(x):
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return dim0, dim1


# dataframe 에 wafer size 추가
df['waferMapDim'] = df.waferMap.apply(find_dim)
# print(max(df.waferMapDim), min(df.waferMapDim))  # df.waferMapDim 최대 최소값 확인

# failureType Data Labeling

df.loc[df['failureType'].str.len() == 0, "failureType"] = np.nan
# df['failureType'] = df['failureType'].fillna("Nan")

df['failureNum'] = df.failureType
df['trainTestNum'] = df.trianTestLabel

mapping_type = {'Center': 1, 'Donut': 2, 'Edge-Loc': 3, 'Edge-Ring': 4, 'Loc': 5, 'Random': 6, 'Scratch': 7,
                'Near-full': 8, 'none': 0}  # , "Nan": 9
mapping_traintest = {'Training': 0, 'Test': 1}
df = df.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})

df = df.drop(['trianTestLabel'], axis=1)


# 'none' -> 'Normal'
def replace_value(defect):
    # 만약 cell_value가 [['none']]이면 [['Normal']]로 변경
    return [['Normal']] if defect == [['none']] else defect


# 'failureType' 열에 대해 사용자 정의 함수를 적용하여 값 변경
df['failureType'] = df['failureType'].apply(replace_value)

print(df['failureType'].value_counts())

# 분류된 데이터
df_withlabel = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 8)]
df_withlabel = df_withlabel.reset_index()
# 결함의 종류가 명확한 데이터
df_withpattern = df[(df['failureNum'] >= 1) & (df['failureNum'] <= 8)]
df_withpattern = df_withpattern.reset_index()
# 결함의 종류가 불명확한 데이터
df_nonpattern = df[(df['failureNum'] == 0)]
df_nonpattern = df_nonpattern.reset_index()
# 분류 없음
# df_Nan = df[(df['failureNum'] == 9)]
# df_Nan = df_Nan.reset_index()

# x, y < 25 인 wafer 제거
print(max(df_nonpattern.waferMapDim), min(df_nonpattern.waferMapDim))  # df.waferMapDim 최대 최소값 확인
sorted_list_X = sorted(df_nonpattern.waferMapDim, key=lambda x: x[0], reverse=False)
sorted_list_Y = sorted(df_nonpattern.waferMapDim, key=lambda x: x[1], reverse=False)

ordered_set_X = list(OrderedDict.fromkeys(sorted_list_X))
ordered_set_Y = list(OrderedDict.fromkeys(sorted_list_Y))
topX_values = ordered_set_X[:10]
topY_values = ordered_set_Y[:10]
print('minX:', topX_values)
print('minY:', topY_values)

# # 가로 세로 비가 50 이하인 wafer 제거
# filtered_half_X = [t for t in df_nonpattern.waferMapDim if 0.0 <= t[0] / t[1] <= 0.50]
# filtered_half_Y = [t for t in df_nonpattern.waferMapDim if 0.0 <= t[1] / t[0] <= 0.50]
# less50_X = list(OrderedDict.fromkeys(filtered_half_X))
# less50_Y = list(OrderedDict.fromkeys(filtered_half_Y))
# print('less X/Y 50%:', less50_X)
# print('less Y/X 50%:', less50_Y)

index_Num_np = df_nonpattern.index[
    (df_nonpattern['waferMapDim'] == (15, 3)) | (df_nonpattern['waferMapDim'] == (18, 4)) |
    (df_nonpattern['waferMapDim'] == (18, 44)) | (df_nonpattern['waferMapDim'] == (24, 13)) |
    (df_nonpattern['waferMapDim'] == (27, 15)) | (df_nonpattern['waferMapDim'] == (24, 18))]
index_Num_df = df.index[(df['waferMapDim'] == (15, 3)) | (df['waferMapDim'] == (18, 4)) |
                        (df['waferMapDim'] == (18, 44)) | (df['waferMapDim'] == (24, 13)) |
                        (df['waferMapDim'] == (27, 15)) | (df['waferMapDim'] == (24, 18))]

index_list_np = index_Num_np.tolist()
index_list_df = index_Num_df.tolist()
print(len(index_list_np), len(index_list_df))
df_nonpattern = df_nonpattern[~df_nonpattern.index.isin(index_list_np)]
df = df[~df.index.isin(index_list_df)]
df_nonpattern = df_nonpattern.reset_index()


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

df.dropna(inplace=True)
df.info()
print(df.head())

print(df['failureType'].value_counts())

with open('./datasets/LSWMD_CleanData.pickle', 'wb') as f:
    pickle.dump(df, f)
print('save success')
