from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
df['failureType'] = df['failureType'].fillna("Nan")

df['failureNum'] = df.failureType
df['trainTestNum'] = df.trianTestLabel

mapping_type = {'Center': 1, 'Donut': 2, 'Edge-Loc': 3, 'Edge-Ring': 4, 'Loc': 5, 'Random': 6, 'Scratch': 7,
                'Near-full': 8, 'none': 0, "Nan": 9}
mapping_traintest = {'Training': 0, 'Test': 1}
df = df.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})

df = df.drop(['trianTestLabel'], axis=1)

df_withlabel = df[(df['failureNum'] >= 1) & (df['failureNum'] <= 9)]
df_withlabel = df_withlabel.reset_index()
# 결함의 종류가 명확한 데이터
df_withpattern = df[(df['failureNum'] >= 1) & (df['failureNum'] <= 8)]
df_withpattern = df_withpattern.reset_index()
# 결함의 종류가 불명확한 데이터
df_nonpattern = df[(df['failureNum'] == 0)]
df_nonpattern = df_nonpattern.reset_index()
# 분류 없음
df_Nan = df[(df['failureNum'] == 9)]
df_Nan = df_Nan.reset_index()

from collections import OrderedDict

print(max(df_nonpattern.waferMapDim), min(df_nonpattern.waferMapDim))  # df.waferMapDim 최대 최소값 확인
sorted_list_X = sorted(df_nonpattern.waferMapDim, key=lambda x: x[0], reverse=False)
sorted_list_Y = sorted(df_nonpattern.waferMapDim, key=lambda x: x[1], reverse=False)

ordered_set = list(OrderedDict.fromkeys(sorted_list_X))

topX_values = ordered_set[:50]
topY_values = ordered_set[:50]
print(topX_values)
print(topY_values)

index_Num = df_nonpattern.index[(df_nonpattern['waferMapDim'] == (15, 3)) |
                                (df_nonpattern['waferMapDim'] == (18, 4)) |
                                (df_nonpattern['waferMapDim'] == (18, 44)) |
                                (df_nonpattern['waferMapDim'] == (24, 13)) |
                                (df_nonpattern['waferMapDim'] == (22, 50)) |
                                (df_nonpattern['waferMapDim'] == (27, 15)) |
                                (df_nonpattern['waferMapDim'] == (24, 18))]

index_list = index_Num.tolist()
df.test = df_nonpattern['failureType'].iloc[index_list]
print(df.test)
print(len(df.test))

fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))
ax = ax.ravel(order='C')

for i in range(0, 100):
    img = df_nonpattern.waferMap[index_Num[i]]
    ax[i].imshow(img)
    ax[i].set_title(df_nonpattern.failureType[index_Num[i]][0][0], fontsize=10)
    ax[i].set_xlabel(df_nonpattern.index[index_Num[i]], fontsize=8)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()
# df.info()
# print(df.head())


# 1. Center': 4294, 2. 'Donut': 555, 3. 'Edge-Loc': 5189, 4. 'Edge-Ring': 9680, 5. 'Loc': 3593,
# 6. 'Random': 866, 7. 'Scratch': 1193, 8. 'Near-full': 149, 0. 'none': 147431, 9. 'Nan': 638507

# with open('./models/wafer_defect.pickle', 'wb') as f:
#     pickle.dump(df, f)
