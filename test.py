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
print(max(df.waferMapDim), min(df.waferMapDim))  # df.waferMapDim 최대 최소값 확인
sorted_list_X = sorted(df.waferMapDim, key=lambda x: x[0], reverse=False)
sorted_list_Y = sorted(df.waferMapDim, key=lambda x: x[1], reverse=False)
topX_values = sorted_list_X[:50]
topY_values = sorted_list_Y[:50]
print(topX_values)
print(topY_values)

index_Num = df.index[df['waferMapDim'] == (10, 150)]
# print(index_Num[:100])
idx = [266785, 266786, 266787, 266788, 266789, 266790, 266791, 266792,
        266793, 266794, 266795, 266796, 266797, 266798, 266799, 266800,
        266801, 266802, 266803, 266804, 266805, 266806, 266807, 266808,
        266809, 266835, 266836, 266837, 266838, 266839, 266840, 266841,
        266842, 266843, 266844, 266845, 266846, 266847, 266848, 266849,
        266850, 266851, 266852, 266853, 266854, 266855, 266856, 266857,
        266858, 266859, 266860, 266861, 266862, 266863, 266864, 266865,
        266866, 266867, 266868, 266869, 266870, 266871, 266872, 266873,
        266874, 266875, 266876, 266877, 266878, 266879, 266880, 266881,
        266882, 266883, 266884, 277186, 277187, 277188, 277189, 277190,
        277191, 277192, 277193, 277194, 277195, 277196, 277197, 277198,
        277199, 277200, 277201, 277202, 277203, 277204, 277205, 277206,
        277207, 277208, 277209, 277210]
test = df['failureType'].iloc[idx]
print(test)
exit()

# failureType Data Labeling

df.loc[df['failureType'].str.len() == 0, "failureType"] = np.nan
df['failureType'] = df['failureType'].fillna("Non")

df['failureNum'] = df.failureType
df['trainTestNum'] = df.trianTestLabel

mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6,
                'Near-full': 7, 'none': 8, "Non": 9}
mapping_traintest = {'Training': 0, 'Test': 1}
df = df.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})

df = df.drop(['trianTestLabel'], axis=1)

df_withlabel = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 8)]
df_withlabel = df_withlabel.reset_index()
# 결함의 종류가 명확한 데이터
df_withpattern = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 7)]
df_withpattern = df_withpattern.reset_index()
# 결함의 종류가 불명확한 데이터
df_nonpattern = df[(df['failureNum'] == 8)]
df_nonpattern = df_nonpattern.reset_index()
# 분류 없음
df_Nan = df[(df['failureNum'] == 9)]
df_Nan = df_Nan.reset_index()

index_Num = df_nonpattern.index[df_nonpattern['waferMapDim'] == (6, 21)]
print(index_Num)

fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))
ax = ax.ravel(order='C')

for i in range(0, 100):
    img = df_nonpattern.waferMap[i+17283]
    ax[i].imshow(img)
    ax[i].set_title(df_nonpattern.failureType[i][0][0], fontsize=10)
    ax[i].set_xlabel(df_nonpattern.index[i], fontsize=8)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()
# df.info()
# print(df.head())


# 0. Center': 4294, 1. 'Donut': 555, 2. 'Edge-Loc': 5189, 3. 'Edge-Ring': 9680, 4. 'Loc': 3593,
# 5. 'Random': 866, 6. 'Scratch': 1193, 7. 'Near-full': 149, 8. 'none': 147431, 9. 'Non': 638507

# with open('./models/wafer_defect.pickle', 'wb') as f:
#     pickle.dump(df, f)
