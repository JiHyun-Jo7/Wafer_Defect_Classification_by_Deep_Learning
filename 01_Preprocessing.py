from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pickle
import warnings
import sys

warnings.filterwarnings("ignore")   # 경고문 출력 제거
np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
df = pd.read_pickle("./input/LSWMD.pkl")  # 피클에 있는 데이터 프레임 read
df.info()
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

uni_waferDim = np.unique(df.waferMapDim, return_counts=True)
print(uni_waferDim[0].shape[0])

# str 데이터 -> float 데이터
df['failureNum'] = df.failureType
df['trainTestNum'] = df.trianTestLabel
mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6,
                'Near-full': 7, 'none': 8}
mapping_traintest = {'Training': 0, 'Test': 1}
df = df.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})

df = df[df['failureNum'].isin([0, 1, 2, 3, 4, 5, 6, 7])]
df.info()

scaler = StandardScaler()

# 데이터 표준화
standardized_data = scaler.fit_transform(df.waferMapDim)

# 표준화된 데이터 출력
standardized_df = pd.DataFrame(standardized_data, columns=df.columns)
print(standardized_df)

# with open('./models/wafer_defect.pickle', 'wb') as f:
#     pickle.dump(df, f)
