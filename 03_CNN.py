import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import warnings

warnings.filterwarnings("ignore")  # 경고문 출력 제거
np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
pd.set_option('display.max_columns', None)

# 주어진 DataFrame
df = pd.read_pickle("./datasets/LSWMD_value_counts.pickle")
filtered_df = df[df['failureType'] != 'Normal']
filtered_df.reset_index(drop=True, inplace=True)
filtered_df.info()

# 'failureNum'을 'failureType'으로 매핑
label_mapping = dict(zip(filtered_df['failureNum'], filtered_df['failureType']))

# 훈련, 테스트 데이터 분리 전 웨이퍼 최대 크기 구하기
max_wafer_size = max(filtered_df['waferMapDim'])
print('max:')
print(max(filtered_df.waferMapDim))


# 이미지 크기를 통일시키는 함수
def resize_wafer_map(wafer_map, target_size, resample_method=Image.BILINEAR):
    try:
        # Numpy 배열을 이미지로 변환
        image_array = np.array(wafer_map)
        image = Image.fromarray(image_array.astype('uint8'))  # Numpy 배열을 이미지로 변환

        # 이미지 리사이징
        resized_image = image.resize((int(target_size[1]), int(target_size[0])), resample=resample_method)

        return np.array(resized_image)
    except Exception as e:
        print(f"Error in resizing: {e}")
        return None


# 예시: 최대 크기 기준으로 통일된 크기 설정
target_size = (max_wafer_size[1], max_wafer_size[0])
print('\ntarget_size:')
print(target_size)

# 'resized_waferMap' 열에 리사이즈된 데이터 추가
filtered_df['resized_waferMap'] = filtered_df['waferMap'].apply(lambda x: resize_wafer_map(x, target_size))
print('\nresize waferMap :')
print(filtered_df.resized_waferMap[0])

# 데이터를 훈련 및 테스트 세트로 분할
X_train, X_test, Y_train, Y_test = train_test_split(filtered_df['resized_waferMap'], filtered_df['failureNum'], test_size=0.2, random_state=42)

# 훈련 데이터 및 레이블 확인
print("\n훈련 데이터 형태:")
print(X_train.shape, Y_train.shape)

# 테스트 데이터 및 레이블 확인
print("\n테스트 데이터 형태:")
print(X_test.shape, Y_test.shape)

# 이미지 플로팅 및 레이블 확인
my_sample = np.random.randint(len(X_train))
plt.imshow(X_train.iloc[my_sample], cmap='gray')  # cmap = 'gray': 흑백 처리
plt.show()

# 레이블 확인
original_label = Y_train.iloc[my_sample]
print("\n원래 레이블:", original_label)

# 매핑된 레이블 확인
mapped_label = label_mapping[original_label]
print("\n매핑된 레이블:", mapped_label)

# 레이블을 원-핫 인코딩
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# 원-핫 인코딩된 레이블 확인
print("\n원-핫 인코딩된 레이블:")
print(Y_train[my_sample])

labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

# 'labels' 리스트를 사용하여 레이블 출력
print("\ntag:", labels[original_label])
print(X_train.iloc[my_sample])  # 픽셀 값. 0~255 밝을수록 값이 커짐
print('type: ', type(X_train.iloc[my_sample]))

x_train = X_train.values / 2    # max(x_train) = 2
x_test = X_test.values / 2      # max(x_test) = 2

print('\nx_train.shape:')
print(x_train.shape)
print("\nx_test.shape:")
print(x_test.shape)

x_dim, y_dim = target_size
x_train = x_train.reshape(len(X_train), x_dim, y_dim, 1)
x_test = x_test.reshape(-1, x_dim, y_dim, 1)
