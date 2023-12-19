import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statistics import median
import time
import sys
import warnings

warnings.filterwarnings("ignore")  # 경고문 출력 제거
np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
pd.set_option('display.max_columns', None)

df=pd.read_pickle("./datasets/LSWMD_withpt.pkl")
df.info()
# print(df.failureType.value_counts())

# LabelEncoder를 사용하여 문자열 레이블을 숫자로 인코딩
label_encoder = LabelEncoder()
df['encoded_labels'] = label_encoder.fit_transform(df['failureType'])
print(df[['failureType', 'encoded_labels']].head())

# 'failureNum'을 'failureType'으로 매핑
label_mapping = dict(zip(df['failureNum'], df['failureType']))

# 각 행의 x 좌표와 y 좌표를 추출하여 새로운 열을 만듭니다.
df['x'] = df['waferMapDim'].apply(lambda x: x[0])
df['y'] = df['waferMapDim'].apply(lambda x: x[1])

# 각 차원에 대한 중앙값 계산
median_x = df['x'].median()
median_y = df['y'].median()

# target_size = (median_x, median_y)
target_size = (38, 38)
print("Median :", target_size)

df.drop(['x', 'y'], axis=1, inplace=True)
df.info()


# 이미지 크기를 통일시키는 함수
def resize_wafer_map(wafer_map, target_size, resample_method=Image.BILINEAR):
    try:
        global cnt  # 함수 외부의 cnt 변수를 사용하겠다고 선언
        # Numpy 배열을 이미지로 변환
        image_array = np.array(wafer_map)
        image = Image.fromarray(image_array.astype('uint8'))  # Numpy 배열을 이미지로 변환

        # 이미지 리사이징
        resized_image = image.resize((int(target_size[1]), int(target_size[0])), resample=resample_method)

        return np.array(resized_image)
    except Exception as e:
        print(f"Error in resizing: {e}")
        return None


# 'resized_waferMap' 열에 리사이즈된 데이터 추가
df['resized_waferMap'] = df['waferMap'].apply(lambda x: resize_wafer_map(x, target_size))
print('finish resizing image')

plt.subplot(1, 2, 1)
plt.title("Original WaferMap")
plt.imshow(df.waferMap[0], cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Resized WaferMap")
plt.imshow(df.resized_waferMap[0], cmap='gray')
plt.show()

# 데이터를 훈련 및 테스트 세트로 분할
X_train, X_test, Y_train, Y_test = train_test_split(np.array(df['resized_waferMap'].tolist()),
                                                    df['failureNum'], test_size=0.2, random_state=42)
# 훈련 데이터 및 레이블 확인
print("\n훈련 데이터 형태:")
print(X_train.shape, Y_train.shape)

# 테스트 데이터 및 레이블 확인
print("\n테스트 데이터 형태:")
print(X_test.shape, Y_test.shape)

# 이미지 플로팅 및 레이블 확인
my_sample = np.random.randint(len(X_train))
plt.imshow(X_train[my_sample], cmap='gray')  # cmap = 'gray': 흑백 처리
plt.show()

# 레이블을 원-핫 인코딩
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

# 레이블 확인
original_label = np.argmax(y_train[my_sample])
print("\n원래 레이블:", original_label)

# 매핑된 레이블 확인
mapped_label = label_mapping[original_label]
print("\n매핑된 레이블:", mapped_label)

# 원-핫 인코딩된 레이블 확인
print("\n원-핫 인코딩된 레이블:")
print(y_train[my_sample])

# 'labels' 리스트를 사용하여 레이블 출력
labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
print("\ntag:", labels[original_label])

# print(X_train.iloc[my_sample])  # 픽셀 값. 0~255 밝을수록 값이 커짐
print('type: ', type(X_train[my_sample]))

x_train = X_train / 2  # max(x_train) = 2
x_test = X_test / 2  # max(x_test) = 2

x_dim, y_dim = target_size
x_train = x_train.reshape(len(X_train), x_dim, y_dim, 1)
x_test = x_test.reshape(-1, x_dim, y_dim, 1)

print('\nx_train.shape:')
print(x_train.shape)
print("\nx_test.shape:")
print(x_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', padding='same', input_shape=(x_dim, y_dim, 1)))
model.add(MaxPool2D(padding='same', pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(padding='same', pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))  # 예측 값 (카테고리 수)
model.summary()

model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

fit_hist = model.fit(x_train, y_train, batch_size=128,
                     epochs=20, validation_split=0.2, verbose=1)

val_acc = round(fit_hist.history['val_accuracy'][-1], 3)
model.save('./models/CNN_{}.h5'.format(val_acc))

score = model.evaluate(x_test, y_test, verbose=0)
print('Final test set accuracy', score[1])

plt.plot(fit_hist.history['accuracy'])
plt.plot(fit_hist.history['val_accuracy'])
plt.show()

my_sample = np.random.randint(10000)
plt.imshow(X_test[my_sample], cmap='gray')
print(labels[Y_test.iloc[my_sample]])

pred = model.predict(x_test[my_sample].reshape(-1, x_dim, y_dim, 1))

labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
print("pred: ", pred)  # 0~9까지 각 숫자일 확률 출력
print("argmax: ", labels[np.argmax(pred)])