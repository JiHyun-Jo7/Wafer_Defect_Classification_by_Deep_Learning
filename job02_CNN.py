import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import sys
import warnings

warnings.filterwarnings("ignore")  # 경고문 출력 제거
# np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
pd.set_option('display.max_columns', None)

failureType_Label = ['Normal', 'Edge-Ring', 'Edge-Loc', 'Center', 'Loc', 'Scratch', 'Random', 'Donut', 'Near-full']

df_train, df_test = np.load('./datasets/train_test_0.949.pkl', allow_pickle=True)
df_train.reset_index(inplace=True)
df_test.reset_index(inplace=True)
# print(df_train)
# print(df_test)
df_train.info()
# df_test.info()

# exit()
# x_train, x_test
X_train = df_train.fea_cub_mean     ## 수정
X_test = df_test.fea_cub_mean       ## 수정
Y_train = df_train.failureNum
Y_test = df_test.failureNum

list_train = []
for i in X_train:
    # a = list(i)
    x = list(i.reshape(-1, 1))
    list_train.append(x)
x_train = np.array(list_train)

list_test = []
for i in X_test:
    # a = list(i)
    x = list(i.reshape(-1, 1))
    list_test.append(x)
x_test = np.array(list_test)

print(X_train, type(X_train))

# y_train, y_test
# one-hot 인코딩 & softmax
# 레이블을 원-핫 인코딩
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)
print(x_test.shape, x_train.shape)
print(y_test.shape, y_train.shape)

input_shape = x_train[0].shape
print(input_shape)
# exit()
model = Sequential()
model.add(Conv1D(32,input_shape=input_shape, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(1, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(9, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(x_train, y_train, batch_size=4096, epochs=90, validation_data=(x_test, y_test))

model.save('./models/Wafer_fea_geom_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))

plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()

## 나머지 입력값 쓸모없다고 판단.

