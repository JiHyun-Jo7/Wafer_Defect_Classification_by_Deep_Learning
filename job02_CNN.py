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

labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

df_training=pd.read_pickle('./datasets/LSWMD_Train.pickle')
# df_training.info()
df_test=pd.read_pickle('./datasets/LSWMD_Test.pickle')
# df_test.info()

# print(type(df_training['fea_cub_mean'][0]), type(df_training['failureNum'][0]))
X_train, Y_train = df_training['fea_cub_mean'], df_training['failureNum']
# print(X_train.shape, Y_train.shape)
X_test, Y_test = df_test['fea_cub_mean'], df_test['failureNum']
# print(X_test.shape, Y_test.shape)

# one-hot 인코딩 & softmax
# 레이블을 원-핫 인코딩
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)
# print(y_train, Y_test)

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




