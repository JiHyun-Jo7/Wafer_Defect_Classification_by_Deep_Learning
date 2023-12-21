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
# np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
pd.set_option('display.max_columns', None)

df_training=pd.read_pickle('./datasets/LSWMD_Train.pickle')
df_training.drop(columns=['index'], inplace=True)
# df_training.info()
df_test=pd.read_pickle('./datasets/LSWMD_Test.pickle')
df_test.drop(columns=['index'], inplace=True)
# df_test.info()

# X_train, numpy.array(numpy.array)형태 -> numpy.array(list)형태로 변환(안 해주면 tensor에서 못 받음)
X_train = []
for i in df_training['fea_cub_std']:
    X_train.append(list(i))
x_train = np.array(X_train)

X_test = []
for i in df_test['fea_cub_std']:
    X_test.append(list(i))
x_test = np.array(X_test)

# Y_train, one-hot 인코딩 & softmax
# 레이블을 원-핫 인코딩
Y_train = np.array(df_training.failureNum.values.reshape(-1,1))
Y_test = np.array(df_test.failureNum.values.reshape(-1,1))
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

# print(y_train, y_test)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# modeling
model = Sequential()
model.add(Dense(256, input_dim=20, activation='relu'))  # len(X_train[0]) = 20
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(9, activation='softmax'))               # len(y_train) = 9
model.summary()

opt = Adam(lr=0.001)        # Adam객체 생성. Adam: 경사하강알고리즘 종류 중 하나. 최종본의 느낌. lr=learning_rate. 어차피 알아서 조절한다. 크게 신경 쓸 필요 없다.
model.compile(opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

fit_hist = model.fit(x_train, y_train,batch_size=5,epochs=50,verbose=1)
score = model.evaluate(x_test, y_test,verbose=0)
print('정확도:',score[1])

plt.plot(fit_hist.history['accuracy'])
plt.show()

# val_acc = round(fit_hist.history['val_accuracy'][-1], 3)
# model.save('./models/CNN_{}.h5'.format(val_acc))
#
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Final test set accuracy', score[1])
#
# plt.plot(fit_hist.history['accuracy'])
# plt.plot(fit_hist.history['val_accuracy'])
# plt.show()
#
# my_sample = np.random.randint(10000)
# plt.imshow(X_test[my_sample], cmap='gray')
# print(labels[Y_test.iloc[my_sample]])
#
# pred = model.predict(x_test[my_sample].reshape(-1, x_dim, y_dim, 1))
#
# labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
# print("pred: ", pred)  # 0~9까지 각 숫자일 확률 출력
# print("argmax: ", labels[np.argmax(pred)])




