import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import random, pickle, sys, warnings

warnings.filterwarnings("ignore")  # 경고문 출력 제거
np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
pd.set_option('display.max_columns', None)

df = pd.read_pickle("./datasets/LSWMD_Copy.pickle")
# df = pd.read_pickle("./datasets/LSWMD_Rotation.pickle")
# df = pd.read_pickle("./datasets/LSWMD_Smote.pickle")
df.reset_index(drop=True, inplace=True)
df.info()
print(df[['failureType', 'failureNum']])
print(df.failureType.value_counts())

# 'failureNum'을 'failureType'으로 매핑
label_mapping = dict(zip(df['failureNum'], df['failureType']))

# 각 행의 x 좌표와 y 좌표를 추출하여 새로운 열을 생성
df['x'] = df['waferMapDim'].apply(lambda x: x[0])
df['y'] = df['waferMapDim'].apply(lambda x: x[1])

# 각 차원에 대한 중앙값 계산
median_x = df['x'].median()
median_y = df['y'].median()

# 이미지를 정사각형으로 만들기
if median_x > median_y:
    target_size = (median_x, median_x)
else: target_size = (median_y, median_y)
print("Median :", target_size)

df.drop(['x', 'y'], axis=1, inplace=True)
df.info()


# 이미지 크기를 통일시키는 함수
def resize_wafer_map(wafer_map, target_size, resample_method=Image.BILINEAR):
    try:
        # Numpy 배열을 이미지로 변환
        image_array = np.array(wafer_map)
        image = Image.fromarray(image_array.astype('uint8'))  # Numpy 배열을 이미지로 변환

        # 이미지 크기 변환
        resized_image = image.resize((int(target_size[1]), int(target_size[0])), resample=resample_method)

        return np.array(resized_image)
    except Exception as e:
        print(f"Error in resizing: {e}")
        return None


# 'resized_waferMap' 열에 변환한 데이터 추가
df['resized_waferMap'] = df['waferMap'].apply(lambda x: resize_wafer_map(x, target_size))
print('finish resizing image')


resize_sample = np.random.randint(len(df.waferMap))
compare_img = [df.waferMap[resize_sample], df.resized_waferMap[resize_sample]]
title = ['Original WaferMap\nIdx: [{}]', 'Resized WaferMap\nIdx: [{}]']
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
for i in range(2):
    img = compare_img[i]
    ax[i].imshow(img, cmap='gray')
    ax[i].set_title(title[i].format(resize_sample), fontsize=15)
    ax[i].set_xlabel(df.failureType[resize_sample], fontsize=12)
plt.show()

# 훈련 및 테스트 세트로 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(df['resized_waferMap'], df['failureNum'], test_size=0.2, random_state=42)

X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())


# 훈련 데이터 및 레이블 확인
print("\n훈련 데이터 형태:")
print(X_train.shape, Y_train.shape)

# 테스트 데이터 및 레이블 확인
print("\n테스트 데이터 형태:")
print(X_test.shape, Y_test.shape)

# 레이블 원-핫 인코딩
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

my_sample = np.random.randint(len(X_train))

# 레이블 확인
original_label = np.argmax(y_train[my_sample])
print("\n원래 레이블:", original_label)

# 매핑된 레이블 확인
mapped_label = label_mapping[original_label]
print("\n매핑된 레이블:", mapped_label)

# 원-핫 인코딩된 레이블 확인
print("\n원-핫 인코딩된 레이블:")
print(y_train[my_sample])

# 이미지 플로팅 및 레이블 확인
plt.imshow(X_train[my_sample], cmap='gray')  # cmap = 'gray': 흑백 처리
plt.title(mapped_label)
plt.xlabel("label: [{}]".format(original_label))
plt.show()

# 'labels' 리스트를 사용하여 레이블 출력
labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-Full']
print("\ntag:", labels[original_label])

# print(X_train.iloc[my_sample])
print('type: ', type(X_train[my_sample]))

# X_train = X_train / 2  # max(X_train) = 2
# X_test = X_test / 2  # max(X_test) = 2

x_dim, y_dim = target_size
x_dim, y_dim = int(x_dim), int(y_dim)
x_train = X_train.reshape(len(X_train), x_dim, y_dim, 1)
x_test = X_test.reshape(-1, x_dim, y_dim, 1)

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
                     epochs=70, validation_split=0.2, verbose=1)

val_acc = round(fit_hist.history['val_accuracy'][-1], 3)


train_test = X_train, X_test, x_train, x_test, Y_train, Y_test, y_train, y_test
with open('./datasets/train_test_Copy_{}.pkl'.format(val_acc), 'wb') as file:
    pickle.dump(train_test, file)
# with open('./datasets/train_test_Rotation_{}.pkl'.format(val_acc), 'wb') as file:
#     pickle.dump(train_test, file)
with open('./datasets/train_test_Smote_{}.pkl'.format(val_acc), 'wb') as file:
    pickle.dump(train_test, file)

# 모델 피클로 저장
model.save('./models/CNN_Copy_{}.h5'.format(val_acc))
# model.save('./models/CNN_Rotation_{}.h5'.format(val_acc))
# model.save('./models/CNN_Smote_{}.h5'.format(val_acc))

score = model.evaluate(x_test, y_test, verbose=0)
print('Final test set accuracy', score[1])

plt.plot(fit_hist.history['accuracy'])
plt.plot(fit_hist.history['val_accuracy'])
plt.xticks(range(0, len(fit_hist.history['accuracy']) + 1, 5))
plt.show()

# 각 카테고리별로 랜덤한 이미지 하나씩 선택
selected_images = []

for i in range(len(labels)):
    category_indices = np.where(Y_test == i)[0]
    selected_index = random.choice(category_indices)
    selected_images.append(selected_index)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, index in enumerate(selected_images):
    true_label = labels[Y_test.iloc[index]]

    # 모델 예측
    pred = model.predict(X_test[index].reshape(1, x_dim, y_dim, 1))
    predicted_label = labels[np.argmax(pred)]

    # 이미지와 예측 결과 시각화

    ax = axes[i // 3, i % 3]
    ax.imshow(X_test[index], cmap='gray', vmin=0, vmax=2)
    ax.set_title(f'True Label: {true_label}')
    text_color = 'red' if true_label != predicted_label else 'black'
    ax.set_xlabel(f'argmax: {predicted_label}', color=text_color)
    ax.set_xticks([])
    ax.set_yticks([])
    print("Index: [{}]".format(index))
    print(pred)
    print("argmax: ", predicted_label)

plt.tight_layout()
plt.show()

# 모델 예측
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# 실제 레이블 정수화
true_labels = np.argmax(y_test, axis=1)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# 정확도 계산
accuracy = accuracy_score(true_labels, predicted_labels)

# 각 클래스에 대한 정확도 계산
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# 히트맵 시각화
plt.figure(figsize=(16, 6))

# 혼동 행렬 (Without Normalization)
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# 정규화된 혼동 행렬
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')

plt.suptitle('Model Evaluation')
plt.show()

# 분류 보고서 출력
class_report = classification_report(true_labels, predicted_labels)
print("Classification Report:")
print(class_report)