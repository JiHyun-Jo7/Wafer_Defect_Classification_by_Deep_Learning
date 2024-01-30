import numpy as np
import pandas as pd
import smote
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
from PIL import Image

df = pd.read_pickle("./datasets/LSWMD_CleanData.pickle")
df.info()
print(df['failureType'].value_counts())

# 데이터프레임의 인덱스 재설정
df.reset_index(drop=True, inplace=True)

# 'Normal' 클래스의 샘플 수를 선택한 클래스에 맞춰 언더샘플링
X = df.drop(columns=['failureType'])
Y = df['failureType']
X_under, Y_under = RandomUnderSampler(sampling_strategy={'Normal': 10000}).fit_resample(X, Y)

# 언더샘플링된 데이터프레임 생성
df = pd.concat([X_under, pd.Series(Y_under, name='failureType')], axis=1)

# 최종 결과 확인
print(df['failureType'].value_counts())


# 데이터 복제 증강
df_copy = df.copy()
selected_rows = df_copy[df_copy['failureType'].isin(['Edge-Loc', 'Center'])]
df_copy = pd.concat([df_copy, selected_rows], ignore_index=True)

def replicate_rows(df, failure_type, replication_factor):
    selected_rows = df[df['failureType'] == failure_type]
    df = pd.concat([df] + [selected_rows] * replication_factor, ignore_index=True)
    return df


# 각 'failureType'에 따라 복제 횟수 설정
replication_factors = {'Loc': 2, 'Scratch': 7, 'Random': 11, 'Donut': 17, 'Near-full': 66}

# 함수를 사용하여 각 'failureType'에 대해 복제 수행
for failure_type, replication_factor in replication_factors.items():
    df_copy = replicate_rows(df_copy, failure_type, replication_factor)

print('df_copy')
# print(df_copy['failureType'].value_counts())
with open('./datasets/LSWMD_Copy.pickle', 'wb') as f:
    pickle.dump(df, f)


# 데이터 회전, 반전 증강
df_rotation = df.copy()
replication_factors = {'Donut': 1, 'Near-full': 4}
for failure_type, replication_factor in replication_factors.items():
    df_rotation = replicate_rows(df_rotation, failure_type, replication_factor)

# WaferMap을 회전하는 함수 (90, 180, 270도 회전)
def rotate_wafermap(wafermap, degrees):
    if degrees == 90:
        return np.rot90(wafermap, k=1)  # 시계방향으로 90도 회전
    elif degrees == 180:
        return np.rot90(wafermap, k=2)  # 시계방향으로 180도 회전
    elif degrees == 270:
        return np.rot90(wafermap, k=3)  # 시계방향으로 270도 회전
    else:
        raise ValueError("Unsupported rotation degrees")


# Edge-Loc 및 Center 상하 반전
selected_rows = df_rotation[df_rotation['failureType'].isin(['Edge-Loc', 'Center'])].copy()
selected_rows['waferMap'] = selected_rows['waferMap'].apply(np.flipud)
df_rotation = pd.concat([df_rotation, selected_rows], ignore_index=True)

# Loc 상하, 좌우 반전
selected_Loc = df_rotation[df_rotation['failureType'] == 'Loc'].copy()
selected_Loc['waferMap'] = selected_Loc['waferMap'].apply(np.flipud)
df_rotation = pd.concat([df_rotation, selected_Loc], ignore_index=True)
selected_Loc['waferMap'] = selected_Loc['waferMap'].apply(np.fliplr)
df_rotation = pd.concat([df_rotation, selected_Loc], ignore_index=True)

# Scratch 상하, 좌우 반전
selected_Scratch = df_rotation[df_rotation['failureType'] == 'Scratch'].copy()
selected_Scratch['waferMap'] = selected_Scratch['waferMap'].apply(np.flipud)
df_rotation = pd.concat([df_rotation, selected_Scratch], ignore_index=True)
selected_Scratch['waferMap'] = selected_Scratch['waferMap'].apply(np.fliplr)
df_rotation = pd.concat([df_rotation, selected_Scratch], ignore_index=True)

# Scratch 90, 180, 270 회전 데이터
degrees_to_rotate = [90, 180, 270]
for degrees in degrees_to_rotate:
    selected_Scratch['waferMap'] = selected_Scratch['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Scratch], ignore_index=True)

# Scratch 상하 반전 후 90, 180 회전 데이터
degrees_to_rotate = [90, 180]
for degrees in degrees_to_rotate:
    selected_Scratch['waferMap'] = selected_Scratch['waferMap'].apply(np.flipud)
    selected_Scratch['waferMap'] = selected_Scratch['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Scratch], ignore_index=True)


# Random 상하, 좌우, 상하좌우 반전
selected_Random = df_rotation[df_rotation['failureType'] == 'Random'].copy()
selected_Random['waferMap'] = selected_Random['waferMap'].apply(np.flipud)
df_rotation = pd.concat([df_rotation, selected_Random], ignore_index=True)
selected_Random['waferMap'] = selected_Random['waferMap'].apply(np.fliplr)
df_rotation = pd.concat([df_rotation, selected_Random], ignore_index=True)
selected_Random['waferMap'] = selected_Random['waferMap'].apply(lambda x: np.flipud(np.fliplr(x)))
df_rotation = pd.concat([df_rotation, selected_Random], ignore_index=True)

# Random 90, 180, 270 회전 데이터
degrees_to_rotate = [90, 180, 270]
for degrees in degrees_to_rotate:
    selected_Random['waferMap'] = selected_Random['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Random], ignore_index=True)

# Random 상하 반전 후 90, 180 회전 데이터
degrees_to_rotate = [90, 180]
for degrees in degrees_to_rotate:
    selected_Random['waferMap'] = selected_Random['waferMap'].apply(np.flipud)
    selected_Random['waferMap'] = selected_Random['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Random], ignore_index=True)

# Random 좌우 반전 후 90, 180 회전 데이터
degrees_to_rotate = [90, 180]
for degrees in degrees_to_rotate:
    selected_Random['waferMap'] = selected_Random['waferMap'].apply(np.fliplr)
    selected_Random['waferMap'] = selected_Random['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Random], ignore_index=True)


# Donut 상하, 좌우 반전
selected_Donut = df_rotation[df_rotation['failureType'] == 'Donut'].copy()
selected_Donut['waferMap'] = selected_Donut['waferMap'].apply(np.flipud)
df_rotation = pd.concat([df_rotation, selected_Donut], ignore_index=True)
selected_Donut['waferMap'] = selected_Donut['waferMap'].apply(np.fliplr)
df_rotation = pd.concat([df_rotation, selected_Donut], ignore_index=True)

# Donut 90, 180 회전 데이터
degrees_to_rotate = [90, 180]
for degrees in degrees_to_rotate:
    selected_Donut['waferMap'] = selected_Donut['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Donut], ignore_index=True)

# Donut 상하 반전 후 90, 180 회전 데이터
degrees_to_rotate = [90, 180]
for degrees in degrees_to_rotate:
    selected_Donut['waferMap'] = selected_Donut['waferMap'].apply(np.flipud)
    selected_Donut['waferMap'] = selected_Donut['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Donut], ignore_index=True)

# Donut 좌우 반전 후 90, 180 회전 데이터
degrees_to_rotate = [90, 180]
for degrees in degrees_to_rotate:
    selected_Donut['waferMap'] = selected_Donut['waferMap'].apply(np.fliplr)
    selected_Donut['waferMap'] = selected_Donut['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Donut], ignore_index=True)


# Near-full 상하, 좌우, 상하좌우 반전
selected_Near_full = df_rotation[df_rotation['failureType'] == 'Near-full'].copy()
selected_Near_full['waferMap'] = selected_Near_full['waferMap'].apply(np.flipud)
df_rotation = pd.concat([df_rotation, selected_Near_full], ignore_index=True)
selected_Near_full['waferMap'] = selected_Near_full['waferMap'].apply(np.fliplr)
df_rotation = pd.concat([df_rotation, selected_Near_full], ignore_index=True)
selected_Near_full['waferMap'] = selected_Near_full['waferMap'].apply(lambda x: np.flipud(np.fliplr(x)))
df_rotation = pd.concat([df_rotation, selected_Near_full], ignore_index=True)

# Near-full 90, 180, 270 회전 데이터
degrees_to_rotate = [90, 180, 270]
for degrees in degrees_to_rotate:
    selected_Near_full['waferMap'] = selected_Near_full['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Near_full], ignore_index=True)

# Near-full 상하 반전 후 90, 180 회전 데이터
degrees_to_rotate = [90, 180]
for degrees in degrees_to_rotate:
    selected_Near_full['waferMap'] = selected_Near_full['waferMap'].apply(np.flipud)
    selected_Near_full['waferMap'] = selected_Near_full['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Near_full], ignore_index=True)

# Near-full 좌우 반전 후 90, 180 회전 데이터
degrees_to_rotate = [90, 180]
for degrees in degrees_to_rotate:
    selected_Near_full['waferMap'] = selected_Near_full['waferMap'].apply(np.fliplr)
    selected_Near_full['waferMap'] = selected_Near_full['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Near_full], ignore_index=True)

# Near-full 좌우 반전 후 90, 180 회전 데이터
degrees_to_rotate = [90, 180]
for degrees in degrees_to_rotate:
    selected_Near_full['waferMap'] = selected_Near_full['waferMap'].apply(lambda x: np.flipud(np.fliplr(x)))
    selected_Near_full['waferMap'] = selected_Near_full['waferMap'].apply(lambda x: rotate_wafermap(x, degrees))
    df_rotation = pd.concat([df_rotation, selected_Near_full], ignore_index=True)

print('df_rotation')
print(df_rotation['failureType'].value_counts())

with open('./datasets/LSWMD_Rotation.pickle', 'wb') as f:
    pickle.dump(df, f)


# SMOTE 데이터 생성
df_smote = df.copy()

# 각 행의 x 좌표와 y 좌표를 추출하여 새로운 열을 생성
df_smote['x'] = df_smote['waferMapDim'].apply(lambda x: x[0])
df_smote['y'] = df_smote['waferMapDim'].apply(lambda x: x[1])

# 각 차원에 대한 중앙값 계산
median_x = df_smote['x'].median()
median_y = df_smote['y'].median()

# 이미지를 정사각형으로 만들기
if median_x > median_y:
    target_size = (median_x, median_x)
else: target_size = (median_y, median_y)
print("Median :", target_size)

df_smote.drop(['x', 'y'], axis=1, inplace=True)
df_smote.info()


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
df_smote['resized_waferMap'] = df_smote['waferMap'].apply(lambda x: resize_wafer_map(x, target_size))
print('finish resizing image')

# 이미지 데이터와 클래스 정보 추출
X = np.array(df_smote['resized_waferMap'].tolist())  # 2D 넘파이 배열로 변환
y = df_smote['failureType']

# SMOTE 적용 전 클래스 분포 확인
print("원본 클래스 분포:\n", y.value_counts())

# 이미지 데이터를 1D로 펼친 후 SMOTE 적용
X_flat = X.reshape(X.shape[0], -1)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_flat, y)

# SMOTE로 증강된 데이터를 새로운 데이터프레임에 저장
df_smote = pd.DataFrame({
    'waferMap': [image.reshape(X.shape[1:]) for image in X_resampled],
    'failureType': y_resampled})

# SMOTE 적용 후 클래스 분포 확인
print("SMOTE 적용 후 클래스 분포:\n", df_smote[피클을 완벽하게 만들다'failureType'].value_counts())

with open('./datasets/LSWMD_Smote.pickle', 'wb') as f:
    pickle.dump(df, f)