import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import skimage
from skimage import measure
from skimage.transform import radon
from skimage.transform import probabilistic_hough_line
from skimage import measure
from scipy import interpolate
from scipy import stats
import warnings
import sys
np.set_printoptions(threshold=sys.maxsize)

warnings.filterwarnings("ignore")

df = pd.read_pickle("./input/LSWMD.pkl")  # 피클에 있는 데이터 프레임 read
# df.info()

# waferMap          웨이퍼 사진 (0 : 공백 | 1 : 웨이퍼 | 2 : 결함)
# dieSize           다이싱 사이즈
# lotName           웨이퍼 1묶음의 번호
# waferIndex        웨이퍼 1묶음 내 번호
# trianTestLabel    훈련, 테스트 라벨
# failureType       웨이퍼 결함 타입

print(df.waferMap[3])
exit()

# 웨이퍼 수량 측정 그래프
# uni_Index = np.unique(df.waferIndex, return_counts=True)
# plt.bar(uni_Index[0], uni_Index[1], color='gold', align='center', alpha=0.5)
# plt.title(" wafer Index distribution")  # 그래프 제목
# plt.xlabel("index #")  # X 라벨 명
# plt.ylabel("frequency")  # Y 라벨 명
# plt.xlim(0, 26)  # X 축 범위
# plt.ylim(32000, 33000)  # Y 축 범위
# plt.show()

# 더미 데이터 제거
df = df.drop(['waferIndex'], axis=1)

# 웨이퍼 크기 계산
def find_dim(x):
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return dim0, dim1

df['waferMapDim'] = df.waferMap.apply(find_dim)
# print(df.sample(5))            # 데이터 5개만 무작위로 보여줌
# print(max(df.waferMapDim), min(df.waferMapDim))  # df.waferMapDim 최대 최소값 확인

uni_waferDim = np.unique(df.waferMapDim, return_counts=True)
# print(uni_waferDim[0].shape[0])

# str 데이터 -> float 데이터
df['failureNum'] = df.failureType
df['trainTestNum'] = df.trianTestLabel
mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6,
                'Near-full': 7, 'none': 8}
mapping_traintest = {'Training': 0, 'Test': 1}
df = df.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})

# 결함 분류가 된 데이터
df_withlabel = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 8)]
df_withlabel = df_withlabel.reset_index()
# 결함 분류가 안 된 데이터
df_withpattern = df[(df['failureNum'] >= 0) & (df['failureNum'] <= 7)]
df_withpattern = df_withpattern.reset_index()
# 결함이 불명확한 데이터
df_nonpattern = df[(df['failureNum'] == 8)]
print(df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0])
# 전체 데이터 수
tol_wafers = df.shape[0]

fig = plt.figure(figsize=(20, 4.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

no_wafers = [tol_wafers - df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]]

# colors = ['silver', 'orange', 'gold']
# explode = (0.1, 0, 0)  # explode 1st slice
# labels = ['no-label', 'label&pattern', 'label&non-pattern']
# ax1.pie(no_wafers, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

# uni_pattern = np.unique(df_withpattern.failureNum, return_counts=True)
# print(uni_pattern)
# labels2 = {'', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full'}
# ax2.bar(uni_pattern[0], uni_pattern[1] / df_withpattern.shape[0], color='gold', align='center', alpha=0.9)
# ax2.set_title("failure type frequency")
# ax2.set_ylabel("% of pattern wafers")
# ax2.set_xticklabels(labels2)
# plt.show()

fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))
# ravel() : 다차원 배열 -> 1차원 으로 변경 (flatten() 과 달리 원본 변형, 필요할 경우만 사본 생성)
# 'C' row 우선 변경 (Default 설정) / 'F' column 우선 변경
# 'K' 메모리에서 발생하는 순서 / 'A' 메모리에서 연속적인 포트란인 경우 포트란과 같은 인덱스 순서
ax = ax.ravel(order='C')

# wafer map image 100 개 출력
# for i in range(100):
#     img = df_withpattern.waferMap[i]
#     ax[i].imshow(img)
#     ax[i].set_title(df_withpattern.failureType[i][0][0], fontsize=10)
#     ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
# plt.tight_layout()
# plt.show()

x = [0, 1, 2, 3, 4, 5, 6, 7]
labels2 = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

# defect 당 10개씩 wafermap image 출력
# for k in x:
#     fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(18, 12))
#     ax = ax.ravel(order='C')
#     for j in [k]:
#         img = df_withpattern.waferMap[df_withpattern.failureType == labels2[j]]
#         for i in range(10):
#             ax[i].imshow(img[img.index[i]])
#             ax[i].set_title(df_withpattern.failureType[img.index[i]][0][0], fontsize=10)
#             ax[i].set_xlabel(df_withpattern.index[img.index[i]], fontsize=10)
#             ax[i].set_xticks([])
#             ax[i].set_yticks([])
#     plt.tight_layout()
#     plt.show()

# ind_def = {'Center': 9, 'Donut': 340, 'Edge-Loc': 3, 'Edge-Ring': 16, 'Loc': 0, 'Random': 25,  'Scratch': 84,
# 'Near-full': 37}
x = [9, 340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

# defect 별 wafer 수 막대그래프
# fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
# ax = ax.ravel(order='C')
# for i in range(8):
#     img = df_withpattern.waferMap[x[i]]
#     ax[i].imshow(img)
#     ax[i].set_title(df_withpattern.failureType[x[i]][0][0], fontsize=24)
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
# plt.tight_layout()
# plt.show()


# wafer 구역 나누기

# an = np.linspace(0, 2 * np.pi, 100)
# plt.plot(2.5 * np.cos(an), 2.5 * np.sin(an))
# plt.axis('equal')           # 축 간격 동일하게
# plt.axis([-4, 4, -4, 4])    # X, Y 축 범위 -4, 4 로 제한
# plt.plot([-2.5, 2.5], [1.5, 1.5])
# plt.plot([-2.5, 2.5], [0.5, 0.5])
# plt.plot([-2.5, 2.5], [-0.5, -0.5])
# plt.plot([-2.5, 2.5], [-1.5, -1.5])
# plt.plot([0.5, 0.5], [-2.5, 2.5])
# plt.plot([1.5, 1.5], [-2.5, 2.5])
# plt.plot([-0.5, -0.5], [-2.5, 2.5])
# plt.plot([-1.5, -1.5], [-2.5, 2.5])
# plt.title(" Devide wafer map to 13 regions")
# plt.xticks([])            # X, Y 축 간격 숨기기
# plt.yticks([])
# plt.show()


def cal_den(x):
    return 100 * (np.sum(x == 2) / np.size(x))  # 밀도 구하기


def find_regions(x):    # 밀도 구할 범위 지정
    rows = np.size(x, axis=0)
    cols = np.size(x, axis=1)
    ind1 = np.arange(0, rows, rows // 5)
    ind2 = np.arange(0, cols, cols // 5)

    reg1 = x[ind1[0]:ind1[1], :]
    reg3 = x[ind1[4]:, :]
    reg4 = x[:, ind2[0]:ind2[1]]
    reg2 = x[:, ind2[4]:]

    reg5 = x[ind1[1]:ind1[2], ind2[1]:ind2[2]]
    reg6 = x[ind1[1]:ind1[2], ind2[2]:ind2[3]]
    reg7 = x[ind1[1]:ind1[2], ind2[3]:ind2[4]]
    reg8 = x[ind1[2]:ind1[3], ind2[1]:ind2[2]]
    reg9 = x[ind1[2]:ind1[3], ind2[2]:ind2[3]]
    reg10 = x[ind1[2]:ind1[3], ind2[3]:ind2[4]]
    reg11 = x[ind1[3]:ind1[4], ind2[1]:ind2[2]]
    reg12 = x[ind1[3]:ind1[4], ind2[2]:ind2[3]]
    reg13 = x[ind1[3]:ind1[4], ind2[3]:ind2[4]]

    fea_reg_den = []
    fea_reg_den = [cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4), cal_den(reg5), cal_den(reg6),
                   cal_den(reg7), cal_den(reg8), cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12),
                   cal_den(reg13)]
    return fea_reg_den


df_withpattern['fea_reg'] = df_withpattern.waferMap.apply(find_regions)

# defect 구역 별 밀도 그래프로 표시
# fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
# ax = ax.ravel(order='C')
# for i in range(8):
#     ax[i].bar(np.linspace(1, 13, 13), df_withpattern.fea_reg[x[i]])
#     ax[i].set_title(df_withpattern.failureType[x[i]][0][0], fontsize=15)
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
#
# plt.tight_layout()
# plt.show()


def change_val(img):
    img[img == 1] = 0
    return img

df_withpattern_copy = df_withpattern.copy()
df_withpattern_copy['new_waferMap'] = df_withpattern_copy.waferMap.apply(change_val)

x = [9, 340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    img = df_withpattern_copy.waferMap[x[i]]
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)

    ax[i].imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0], fontsize=15)
    ax[i].set_xticks([])
plt.tight_layout()

plt.show()


def cubic_inter_mean(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis=1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    y = xMean_Row
    f = interpolate.interp1d(x, y, kind='cubic')
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew) / 100  # use interpolation function returned by `interp1d`
    return ynew


def cubic_inter_std(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xStd_Row = np.std(sinogram, axis=1)
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    y = xStd_Row
    f = interpolate.interp1d(x, y, kind='cubic')
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew) / 100  # use interpolation function returned by `interp1d`
    return ynew


df_withpattern_copy['fea_cub_mean'] = df_withpattern_copy.waferMap.apply(cubic_inter_mean)
df_withpattern_copy['fea_cub_std'] = df_withpattern_copy.waferMap.apply(cubic_inter_std)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    ax[i].bar(np.linspace(1, 20, 20), df_withpattern_copy.fea_cub_mean[x[i]])
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0], fontsize=10)
    ax[i].set_xticks([])
    ax[i].set_xlim([0, 21])
    ax[i].set_ylim([0, 1])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    ax[i].bar(np.linspace(1, 20, 20), df_withpattern_copy.fea_cub_std[x[i]])
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0], fontsize=10)
    ax[i].set_xticks([])
    ax[i].set_xlim([0, 21])
    ax[i].set_ylim([0, 0.3])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    img = df_withpattern_copy.waferMap[x[i]]
    zero_img = np.zeros(img.shape)
    img_labels = measure.label(img, neighbors=4, connectivity=1, background=0)
    img_labels = img_labels - 1
    if img_labels.max() == 0:
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels > -1], axis=None)
        no_region = info_region[0]

    zero_img[np.where(img_labels == no_region)] = 2
    ax[i].imshow(zero_img)
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0], fontsize=10)
    ax[i].set_xticks([])
plt.tight_layout()
plt.show()


def cal_dist(img, x, y):
    dim0 = np.size(img, axis=0)
    dim1 = np.size(img, axis=1)
    dist = np.sqrt((x - dim0 / 2) ** 2 + (y - dim1 / 2) ** 2)
    return dist


def fea_geom(img):
    norm_area = img.shape[0] * img.shape[1]
    norm_perimeter = np.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2)

    img_labels = measure.label(img, neighbors=4, connectivity=1, background=0)

    if img_labels.max() == 0:
        img_labels[img_labels == 0] = 1
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels > 0], axis=None)
        no_region = info_region[0][0] - 1

    prop = measure.regionprops(img_labels)
    prop_area = prop[no_region].area / norm_area
    prop_perimeter = prop[no_region].perimeter / norm_perimeter

    prop_cent = prop[no_region].local_centroid
    prop_cent = cal_dist(img, prop_cent[0], prop_cent[1])

    prop_majaxis = prop[no_region].major_axis_length / norm_perimeter
    prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter
    prop_ecc = prop[no_region].eccentricity
    prop_solidity = prop[no_region].solidity

    return prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity


df_withpattern_copy['fea_geom'] = df_withpattern_copy.waferMap.apply(fea_geom)

print(df_withpattern_copy.fea_geom[340])  # donut

df_all = df_withpattern_copy.copy()
a = [df_all.fea_reg[i] for i in range(df_all.shape[0])]  # 13
b = [df_all.fea_cub_mean[i] for i in range(df_all.shape[0])]  # 20
c = [df_all.fea_cub_std[i] for i in range(df_all.shape[0])]  # 20
d = [df_all.fea_geom[i] for i in range(df_all.shape[0])]  # 6
fea_all = np.concatenate((np.array(a), np.array(b), np.array(c), np.array(d)), axis=1)  # 59 in total

label = [df_all.failureNum[i] for i in range(df_all.shape[0])]
label = np.array(label)
