import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import radon
from scipy import interpolate, stats
import sys
import warnings

warnings.filterwarnings("ignore")  # 경고문 출력 제거
np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
pd.set_option('display.max_columns', None)

df_clean = pd.read_pickle("./datasets/LSWMD_CleanData.pickle")
df_clean.info()
print(df_clean)

# failureType label 별 index 찾기
x = []
labels = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
for label in labels:
    idx = df_clean[df_clean['failureType'] == label].index
    x.append(idx[0])
print(x)  # x = [0, 43, 6492, 35, 97, 19, 541, 130, 814]


def cal_den(x):
    return 100 * (np.sum(x == 2) / np.size(x))


def find_regions(x):
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

    fea_reg_den = [cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4), cal_den(reg5), cal_den(reg6),
                   cal_den(reg7), cal_den(reg8), cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12),
                   cal_den(reg13)]
    return fea_reg_den


df_clean['fea_reg'] = df_clean.waferMap.apply(find_regions)
print(df_clean.head())

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
ax = ax.ravel(order='C')
for i in range(9):
    ax[i].bar(np.linspace(1, 13, 13), df_clean.fea_reg[x[i]])
    ax[i].set_title(df_clean.failureType[x[i]], fontsize=15)
    ax[i].set_xticklabels(labels)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
# plt.tight_layout()
plt.show()

# test


def change_val(img):
    img[img==1] =0
    return img


df_copy = df_clean.copy()
df_copy['new_waferMap'] =df_copy.waferMap.apply(change_val)

theta = np.linspace(0., 180., max(df_copy.waferMap[20256].shape), endpoint=False)
sinogram = radon(df_copy.waferMap[20256], theta=theta, preserve_range=True)
print('sinogram: ', sinogram)
xMean_Row = np.mean(sinogram, axis=1)
print('xMean_Row: ', xMean_Row)
x = np.linspace(1, xMean_Row.size, xMean_Row.size)
print('x: ', x)
y = xMean_Row
f = interpolate.interp1d(x, y, kind='cubic')
xnew = np.linspace(1, xMean_Row.size, 20)
print('xnew: ', xnew)
ynew = f(xnew) /100 # use interpolation function returned by `interp1d`
print('ynew: ', ynew)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

ax.bar(np.linspace(1, 20, 20), ynew)
ax.set_title(df_copy.failureType[20256], fontsize=10)
ax.set_xticks([])
ax.set_xlim([0, 21])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.show()

exit()


def cubic_inter_mean(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta) * 300
    xMean_Row = np.mean(sinogram, axis=1)
    # xMean_Row = np.mean(img, axis=1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    y = xMean_Row
    f = interpolate.interp1d(x, y, kind='cubic')
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew) / 100 # use interpolation function returned by `interp1d`
    return ynew


def cubic_inter_std(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta) * 1000
    xStd_Row = np.std(sinogram, axis=1)
    # xStd_Row = np.std(img, axis=1)
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    y = xStd_Row
    f = interpolate.interp1d(x, y, kind='cubic')
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew) / 100  # use interpolation function returned by `interp1d`
    return ynew


df_clean['fea_cub_mean'] = df_clean.waferMap.apply(cubic_inter_mean)
df_clean['fea_cub_std'] = df_clean.waferMap.apply(cubic_inter_std)

print(df_clean.fea_cub_mean.head())

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
ax = ax.ravel(order='C')
for i in range(9):
    ax[i].bar(np.linspace(1, 20, 20), df_clean.fea_cub_mean[x[i]])
    ax[i].set_title(df_clean.failureType[x[i]], fontsize=10)
    ax[i].set_xticks([])
    ax[i].set_xlim([0, 21])
    ax[i].set_ylim([0, 1])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
ax = ax.ravel(order='C')
for i in range(9):
    ax[i].bar(np.linspace(1, 20, 20), df_clean.fea_cub_std[x[i]])
    ax[i].set_title(df_clean.failureType[x[i]], fontsize=10)
    ax[i].set_xticks([])
    ax[i].set_xlim([0, 21])
    ax[i].set_ylim([0, 0.3])
plt.tight_layout()
plt.show()
