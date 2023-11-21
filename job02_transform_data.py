import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import skimage
from skimage import measure
from skimage.transform import radon, probabilistic_hough_line
from scipy.io import mmwrite, mmread
from scipy import interpolate, stats
import sys
import warnings

warnings.filterwarnings("ignore")  # 경고문 출력 제거
np.set_printoptions(threshold=sys.maxsize)  # 배열 전체 출력
pd.set_option('display.max_columns', None)

# loading libraries
# 0. 'Normal': 147431, 1. Center': 4294, 2. 'Donut': 555, 3. 'Edge-Loc': 5189,
# 4. 'Edge-Ring': 9680, 5. 'Loc': 3593, 6. 'Random': 866, 7. 'Scratch': 1193, 8. 'Near-full': 149

# 카테고리별 데이터양 맞춰주기
df_clean = pd.read_pickle("./datasets/LSWMD_clean_data.pickle")
pd.set_option('display.max_columns', None)
df_clean.info()
# df_clean.sort_values('failureNum', inplace=True)
df_clean.set_index('waferMap', inplace=True)
df_clean.reset_index(inplace=True)
print('df_clean')
# print(df_clean.head())
test = df_clean.iloc[34839]
print(test)

exit()
# index_Num = df_clean.index[df_clean['failureType'] == 'Near-full']
# print(index_Num)

x = [0, 43, 6492, 35, 97, 19, 541, 130, 814]
labels2 = ['Normal', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

# 구역별 밀도 설정


def cal_den(x):
    return 100 * (np.sum(x == 2) / np.size(x))


def find_regions(x):
    rows = np.size(x, axis=0)
    cols = np.size(x, axis=1)

    ind1 = np.arange(0, rows, rows//5)
    ind2 = np.arange(0, cols, cols//5)

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


df_clean['fea_reg'] = df_clean.waferMap.apply(find_regions)


# 보간법으로 waferMap 사이즈 맞추기
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


df_clean['fea_cub_mean'] = df_clean.waferMap.apply(cubic_inter_mean)
df_clean['fea_cub_std'] = df_clean.waferMap.apply(cubic_inter_std)



