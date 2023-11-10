# https://www.kaggle.com/code/ashishpatel26/wm-811k-wafermap
import os
from os.path import join
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, Input, models
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
# loading libraries
import skimage
from skimage import measure
from skimage.transform import radon
from skimage.transform import probabilistic_hough_line
from skimage import measure
from scipy import interpolate
from scipy import stats

warnings.filterwarnings(action='ignore')
# datapath = join('data', 'wafer')

df=pd.read_pickle("./datasets/LSWMD.pkl")
# print(df.info())          # waferMap          웨이퍼 사진
                            # dieSize           다이싱 사이즈
                            # lotName           웨이퍼 1묶음의 번호
                            # waferIndex        웨이퍼 1묶음 내에서의 번호
                            # trianTestLabel    이게 왜 필요한지?
                            # failureType       웨이퍼 결함 타입

# uni_Index=np.unique(df.waferIndex, return_counts=True)
# plt.bar(uni_Index[0],uni_Index[1], color='gold', align='center', alpha=0.5)
# plt.title(" wafer Index distribution")
# plt.xlabel("index #")
# plt.ylabel("frequency")
# plt.xlim(0,26)
# plt.ylim(30000,34000)
# plt.show()

df = df.drop(['waferIndex'], axis = 1)      # 필요 없음.
def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1
df['waferMapDim']=df.waferMap.apply(find_dim)
# df.sample(5)
# print(max(df.waferMapDim), min(df.waferMapDim))       # ((300, 202), (6, 21))

uni_waferDim=np.unique(df.waferMapDim, return_counts=True)
uni_waferDim[0].shape[0]

df['failureNum']=df.failureType
df['trainTestNum']=df.trianTestLabel
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
mapping_traintest={'Training':0,'Test':1}
df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

# tol_wafers = df.shape[0]
# tol_wafers

df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
df_withlabel =df_withlabel.reset_index()
df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
df_withpattern = df_withpattern.reset_index()
df_nonpattern = df[(df['failureNum']==8)]
# df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]

# fig = plt.figure(figsize=(20, 4.5))
# gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5])
# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1])

# no_wafers=[tol_wafers-df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]]

# colors = ['silver', 'orange', 'gold']
# explode = (0.1, 0, 0)  # explode 1st slice
# labels = ['no-label','label&pattern','label&non-pattern']
# ax1.pie(no_wafers, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
#
# uni_pattern=np.unique(df_withpattern.failureNum, return_counts=True)
# labels2 = ['','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
# ax2.bar(uni_pattern[0],uni_pattern[1]/df_withpattern.shape[0], color='gold', align='center', alpha=0.9)
# ax2.set_title("failure type frequency")
# ax2.set_ylabel("% of pattern wafers")
# ax2.set_xticklabels(labels2)
# plt.show()

# fig, ax = plt.subplots(nrows = 10, ncols = 10, figsize=(20, 20))
# ax = ax.ravel(order='C')    # flatten() 함수랑 비슷함. 옵션 order=C,F,K
#                             # order='C': C와 같은 순서로 인덱싱하여 평평하게 배열 (디폴트)
#                             # order='F': Fortran과 같은 순서로 인덱싱하여 평평하게 배열
#                             # order='K': 메모리에서 발생하는 순서대로 인덱싱하여 평평하게 배열
# for i in range(100):
#     img = df_withpattern.waferMap[i]
#     ax[i].imshow(img)
#     ax[i].set_title(df_withpattern.failureType[i][0][0], fontsize=10)
#     ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
# plt.tight_layout()
# plt.show()

# x = [0,1,2,3,4,5,6,7]
# labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
#
# for k in x:
#     fig, ax = plt.subplots(nrows = 1, ncols = 10, figsize=(18, 12))
#     ax = ax.ravel(order='C')
#     for j in [k]:
#         img = df_withpattern.waferMap[df_withpattern.failureType==labels2[j]]
#         for i in range(10):
#             ax[i].imshow(img[img.index[i]])
#             ax[i].set_title(df_withpattern.failureType[img.index[i]][0][0], fontsize=10)
#             ax[i].set_xlabel(df_withpattern.index[img.index[i]], fontsize=10)
#             ax[i].set_xticks([])
#             ax[i].set_yticks([])
#     plt.tight_layout()
#     plt.show()

# x = [9,340, 3, 16, 0, 25, 84, 37]
# labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
#
# #ind_def = {'Center': 9, 'Donut': 340, 'Edge-Loc': 3, 'Edge-Ring': 16, 'Loc': 0, 'Random': 25,  'Scratch': 84, 'Near-full': 37}
# fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(20, 10))
# ax = ax.ravel(order='C')
# for i in range(8):
#     img = df_withpattern.waferMap[x[i]]
#     ax[i].imshow(img)
#     ax[i].set_title(df_withpattern.failureType[x[i]][0][0],fontsize=24)
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
# plt.tight_layout()
# plt.show()


# # illustration of 13 regions
# an = np.linspace(0, 2*np.pi, 100)
# plt.plot(2.5*np.cos(an), 2.5*np.sin(an))
# plt.axis('equal')
# plt.axis([-4, 4, -4, 4])        # x축의 제한을 -4에서 4로, y축의 제한을 -4에서 4로 설정
# plt.plot([-2.5, 2.5], [1.5, 1.5])
# plt.plot([-2.5, 2.5], [0.5, 0.5 ])
# plt.plot([-2.5, 2.5], [-0.5, -0.5 ])
# plt.plot([-2.5, 2.5], [-1.5,-1.5 ])
#
# plt.plot([0.5, 0.5], [-2.5, 2.5])
# plt.plot([1.5, 1.5], [-2.5, 2.5])
# plt.plot([-0.5, -0.5], [-2.5, 2.5])
# plt.plot([-1.5, -1.5], [-2.5, 2.5])
# plt.title(" Devide wafer map to 13 regions")
# plt.xticks([])
# plt.yticks([])
# plt.show()
########################## 이미지 보정 불량 반도체 이미지만 뽑아서 볼 수 있게 하는 과정 ##########################
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

    fea_reg_den = []
    fea_reg_den = [cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4), cal_den(reg5), cal_den(reg6),
                   cal_den(reg7), cal_den(reg8), cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12),
                   cal_den(reg13)]
    return fea_reg_den
df_withpattern['fea_reg'] = df_withpattern.waferMap.apply(find_regions)     # 1개의 웨이퍼의 위치별 불량 밀도

# x = [9, 340, 3, 16, 0, 25, 84, 37]      # 위에서 확인한 불량 종류 별 index
# labels2 = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
#
# fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))  # subplots: 여러 그래프 그려줌
# ax = ax.ravel(order='C')
# for i in range(8):
#     ax[i].bar(np.linspace(1, 13, 13), df_withpattern.fea_reg[x[i]])
#     ax[i].set_title(df_withpattern.failureType[x[i]][0][0], fontsize=15)
#     ax[i].set_xticks([])        # 눈금 없음
#     ax[i].set_yticks([])
#
# plt.tight_layout()
# plt.show()

## randon transform(라돈 변환,사이노그램(Sinogram)): 이걸 왜 하는건지?????
def change_val(img):            # 1이하의 값을 0으로 치환. 왜??????????
    img[img==1] =0
    return img
df_withpattern_copy = df_withpattern.copy()
df = df_withpattern_copy
df_withpattern_copy['new_waferMap'] = df_withpattern_copy.waferMap.apply(change_val) # apply: waferMap의 값을 ()으로 변경

# x = [9, 340, 3, 16, 0, 25, 84, 37]
# labels2 = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']
#
# fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))
# ax = ax.ravel(order='C')
# for i in range(8):
#     img = df_withpattern_copy.waferMap[x[i]]
#     theta = np.linspace(0., 180., max(img.shape), endpoint=False)   # 구간 시작점, 구간 끝점, 구간 내 숫자 개수
#     sinogram = radon(img, theta=theta)
#
#     ax[i].imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
#     ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0], fontsize=15)
#     ax[i].set_xticks([])
# plt.tight_layout()
# plt.show()
## cubic interpolation(삼차보간법): 각 차원에 대해 20으로 고정
def cubic_inter_mean(img):  # 삼차보간법 평균값
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis = 1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    y = xMean_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')  # 1차원 데이터 보간 함수
                                                    # x, y: 보간하려는 데이터 포인트의 x와 y 좌표
                                                    # kind: 보간 방법을 결정하는 인수("linear", "nearest", "zero", "slinear", "quadratic", "cubic")
                                                    # fill_value: 보간 함수가 정의되지 않은 지점에서 반환할 값
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew)/100
    return ynew

def cubic_inter_std(img): # 삼차보간법 표준 편차
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xStd_Row = np.std(sinogram, axis=1)     # 표준 편차 함수
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    y = xStd_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew)/100
    return ynew

df_withpattern_copy['fea_cub_mean'] =df_withpattern_copy.waferMap.apply(cubic_inter_mean)
df_withpattern_copy['fea_cub_std'] =df_withpattern_copy.waferMap.apply(cubic_inter_std)


x = [9,340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
# fea_cub_std
fig, ax = plt.subplots(nrows = 2, ncols = 4,figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    ax[i].bar(np.linspace(1,20,20),df_withpattern_copy.fea_cub_mean[x[i]])
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0],fontsize=10)
    ax[i].set_xticks([])
    ax[i].set_xlim([0,21])
    ax[i].set_ylim([0,1])
plt.tight_layout()
plt.show()
# fea_cub_std
fig, ax = plt.subplots(nrows = 2, ncols = 4,figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    ax[i].bar(np.linspace(1,20,20),df_withpattern_copy.fea_cub_std[x[i]])
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0],fontsize=10)
    ax[i].set_xticks([])
    ax[i].set_xlim([0,21])
    ax[i].set_ylim([0,0.3])
plt.tight_layout()
plt.show()



exit()      #################################################



# Geometry-based Features: 면적, 둘레, 장축 길이, 장축 길이, 장축 길이, 견고성 및 편심과 같은 기하학적 특징을 추출
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
ax = ax.ravel(order='C')
for i in range(8):
    img = df_withpattern_copy.waferMap[x[i]]
    zero_img = np.zeros(img.shape)      # np.zeros(shape, dtype, order): 0으로 가득 찬 array를 생성
    img_labels = measure.label(img, neighbors=4, connectivity=1, background=0)  # measure.label(label_image, background=None, return_num=False, connectivity=None)
    img_labels = img_labels - 1                                                 #
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


df_withpattern_copy.fea_geom[340]  # donut

df_all=df_withpattern_copy.copy()
a=[df_all.fea_reg[i] for i in range(df_all.shape[0])] #13
b=[df_all.fea_cub_mean[i] for i in range(df_all.shape[0])] #20
c=[df_all.fea_cub_std[i] for i in range(df_all.shape[0])] #20
d=[df_all.fea_geom[i] for i in range(df_all.shape[0])] #6
fea_all = np.concatenate((np.array(a),np.array(b),np.array(c),np.array(d)),axis=1) #59 in total

label=[df_all.failureNum[i] for i in range(df_all.shape[0])]
label=np.array(label)

import theano
from theano import tensor as T
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

X = fea_all
y = label

from collections import  Counter
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print('Training target statistics: {}'.format(Counter(y_train)))
print('Testing target statistics: {}'.format(Counter(y_test)))

RANDOM_STATE =42

# ---multicalss classification ---#
# One-Vs-One
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
clf2 = OneVsOneClassifier(LinearSVC(random_state = RANDOM_STATE)).fit(X_train, y_train)
y_train_pred = clf2.predict(X_train)
y_test_pred = clf2.predict(X_test)
train_acc2 = np.sum(y_train == y_train_pred, axis=0, dtype='float') / X_train.shape[0]
test_acc2 = np.sum(y_test == y_test_pred, axis=0, dtype='float') / X_test.shape[0]
print('One-Vs-One Training acc: {}'.format(train_acc2*100)) #One-Vs-One Training acc: 80.36
print('One-Vs-One Testing acc: {}'.format(test_acc2*100)) #One-Vs-One Testing acc: 79.04
print("y_train_pred[:100]: ", y_train_pred[:100])
print ("y_train[:100]: ", y_train[:100])

import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)

from matplotlib import gridspec
fig = plt.figure(figsize=(15, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

## Plot non-normalized confusion matrix
plt.subplot(gs[0])
plot_confusion_matrix(cnf_matrix, title='Confusion matrix')

# Plot normalized confusion matrix
plt.subplot(gs[1])
plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')

plt.show()