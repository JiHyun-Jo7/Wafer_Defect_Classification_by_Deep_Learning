import numpy as np
import pandas as pd
from scipy.io import mmwrite, mmread
import pickle
import matplotlib.pyplot as plt

# df=pd.read_pickle("./datasets/LSWMD.pkl")
# pd.set_option('display.max_columns', None)
# df.info()
# # print(df.head(10))
#
# # 필요 없는 컬럼 삭제
# columns_to_drop = ['dieSize', 'lotName', 'waferIndex', 'trianTestLabel']
# df = df.drop(columns=columns_to_drop, axis=1)
# df.info()
# # print(df.head(10))
# # with open('./datasets/LSWMD_drop_data.pickle', 'wb') as f:
# #     pickle.dump(df, f)
#
# #  trianTestLabel&failureType 데이터 타입 변경
# df=pd.read_pickle('./datasets/LSWMD_drop_data.pickle')
# pd.set_option('display.max_columns', None)
# print(df.info())
# print(df.head(40))
# ## trianTestLabel
# for i in range(len(df['trianTestLabel'])):
#     try:
#         df['trianTestLabel'][i] = df['trianTestLabel'][i][0][0]
#         if i % 100 == 0: print(i)
#     except:
#         df['trianTestLabel'][i] = None
# print(df['trianTestLabel'].head(10))
# ## failureType
# for i in range(len(df['failureType'])):
#     try:
#         df['failureType'][i] = df['failure Type'][i][0][0]
#         if i % 100 == 0: print(i)
#     except:
#         df['failureType'][i] = None
# print(df['failureType'].head(10))

# # with open('./datasets/LSWMD_drop_None_data.pickle', 'wb') as f:
# #     pickle.dump(df, f)

# df=pd.read_pickle('./datasets/LSWMD_dropna_replace_Normal.pickle')
# pd.set_option('display.max_columns', None)
# df.info()
# print(df.head())
# ALL_value = df['failureType'].value_counts(dropna=False)
# print('ALL_value\n', ALL_value)
# # trianTestLabel 데이터 파악
# print('\ndf_train')
# df_train = df[df['trianTestLabel'] == 'Training']
# df_train.info()
# train_value = df_train['failureType'].value_counts(dropna=False)
# print(train_value)
# print('\ndf_test')
# df_test = df[df['trianTestLabel'] == 'Test']
# df_test.info()
# test_value = df_test['failureType'].value_counts(dropna=False)
# print('df_test', test_value)

# df=pd.read_pickle("./datasets/LSWMD_drop_None_data.pickle")
# pd.set_option('display.max_columns', None)
# df.dropna(inplace=True)
# df['failureType'].replace('none', 'Normal', inplace=True)
# df = df.drop(columns='trianTestLabel', axis=1)
# df.info()
# print(df.head())
# with open('./datasets/LSWMD_waferMap_failureType.pickle', 'wb') as f:
#     pickle.dump(df, f)

# # wafermap dim 추가
# df=pd.read_pickle('./datasets/LSWMD_waferMap_failureType.pickle')
# pd.set_option('display.max_columns', None)
# df.dropna(inplace=True)
# df.info()
# # print(df.head(50))
#
# # waferMapDim
# def find_dim(x):
#     dim0=np.size(x,axis=0)
#     dim1=np.size(x,axis=1)
#     return dim0, dim1
# df['waferMapDim']=df.waferMap.apply(find_dim)
#
# ## 카테고리 데이터 분류(failureNum)
# df['failureNum']=df.failureType
# mapping_type={'Center':1,'Donut':2,'Edge-Loc':3,'Edge-Ring':4,'Loc':5,'Random':6,'Scratch':7,'Near-full':8,'Normal':0}
# df=df.replace({'failureNum':mapping_type})
# df['failureType'].replace('none', 'Normal', inplace=True)
# df.info()
# print(df)
# with open('./datasets/LSWMD_waferMapDim_failureNum.pickle', 'wb') as f:
#     pickle.dump(df, f)

# df=pd.read_pickle('./datasets/LSWMD_waferMapDim_failureNum.pickle')
# pd.set_option('display.max_columns', None)
# print(df.info())
# count = df['failureType'].value_counts()
# print(count)
# unique = df['waferMapDim'].unique()
# print(max(df.waferMapDim), min(df.waferMapDim))
# print(unique.shape)

# ###################### Density-based Features 13개 ######################
# def cal_den(x):
#     return 100 * (np.sum(x == 2) / np.size(x))
#
# def find_regions(x):
#     rows = np.size(x, axis=0)
#     cols = np.size(x, axis=1)
#     if rows >=5 and cols>=5:
#         ind1 = np.arange(0, rows, rows // 5)
#         ind2 = np.arange(0, cols, cols // 5)
#
#         reg1 = x[ind1[0]:ind1[1], :]
#         reg3 = x[ind1[4]:, :]
#         reg4 = x[:, ind2[0]:ind2[1]]
#         reg2 = x[:, ind2[4]:]
#
#         reg5 = x[ind1[1]:ind1[2], ind2[1]:ind2[2]]
#         reg6 = x[ind1[1]:ind1[2], ind2[2]:ind2[3]]
#         reg7 = x[ind1[1]:ind1[2], ind2[3]:ind2[4]]
#         reg8 = x[ind1[2]:ind1[3], ind2[1]:ind2[2]]
#         reg9 = x[ind1[2]:ind1[3], ind2[2]:ind2[3]]
#         reg10 = x[ind1[2]:ind1[3], ind2[3]:ind2[4]]
#         reg11 = x[ind1[3]:ind1[4], ind2[1]:ind2[2]]
#         reg12 = x[ind1[3]:ind1[4], ind2[2]:ind2[3]]
#         reg13 = x[ind1[3]:ind1[4], ind2[3]:ind2[4]]
#
#         fea_reg_den = []
#         fea_reg_den = [cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4), cal_den(reg5), cal_den(reg6),
#                        cal_den(reg7), cal_den(reg8), cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12),
#                        cal_den(reg13)]
#         return fea_reg_den
#     else: return None
#
# df['fea_reg'] = df.waferMap.apply(find_regions)
# df.dropna(inplace=True)
# df.info()
# with open('./datasets/LSWMD_fea_reg.pickle', 'wb') as f:
#     pickle.dump(df, f)
# print('Save!')

# ###################### Geometry-based Features 6 ######################
# df=pd.read_pickle('./datasets/LSWMD_fea_reg.pickle')
# pd.set_option('display.max_columns', None)
# print(df.info())
# def cal_dist(img, x, y):
#     dim0 = np.size(img, axis=0)
#     dim1 = np.size(img, axis=1)
#     dist = np.sqrt((x - dim0 / 2) ** 2 + (y - dim1 / 2) ** 2)
#     return dist
#
# def fea_geom(img):
#     norm_area = img.shape[0] * img.shape[1]
#     norm_perimeter = np.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2)
#
#     img_labels = measure.label(img, connectivity=1, background=0)     # neighbors=4 옵션 삭제
#
#     if img_labels.max() == 0:
#         img_labels[img_labels == 0] = 1
#         no_region = 0
#     else:
#         info_region = stats.mode(img_labels[img_labels > 0], axis=None, keepdims=True)
#         no_region = info_region[0][0] - 1
#
#     prop = measure.regionprops(img_labels)
#     prop_area = prop[no_region].area / norm_area
#     prop_perimeter = prop[no_region].perimeter / norm_perimeter
#
#     prop_cent = prop[no_region].local_centroid
#     prop_cent = cal_dist(img, prop_cent[0], prop_cent[1])
#
#     prop_majaxis = prop[no_region].major_axis_length / norm_perimeter
#     prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter
#     prop_ecc = prop[no_region].eccentricity
#     prop_solidity = prop[no_region].solidity
#
#     return prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity
#
# df['fea_geom'] = df.waferMap.apply(fea_geom)
# print('df[fea_geom]\n', df)#['fea_geom'])
# df.info()
#
# with open('./datasets/LSWMD_fea_geom.pickle', 'wb') as f:
#     pickle.dump(df, f)
# print('Save!')

# ###################### Randon-based Features 40개 ######################
# df=pd.read_pickle('./datasets/LSWMD_fea_geom.pickle')
# pd.set_option('display.max_columns', None)
# print(df)
# df.info()
# def cubic_inter_mean(img):
#     theta = np.linspace(0., 180., max(img.shape), endpoint=False)
#     sinogram = radon(img, theta=theta)
#     xMean_Row = np.mean(sinogram, axis = 1)
#     x = np.linspace(1, xMean_Row.size, xMean_Row.size)
#     y = xMean_Row
#     f = interpolate.interp1d(x, y, kind = 'cubic')
#     xnew = np.linspace(1, xMean_Row.size, 20)
#     ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
#     return ynew
#
# def cubic_inter_std(img):
#     theta = np.linspace(0., 180., max(img.shape), endpoint=False)
#     sinogram = radon(img, theta=theta)
#     xStd_Row = np.std(sinogram, axis=1)
#     x = np.linspace(1, xStd_Row.size, xStd_Row.size)
#     y = xStd_Row
#     f = interpolate.interp1d(x, y, kind = 'cubic')
#     xnew = np.linspace(1, xStd_Row.size, 20)
#     ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
#     return ynew
#
# df['fea_cub_mean'] = df.waferMap.apply(cubic_inter_mean)
# df['fea_cub_std'] = df.waferMap.apply(cubic_inter_std)
# print('Done')
# print(df)
# df.info()
# print("density-based features",len(df['fea_reg'][0]))
# print("geom-based features", len(df['fea_geom'][0]))
# print("radon-based features mean", len(df['fea_cub_mean'][0]))
# print("radon-based features std", len(df['fea_cub_std'][0]))
# with open('./datasets/LSWMD_fea_cub2.pickle', 'wb') as f:
#     pickle.dump(df, f)
# print('Save!')


# # # 데이터 뻥튀기 해서 학습시키기
# ######################################## 분류별 데이터 수량 맞추기 #################################################
# df=pd.read_pickle('./datasets/LSWMD_fea_cub2.pickle')
# df.reset_index(drop=True, inplace=True)
# pd.set_option('display.max_columns', None)
# print(df.info())
# # failureType value count
# count = df['failureType'].value_counts()
# print(count)
# # # Normal       147431
# # # Edge-Ring      9680   <- 기준
# # # Edge-Loc       5189   *2
# # # Center         4294   *2
# # # Loc            3593   *3
# # # Scratch        1193   *8
# # # Random          866   *
# # # Donut           555
# # # Near-full       149
# df = pd.concat([df, df[df['failureType'] == 'Edge-Loc'], df[df['failureType'] == 'Center'], df[df['failureType'] == 'Loc'], df[df['failureType'] == 'Loc'],
#                 df[df['failureType'] == 'Scratch'], df[df['failureType'] == 'Scratch'], df[df['failureType'] == 'Scratch'], df[df['failureType'] == 'Scratch'],
#                 df[df['failureType'] == 'Scratch'], df[df['failureType'] == 'Scratch'], df[df['failureType'] == 'Scratch'],
#                 df[df['failureType'] == 'Random'], df[df['failureType'] == 'Random'], df[df['failureType'] == 'Random'], df[df['failureType'] == 'Random'],
#                 df[df['failureType'] == 'Random'], df[df['failureType'] == 'Random'], df[df['failureType'] == 'Random'], df[df['failureType'] == 'Random'],
#                 df[df['failureType'] == 'Random'], df[df['failureType'] == 'Random'],
#                 df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'],
#                 df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'],
#                 df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'],
#                 df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'], df[df['failureType'] == 'Donut'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'],
#                 df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full'], df[df['failureType'] == 'Near-full']
#                 ])
# print(df.info())
# count = df['failureType'].value_counts()
# print(count)
# with open('./datasets/LSWMD_value_counts.pickle', 'wb') as f:
#     pickle.dump(df, f)
# print('Save!')

# ###################### Normal 데이터 수량 1만개만 남기기 ######################
# df=pd.read_pickle('./datasets/LSWMD_final_data.pickle')
# pd.set_option('display.max_columns', None)
#
# df_sorted = df[df['failureType'] == 'Normal']
# # print(df_sorted)
# remove_index_Num = df_sorted.iloc[10000].name       ##########
# # print(remove_index_Num)
# indices_to_drop = df[(df['failureType'] == 'Normal') & (df.index >= remove_index_Num)].index
# df.drop(index=indices_to_drop, inplace=True)
# count = df['failureType'].value_counts()
# print(count)
# df.info()
# print(len(df[df['failureType'] == 'Normal']))
#
# with open('./datasets/LSWMD_Normal_count.pickle', 'wb') as f:
#     pickle.dump(df, f)
# print('Save!')


# # ###################### 훈련 데이터와 예측 데이터 나누기 ######################
# df=pd.read_pickle('./datasets/temp/LSWMD_Normal_count.pickle')
# pd.set_option('display.max_columns', None)
# # df.info()
#
# failureType = df['failureType'].unique()
# # print(failureType)
# df_train = pd.DataFrame()
# df_test = pd.DataFrame()
#
# for i in range(len(failureType)):
#     df_temp = df[df['failureType'] == failureType[i]]
#     divided_point = int(len(df_temp)*0.8)
#     # print(divided_point)
#     divided_index = df_temp.iloc[divided_point].name
#     # print(divided_index)
#     train_data = df[(df['failureType'] == failureType[i]) & (df.index < divided_index)]
#     test_data = df[(df['failureType'] == failureType[i]) & (df.index >= divided_index)]
#     # print(train_data)
#     # print(len(train_data), len(test_data))
#     df_train = pd.concat([df_train, train_data])
#     df_test = pd.concat([df_test, test_data])
#
# # df_train.drop(columns=['index'], inplace=True)
# # df_test.drop(columns=['index'], inplace=True)
# df_train.reset_index(inplace=True)
# df_test.reset_index(inplace=True)
# print(df_train, '\n', df_test)
# # print(len(df_train), len(df_test))
# # count_train = df_train['failureType'].value_counts()
# # count_test = df_test['failureType'].value_counts()
# # print(count_train)
# # print(count_test)
# # df_train.info()
# # df_test.info()
# exit()
# with open('./datasets/LSWMD_Train.pickle', 'wb') as f:
#     pickle.dump(df_train, f)
# print('Save!')
# with open('./datasets/LSWMD_Test.pickle', 'wb') as f:
#     pickle.dump(df_test, f)
# print('Save!')


