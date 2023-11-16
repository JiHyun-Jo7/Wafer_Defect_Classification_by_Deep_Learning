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
# columns_to_drop = ['dieSize', 'lotName', 'waferIndex']
# df = df.drop(columns=columns_to_drop, axis=1)
# df.info()
# # print(df.head(10))
# # with open('./datasets/LSWMD_drop_data.pickle', 'wb') as f:
# #     pickle.dump(df, f)
# # mmwrite('./datasets/LSWMD_drop_data.pickle')
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
# # mmwrite('./datasets/LSWMD_drop_None_data.pickle')
#
# # # none값 웨이퍼 이미지보기
# # fig, ax = plt.subplots(nrows = 10, ncols = 10, figsize=(20, 20))
# # ax = ax.ravel(order='C')
# # for i in range(100):
# #     img = df_Normal.waferMap[i]
# #     ax[i].imshow(img)
# #     ax[i].set_title(df_Normal.failureType[i][0][0], fontsize=10)
# #     ax[i].set_xlabel(df_Normal.index[i], fontsize=8)
# #     ax[i].set_xticks([])
# #     ax[i].set_yticks([])
# # plt.tight_layout()
# # plt.show()
# # print(df_category.info())
# # # 0 147431
# # # 1 4294
# # # 2 555
# # # 3 5189
# # # 4 9680
# # # 5 3593
# # # 6 866
# # # 7 1193
# # # 8 149
#
# # wafermap dim 추가
# df=pd.read_pickle("./datasets/LSWMD_drop_None_data.pickle")
# pd.set_option('display.max_columns', None)
# df.dropna(inplace=True)
# df.info()
# # print(df.head(50))
# def find_dim(x):
#     dim0=np.size(x,axis=0)
#     dim1=np.size(x,axis=1)
#     return dim0, dim1
# df['waferMapDim']=df.waferMap.apply(find_dim)
# ## 카테고리 데이터 숫자 대체(failureNum, trainTestNum)
# df['failureNum']=df.failureType
# df['trainTestNum']=df.trianTestLabel
# mapping_type={'Center':1,'Donut':2,'Edge-Loc':3,'Edge-Ring':4,'Loc':5,'Random':6,'Scratch':7,'Near-full':8,'none':0}
# mapping_traintest={'Training':0,'Test':1}
# df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})
# df['failureType'].replace('none', 'Normal', inplace=True)
# df.info()
# print(df)
# # df.to_csv('./datasets/LSWMD_clean_data.csv', index=False)
# # with open('./datasets/LSWMD_clean_data.pickle', 'wb') as f:
# #     pickle.dump(df, f)
# # mmwrite('./datasets/LSWMD_clean_data.pickle')

df_clean=pd.read_pickle("./datasets/LSWMD_clean_data.pickle")
pd.set_option('display.max_columns', None)
df_clean.info()



# # 데이터 뻥튀기 해서 학습시키기
# # 보간법으로 데이터 사이즈 맞추기