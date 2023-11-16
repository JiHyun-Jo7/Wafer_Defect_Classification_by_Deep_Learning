import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.io import mmwrite, mmread
import pickle
import matplotlib.pyplot as plt

# 카테고리별 데이터양 맞춰주기
df_clean=pd.read_pickle("./datasets/LSWMD_clean_data.pickle")
pd.set_option('display.max_columns', None)
# df_clean.info()

mapping_type={'Center':1,'Donut':2,'Edge-Loc':3,'Edge-Ring':4,'Loc':5,'Random':6,'Scratch':7,'Near-full':8,'Normal --':0}
for i in range(len(mapping_type)):
    df = df_clean[df_clean['failureNum'] == i]
    print(df['failureType'][0], df)
    df.info()
    df_test = df[df[['trianTestLabel'] == 'Test']]
    df_Training = df[df[['trianTestLabel'] == 'Training']]
    df_test.info()
    df_Training.info()

    # df.to_csv('./datasets/LSWMD_clean_data.csv', index=False)
    # with open('./datasets/LSWMD_clean_data.pickle', 'wb') as f:
    #     pickle.dump(df, f)
    # mmwrite('./datasets/LSWMD_clean_data.pickle')