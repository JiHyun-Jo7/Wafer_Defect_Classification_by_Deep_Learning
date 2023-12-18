# https://www.kaggle.com/code/ashishpatel26/wm-811k-wafermap
import numpy as np
import pandas as pd
from scipy.io import mmwrite, mmread
import pickle
import matplotlib.pyplot as plt
# loading libraries
import skimage
from skimage import measure
from skimage.transform import radon
from skimage.transform import probabilistic_hough_line
from skimage import measure
from scipy import interpolate
from scipy import stats

###################### 입력 데이터 추합하기(Concat) ######################
df=pd.read_pickle('./datasets/LSWMD_final_data.pickle')
# df.reset_index(drop=True, inplace=True)
# pd.set_option('display.max_columns', None)
print(df)
df.info()
# count = df['failureType'].value_counts()
# print('\n', count)
column_types = type(df['waferMap'][0])
print(column_types)
