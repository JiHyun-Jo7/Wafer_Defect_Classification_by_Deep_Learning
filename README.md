# 🌑🔍Wafer_Defect_Classification_by_Deep_Learning
- 
- Gahee Jung, Jihyun Jo
---
## 🖥️ 개발 환경(IDE)
- Win 10 & 11
- Pycharm 2023.2
- Python 3.8.1
- [Download Pickle](https://www.kaggle.com/code/cchou1217/wm-811k-wafermap)
---
## Preprocessing
```
df = pd.read_pickle("./datasets/LSWMD.pkl")

<class 'pandas.core.frame.DataFrame'>  
RangeIndex: 811457 entries, 0 to 811456  
Data columns (total 6 columns):  
waferMap          811457 non-null object  
dieSize           811457 non-null float64  
lotName           811457 non-null object  
waferIndex        811457 non-null float64  
trianTestLabel    811457 non-null object  
failureType       811457 non-null object  
dtypes: float64(2), object(4)  
memory usage: 37.1+ MB
```
- 
- 

---

