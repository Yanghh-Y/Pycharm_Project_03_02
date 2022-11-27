import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr
# from __future__ import division # python2精确除法

#---Read_Data---#
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_R_197901-202208.nc')
# 筛选夏季平均
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_R = f1.data[Sum_Index, 5, 0:37, :] # 筛选
Cir_R_Sum = np.zeros((44, 37, 144))
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_R_Sum[n, :, :] = (Cir_R[i, :, :] + Cir_R[i+1, :, :] + Cir_R[i+2, :, :])/3
        n = n+1
        print(n)
# 去线性趋势
Cir_R_detrend = scipy.signal.detrend(Cir_R_Sum, axis=0, type='linear', overwrite_data=False)
Cir_R_year_mean = Cir_R_Sum.mean(axis=0)
for iyear in range(44):
    Cir_R_detrend[iyear, :, :] = Cir_R_detrend[iyear, :, :] + Cir_R_year_mean[:, :]

#-----drawing-----#
# 创建画布和绘图对象
fig = plt.figure(figsize=(8, 6), dpi=400)
ax = fig.add_subplot(1, 1, 1)
# 处理中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 绘制折线图
ax.plot(range(39), Cir_R_Sum[5:,20,21], c='blue')
#ax.plot(Temp.YEAR, Temp.T_Detrend_constant_bp_1980, c='blue')
# ax.plot(range(39), Cir_R_Sum[5:,20,20], c='red')
plt.show()
