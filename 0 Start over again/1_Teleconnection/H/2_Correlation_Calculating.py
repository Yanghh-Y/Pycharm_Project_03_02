import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr

# #---Read_Data---#
# f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_H_197901-202208.nc')
# # 筛选夏季平均
# # 夏季平均的CIR_H_Sum[44, 37, 144]
# Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
# Cir_H = f1.data[Sum_Index, 5, 0:27, :] # 筛选
# Cir_H_Sum = np.zeros((44, 27, 144))
# lat, lon = Cir_H.lat, Cir_H.lon
# year = range(1979, 2023)
# n = 0
# for i in range(132): # 计算夏季平均
#     if i%3 == 0:
#         Cir_H_Sum[n, :, :] = (Cir_H[i, :, :] + Cir_H[i + 1, :, :] + Cir_H[i + 2, :, :]) / 3
#         n = n + 1
# Cir_H_Sum = xr.DataArray(Cir_H_Sum, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])
#
# # 计算某两点的相关系数
# r = np.zeros((27, 144))
# for i in range (27):
#     for j in range(144):
#         r[i, j], _ = pearsonr(Cir_H_Sum.sel(lat=42.5, lon=47.5), Cir_H_Sum[:, i, j])

# 某两点
Mean_weieht = [9.0556,9.7272,5.174333333,3.269,5.2164]
a = [1.024519712,0.64393832,0.298677455,0.211414285,0.734586033]
cor, p = pearsonr(Mean_weieht, a)
# cor, p = pearsonr(Cir_H_Sum[:, 4, 1], Cir_H_Sum[:, 10, 128])













