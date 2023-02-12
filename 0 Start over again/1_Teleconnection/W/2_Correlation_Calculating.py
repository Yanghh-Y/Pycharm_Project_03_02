import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr

#---Read_Data---#
f1 = xr.open_dataset(r'Z:\6_Scientific_Research\3_Teleconnection\Stream\Stream_nc\cir_W_197901-202208.nc')
# 筛选夏季平均
# 夏季平均的CIR_H_Sum[44, 37, 144]
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_W = f1.data[Sum_Index, 5, 0:27, :] # 筛选
Cir_W = Cir_W - Cir_W.mean(dim='lon') # 去纬度平均
Cir_W_Sum = np.zeros((44, 27, 144))
lat, lon = Cir_W.lat, Cir_W.lon
year = range(1979, 2023)
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_W_Sum[n, :, :] = (Cir_W[i, :, :] + Cir_W[i + 1, :, :] + Cir_W[i + 2, :, :]) / 3
        n = n + 1
Cir_W_Sum = xr.DataArray(Cir_W_Sum, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])

# # 计算某两点的相关系数
# r = np.zeros((27, 144))
# for i in range (27):
#     for j in range(144):
#         r[i, j], _ = pearsonr(Cir_W_Sum[:, 1, 1], Cir_W_Sum[:, i, j])

# 某两点
cor, p = pearsonr(Cir_W_Sum.sel(lat=70, lon=355), Cir_W_Sum.sel(lat=45, lon=340))
# cor, p = pearsonr(Cir_W_Sum[:, 1, 1], Cir_W_Sum[:, 1, 72])













