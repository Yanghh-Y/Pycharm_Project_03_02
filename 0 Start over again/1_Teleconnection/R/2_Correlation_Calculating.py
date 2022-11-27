import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr

#---Read_Data---#
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_R_197901-202208.nc')
# 筛选夏季平均
# 夏季平均的CIR_H_Sum[44, 37, 144]
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_R = f1.data[Sum_Index, 5, 0:27, :] # 筛选
Cir_R_Sum = np.zeros((44, 27, 144))
lat, lon = Cir_R.lat, Cir_R.lon
year = range(1979, 2023)
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_R_Sum[n, :, :] = (Cir_R[i, :, :] + Cir_R[i + 1, :, :] + Cir_R[i + 2, :, :]) / 3
        n = n + 1
Cir_R_Sum = xr.DataArray(Cir_R_Sum, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])

# # 计算某两点的相关系数
# r = np.zeros((27, 144))
# for i in range (27):
#     for j in range(144):
#         r[i, j], _ = pearsonr(Cir_R_Sum.sel(lat=42.5, lon=47.5), Cir_R_Sum[:, i, j])

# 某两点
cor, p = pearsonr(Cir_R_Sum.sel(lat=87.5, lon=347.5), Cir_R_Sum.sel(lat=87.5, lon=167.5))
# cor, p = pearsonr(Cir_R_Sum[:, 4, 1], Cir_R_Sum[:, 10, 128])













