import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr
# from __future__ import division # python2精确除法

#---Read_Data---#
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_H_197901-202208.nc')
# 筛选夏季平均
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_H = f1.data[Sum_Index, 5, 0:37, :] # 筛选
Cir_H_Sum = np.zeros((44, 37, 144))
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_H_Sum[n, :, :] = (Cir_H[i, :, :] + Cir_H[i + 1, :, :] + Cir_H[i + 2, :, :]) / 3
        n = n + 1
# 去线性趋势
Cir_H_detrend = scipy.signal.detrend(Cir_H_Sum, axis=0, type='linear', overwrite_data=False)
Cir_H_year_mean = Cir_H_Sum.mean(axis=0)
for iyear in range(44):
    Cir_H_detrend[iyear, :, :] = Cir_H_detrend[iyear, :, :] + Cir_H_year_mean[:, :]
r, p = np.zeros((37,144,37,144)), np.zeros((37,144,37,144))

#--识别遥相关--# #
# 计算相关
for ilat in range(0,37):
    for ilon in range(0,144):
        for dlat in range(0,37):
            for dlon in range(0,144):
                r[ilat, ilon, dlat, dlon], p[ilat, ilon, dlat, dlon] = pearsonr(Cir_H_detrend[:, ilat, ilon], Cir_H_detrend[:, dlat, dlon])
                print(r[ilat, ilon, dlat, dlon])

# 存入NC文件
ilon = np.arange(0, 360, 2.5)
ilat = np.arange(90, -2.5, -2.5)
dlon = np.arange(0, 360, 2.5)
dlat = np.arange(90, -2.5, -2.5)

r = xr.DataArray(r, coords=[ilat, ilon, dlat, dlon], dims=['ilat', 'ilon', 'dlat', 'dlon'])
ds = xr.Dataset({'r': r})
ds.to_netcdf(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\R_P\Cir_H_r_all_lon_lat.nc')
p = xr.DataArray(p, coords=[ilat, ilon, dlat, dlon], dims=['ilat', 'ilon', 'dlat', 'dlon'])
dp = xr.Dataset({'p': p})
dp.to_netcdf(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\R_P\Cir_H_p_all_lon_lat.nc')







