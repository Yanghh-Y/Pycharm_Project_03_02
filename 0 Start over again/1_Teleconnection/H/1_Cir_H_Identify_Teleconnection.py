import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr
# from __future__ import division # python2精确除法

# ---Read_Data--- #
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_H_197901-202208.nc')
# 筛选夏季平均
# 夏季平均的CIR_H_Sum[44, 37, 144]
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_H = f1.data[Sum_Index, 5, 0:27, :] # 筛选
Cir_H_Sum = np.zeros((44, 27, 144))
lat, lon = Cir_H.lat, Cir_H.lon
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_H_Sum[n, :, :] = (Cir_H[i, :, :] + Cir_H[i + 1, :, :] + Cir_H[i + 2, :, :]) / 3
        n = n + 1

#--识别遥相关--# #
# 计算相关
Cor_R, Cor_P = np.zeros((27, 144, 27, 144)), np.zeros((27, 144, 27, 144))
for ilat in range(0, 27): # ilat[90, 25, -2.5]
    for ilon in range(0, 144):# ilon [0, 360, 2.5]
        for dlat in range(0, 27):
            for dlon in range(0, 144):
                Cor_R[ilat, ilon, dlat, dlon], Cor_P[ilat, ilon, dlat, dlon] = pearsonr(Cir_H_Sum[:, ilat, ilon], Cir_H_Sum[:, dlat, dlon])
                print(int(lat[ilat]), int(lon[ilon]), int(lat[dlat]), int(lon[dlon])) # 随时看计算到何处

# 存入NC文件
ilon = lon
ilat = lat
dlon = lon
dlat = lat
Cor_R = xr.DataArray(Cor_R, coords=[ilat, ilon, dlat, dlon], dims=['ilat', 'ilon', 'dlat', 'dlon'])
dr = xr.Dataset({'Cor_R': Cor_R})
dr.to_netcdf(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\Cor_R.nc')
Cor_P = xr.DataArray(Cor_P, coords=[ilat, ilon, dlat, dlon], dims=['ilat', 'ilon', 'dlat', 'dlon'])
dp = xr.Dataset({'Cor_R': Cor_R})
dp.to_netcdf(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\Cor_P.nc')


#---筛选每个点上最小的min_r---#
RR = np.array(Cor_R)
Min_R = np.zeros((27, 144))
Min_P = np.zeros((27, 144))
lat_min = np.zeros((27,144))
lon_min = np.zeros((27,144))
for i in range(27): # ilat
    print(i)
    for j in range(144): # ilon
        Min_R[i, j] = np.min(RR[i, j, :, :]) # 将[i,j] 点上最小的相关系数存入Min_R
        location = np.where(RR[i, j, :, :] == Min_R[i, j]) #取(ilat，ilon)所对应最小的相关系数的那个坐标
        if len(location[0])>1: # 如果不止一个点的最小相关系数是一样的
            lat_min[i, j] = location[0][0]
            lon_min[i, j] = location[1][0]
            Min_P[i, j] = Cor_P[i, j, location[0][0], location[1][0]]
        if len(location[0])==1:
            lat_min[i, j] = location[0]
            lon_min[i, j] = location[1]
            Min_P[i, j] = Cor_P[i, j, location[0], location[1]]
# 筛选P值，当显著性<99%，我们直接将其相关性定为0
for i in range(27):
    for j in range(144):
        if Min_R[i, j] > -0.35 or Min_P[i, j] > 0.01 :
            Min_R[i, j] = np.nan

# 将每个点上的min_r
Min_R = xr.DataArray(Min_R, coords=[lat, lon], dims=['lat', 'lon'])
ds = xr.Dataset({'Min_R': Min_R})
ds.to_netcdf(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\Min_R.nc')
Min_P = xr.DataArray(Min_P, coords=[lat, lon], dims=['lat', 'lon'])
da = xr.Dataset({'Min_p': Min_P})
da.to_netcdf(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\Min_p.nc')
lat_min = xr.DataArray(lat_min, coords=[lat, lon], dims=['lat', 'lon'])
dla = xr.Dataset({'lat_min': lat_min})
dla.to_netcdf(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\lat_min.nc')
lon_min = xr.DataArray(lon_min, coords=[lat, lon], dims=['lat', 'lon'])
dlo = xr.Dataset({'lon_min': lon_min})
dlo.to_netcdf(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\lon_min.nc')


