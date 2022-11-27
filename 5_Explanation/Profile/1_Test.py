import xarray as xr
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
import scipy
from scipy import signal
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import linregress


#--读取数据--#
# 读取遥相关指数
TELIS = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\HI.csv', usecols=['B1B2', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
# 读取遥相关两点经纬度
T_Point = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\H_Tel_Manual.csv')
# 读取CIR_H
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA5_GHT_1979-2022_JJA_25.nc')
lat = f1.latitude
lon = f1.longitude
level = f1.level
year = range(1979, 2023)
# 选时间,求平均 hgt [44, 9, 91, 360]
hgt = f1.z.sel(expver=1).groupby('time.year').mean(dim='time', skipna=True)
hgt = hgt.sel(year=range(1979,2023))
hgt = hgt/9.8
# 去线性趋势 hgt_detrend[44, 9, 37, 144] [year, height, lon, lat]
hgt_detrend = scipy.signal.detrend(hgt, axis=0, type='linear', overwrite_data=False)
hgt_year_mean = hgt.mean('year')
for iyear in range(44):
    hgt_detrend[iyear, :, :, :] = hgt_detrend[iyear, :, :, :] + hgt_year_mean[:, :, :]
hgt_detrend = xr.DataArray(hgt_detrend, coords=[year, level, lat, lon], dims=['year', 'level', 'lat', 'lon'])

lat_ver=(np.linspace(65, 72.5, 50))
lon_ver=(np.linspace(260, 292.5, 50))
H_ver = hgt_detrend.interp(lat=lat_ver, lon=lon_ver) # [44, 9, 50, 50] 将两点所组成的正方形区域内的格点圈了起来
H_ver = np.array(H_ver).diagonal(axis1=2,axis2=3)

lat_ver, lon_ver = [], []
k = 0
# 计算插值部分
lat_ver.append(np.linspace(T_Point.Point1_lat[k], T_Point.Point2_lat[k], 50))
lon_ver.append(np.linspace(T_Point.Point1_lon[k], T_Point.Point2_lon[k], 50))
H_ver = hgt_detrend.interp(lat=lat_ver[k], lon=lon_ver[k]) # [44, 9, 50, 50] 将两点所组成的正方形区域内的格点圈了起来
H_ver = np.array(H_ver).diagonal(axis1=2,axis2=3) # [44, 9, 50] 在这个正方形区域内求对角线上的数据
