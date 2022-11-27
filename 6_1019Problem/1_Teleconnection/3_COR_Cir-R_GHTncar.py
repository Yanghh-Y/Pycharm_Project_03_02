import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.util import add_cyclic_point #进行循环
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要
from Longitude_Transform import translon180_360


#---Read_Data---#
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_R_197901-202208.nc')
# 筛选夏季平均
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_R = f1.data[Sum_Index, 5, 0:37, :] # 筛选
Cir_R_Sum = np.zeros((42, 37, 144))
lat = Cir_R.lat
lon = Cir_R.lon
year = np.arange(1979, 2021)
n = 0
for i in range(126): # 计算夏季平均
    if i%3 == 0:
        Cir_R_Sum[n, :, :] = (Cir_R[i, :, :] + Cir_R[i+1, :, :] + Cir_R[i+2, :, :])/3
        n = n + 1
# Cir_R_Sum = xr.DataArray(Cir_R_Sum, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])


# Read-GHT
# hgt-1950-2019-summer-mean
filepath = (r'F:\7_Data\3_NCAR\2_Geopotential_Height\hgt.mon.mean.nc')
# lon_name = 'longitude'
# f2 = translon180_360(filepath, lon_name)
f2 = xr.open_dataset(filepath)
# 选时间，高度，求平均
hgt = f2.hgt.loc[(f2.time.dt.month.isin([6, 7, 8]))].loc['1979-01-01':'2020-08-01'].sel(level=500).sel(lat=np.arange(0, 92.5, 2.5)).groupby('time.year').mean(dim='time', skipna=True)
year = (1979, 2021)
# 去线性趋势
hgt_detrend = scipy.signal.detrend(hgt, axis=0, type='linear', overwrite_data=False)
# 求相关
r, p = np.zeros((37, 144)), np.zeros((37, 144))
for ilat in range(37):
    for ilon in range(144):
        r[ilat, ilon], p[ilat, ilon] = pearsonr(Cir_R_Sum[:, ilat, ilon], hgt_detrend[:, ilat, ilon])



r, cyclic_lons = add_cyclic_point(r, coord=lon)
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
fig = plt.figure(figsize=(10, 3), dpi=400)
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
ax1.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
ax1.set_yticks([0, 30, 60, 90])
ax1.xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.8)  # 添加海岸线
ax_cf1 = ax1.contourf(cyclic_lons, lat, r, transform=ccrs.PlateCarree(), cmap='coolwarm',
                      levels=np.arange(-0.8, 0.85, 0.05 ), extend='both')  # 绘制等值线图
cb1 = fig.colorbar(ax_cf1, shrink=0.55, orientation='horizontal', extend='both')  # shrin收缩比例
plt.show()


