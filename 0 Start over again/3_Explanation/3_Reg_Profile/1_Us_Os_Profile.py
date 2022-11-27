import xarray as xr
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
# 读取CIR_H
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_H_197901-202208.nc')
# 读取风速
f3 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\vs_H_197901-202208.nc')
f2 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\os_H_197901-202208.nc')

# 读取
# 筛选并计算夏季平均
lon = f1.lon
lat = np.arange(90, -92.5, -2.5)
Win_Index = [i for i in range(524) if i%12 == 0] # 筛选JJA
Cir_H = f1.data[Win_Index, :, :, :].mean(dim='lon').mean(dim='time')# 筛选NH
v = f3.data[Win_Index, :, :, :].mean(dim='lon').mean(dim='time')
w = f2.data[Win_Index, :, :, :].mean(dim='lon').mean(dim='time')
w = w * (-100)
height= Cir_H.height
# year = range(1979, 2023)
# Cir_H_Sum = xr.DataArray(Cir_H_Sum, coords=[year, height, lat, lon], dims=['year', 'height', 'lat', 'lon'])
# vs = xr.DataArray(vs, coords=[year, height, lat, lon], dims=['year', 'height', 'lat', 'lon'])
# ws = xr.DataArray(ws, coords=[year, height, lat, lon], dims=['year', 'height', 'lat', 'lon'])

fig = plt.figure(figsize=(6, 4), dpi=600)
# fig.subplots_adjust(hspace=0.6 ) # 子图间距
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('JAN CIR_H (-w)', loc='left', fontsize=6)
# 纵坐标设置
ax1.set_yscale('symlog')
ax1.set_yticks([1000, 500, 300, 200])
ax1.set_yticklabels(['1000', '500', '300', '200'], fontdict={'fontsize': 6})
ax1.invert_yaxis()
ax1.set_ylabel('Level (hPa)', fontsize=6)
# 横坐标设置
ax1.set_xlim(-90, 90)
ax1.set_xlabel('Latitude', fontsize=6)
ws = ax1.get_xticks()  # 获取横坐标值列表，主要为了调节大小
ax1.set_xticklabels(ws, fontdict={'fontsize': 6})
ax_cf1 = ax1.contourf(lat, height[0:12], Cir_H[0:12, :], levels=np.arange(-0.04, 0.044, 0.004), extend='both',
                         zorder=0, cmap='coolwarm')
ax1.quiver(lat, height[0:12], v[0:12, :], w[0:12, :])
ax1.xaxis.set_major_formatter(cticker.LatitudeFormatter())
fig.subplots_adjust(bottom=0.13)
plt.rcParams['font.size'] = 6
position = fig.add_axes([0.15, 0.05, 0.7, 0.015])  # 位置[左,下,右,上]
cb = fig.colorbar(ax_cf1, shrink=0.6, cax=position, orientation='horizontal', extend='both')

plt.show()

