import xarray as xr
import scipy
from scipy import signal # 用于去线性趋势

import pandas as pd
from global_land_mask import globe
# import cmaps
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.util import add_cyclic_point #进行循环
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要
import matplotlib.patches as patches

#import matplotlib.ticker as ticker
#import cartopy.mpl.ticker as cticker #给X轴添加经纬度
#from cartopy import mpl

def mask_land(ds, label='land', lonname='lon'):
    if lonname == 'lon':
        lat = ds.lat.data
        lon = ds.lon.data
        if np.any(lon > 180):
            lon = lon - 180
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
            temp = []
            temp = mask[:, 0:(len(lon) // 2)].copy()
            mask[:, 0:(len(lon) // 2)] = mask[:, (len(lon) // 2):]
            mask[:, (len(lon) // 2):] = temp
        else:
            lons, lats = np.meshgrid(lon, lat)# Make a grid
            mask = globe.is_ocean(lats, lons)# Get whether the points are on ocean.
        ds.coords['mask'] = (('lat', 'lon'), mask)
    elif lonname == 'longitude':
        lat = ds.latitude.data
        lon = ds.longitude.data
        if np.any(lon > 180):
            lon = lon - 180
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
            temp = []
            temp = mask[:, 0:(len(lon) // 2)].copy()
            mask[:, 0:(len(lon) // 2)] = mask[:, (len(lon) // 2):]
            mask[:, (len(lon) // 2):] = temp
        else:
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
        lons, lats = np.meshgrid(lon, lat)
        mask = globe.is_ocean(lats, lons)
        ds.coords['mask'] = (('latitude', 'longitude'), mask)
    if label == 'land':
        ds = ds.where(ds.mask == True)
    elif label == 'ocean':
        ds = ds.where(ds.mask == False)
    return ds

#--读取数据--#
f1 = xr.open_dataset(r'Z:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA5_T_U_V_1.nc')
SAT = f1.t.sel(level=1000).groupby('time.year').mean(dim='time', skipna=True)
# SAT = mask_land(SAT,label='ocean', lonname='longitude')
lat = SAT.latitude
lon = SAT.longitude
# SAT气候态平均
SAT_CL = SAT.loc[1979:2000].mean('year')
# Air_21_ano = SAT.sel(year=2021) - SAT_CL
# Air_22_ano = SAT.sel(year=2022) - SAT_CL
# --- Calculating - SAT_ANO --- #
SAT_ANO = np.zeros((44, 91, 360))
SAT_detrend = scipy.signal.detrend(SAT, axis=0, type='linear', overwrite_data=False)
SAT_year_mean = SAT.mean('year')
for iyear in range(44):
    SAT_detrend[iyear, :, :] = SAT_detrend[iyear, :, :] + SAT_year_mean[:, :]
    SAT_ANO[iyear, :, :] = SAT_detrend[iyear, :, :] - SAT_CL[:, :]
# --- 提取线性分量 --- #
SAT_linear = SAT - SAT_detrend
SAT_linear_2021 = SAT_linear[-2, :, :]
SAT_linear_2022 = SAT_linear[-1, :, :]
Air_21_ano = SAT_ANO[42, :, :] + SAT_linear_2021
Air_22_ano = SAT_ANO[43, : ,:] + SAT_linear_2022

#--绘图系列--#
# 数据准备
Air_21_ano = mask_land(Air_21_ano,label='ocean', lonname='longitude')
Air_22_ano = mask_land(Air_22_ano,label='ocean', lonname='longitude')
Air_21_ano, cyclic_lons = add_cyclic_point(Air_21_ano, coord=lon)
Air_22_ano, cyclic_lons = add_cyclic_point(Air_22_ano, coord=lon)
# 绘制图形
rect11 = patches.Rectangle((-125, 37.5), 27.5, 12.5, linewidth=2, edgecolor='k', fill = False)
rect12 = patches.Rectangle((30, 47.5), 30, 15, linewidth=2, edgecolor='k', fill = False)
rect13 = patches.Rectangle((105, 60), 40, 12.5, linewidth=2, edgecolor='k', fill = False)
rect21 = patches.Rectangle((-115, 62.5), 30, 10, linewidth=2, edgecolor='k', fill = False)
rect22 = patches.Rectangle((-7.5, 30), 27.5, 17.5, linewidth=2, edgecolor='k', fill = False)
rect23 = patches.Rectangle((100, 20), 20, 15, linewidth=2, edgecolor='k', fill = False)





def drawing(cyclic_data1, cyclic_data2):
    # 设置字体为楷体,显示负号
    mpl.rcParams['font.sans-serif'] = ['KaiTi']
    mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=0))
    # 图名
    ax1.set_title('ERA5 1000hPa TEM Ano in 2021')
    ax2.set_title('ERA5 1000hPa TEM Ano in 2022')
    # 刻度线形式
    ax1.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]) #指定要显示的经纬度
    ax1.set_yticks([0, 30, 60, 90])
    ax1.xaxis.set_major_formatter(LongitudeFormatter()) #刻度格式转换为经纬度样式
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    ax2.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
    ax2.set_yticks([0, 30, 60, 90])
    ax2.xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
    ax2.yaxis.set_major_formatter(LatitudeFormatter())
    # 添加图形
    ax1.add_patch(rect11)
    ax1.add_patch(rect12)
    ax1.add_patch(rect13)
    ax2.add_patch(rect21)
    ax2.add_patch(rect22)
    ax2.add_patch(rect23)
    # 调整图的内容：投影方式，海岸线
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4) #添加海岸线
    ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4) #添加海岸线
    ax2.add_feature(cfeature.BORDERS, zorder=0) # 添加国家边界
    ax_cf1 = ax1.contourf(cyclic_lons, lat, cyclic_data1, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-3, 3.2, 0.2), extend='both')  # 绘制等值线图
    ax_cf2 = ax2.contourf(cyclic_lons, lat, cyclic_data2, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-3, 3.2, 0.2), extend='both')  # 绘制等值线图
    fig.subplots_adjust(right=0.9)
    position = fig.add_axes([0.93, 0.25, 0.015, 0.5])  # 位置[左,下,右,上]
    cb = fig.colorbar(ax_cf2, shrink=0.6, cax=position, extend='both')
    # ax1.plot([30, 60],[60,30]) #
    plt.savefig(r'Z:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\0_Background\1_ERA_1000hPa_Tem_Ano_Land_region.png')
    plt.show()
drawing(Air_21_ano, Air_22_ano)