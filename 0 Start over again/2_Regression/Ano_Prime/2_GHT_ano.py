import xarray as xr
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.util import add_cyclic_point #进行循环
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要

from Mask_Ocean_Land import mask_land

#import matplotlib.ticker as ticker
#import cartopy.mpl.ticker as cticker #给X轴添加经纬度
#from cartopy import mpl

#--读取数据--#
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA5_GHT_1979-2022_JJA_25.nc')
lat, lon = f1.latitude, f1.longitude
Air = f1.z.sel(level=500, expver=1) / 9.8
# lat_range = lat[(lat>=0)&(lat<=90)]
# Air = f1.air.sel(level=1000).loc[(f1.time.dt.month.isin([6, 7, 8]))].loc['1979-06-01':'2022-08-01'].sel(lat=lat_range)
Air_Sum_Mean = Air.groupby('time.year').mean()
# 计算Temperature_Climate_Mean, 19_ano, 20_ano, 21_ano, 22_ano
Air_cli_mean = Air_Sum_Mean.loc['1979':'2000'].mean(dim='year')
Air_21_ano = Air_Sum_Mean.sel(year=2021) - Air_cli_mean
Air_22_ano = Air_Sum_Mean.sel(year=2022) - Air_cli_mean

#--绘图系列--#
# 数据准备
Air_21_ano = mask_land(Air_21_ano,label='ocean', lonname='longitude')
Air_22_ano = mask_land(Air_22_ano,label='ocean', lonname='longitude')

Air_21_ano, cyclic_lons = add_cyclic_point(Air_21_ano, coord=lon)
Air_22_ano, cyclic_lons = add_cyclic_point(Air_22_ano, coord=lon)


def drawing(cyclic_data1, cyclic_data2):
    # 设置字体为楷体,显示负号
    mpl.rcParams['font.sans-serif'] = ['KaiTi']
    mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=0))
    # 图名
    ax1.set_title('GHT anomalies in 2021')
    ax2.set_title('GHT anomalies in 2022')
    # 刻度线形式
    ax1.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]) #指定要显示的经纬度
    ax1.set_yticks([0, 30, 60, 90])
    ax1.xaxis.set_major_formatter(LongitudeFormatter()) #刻度格式转换为经纬度样式
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    ax2.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
    ax2.set_yticks([0, 30, 60, 90])
    ax2.xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
    ax2.yaxis.set_major_formatter(LatitudeFormatter())
    # 调整图的内容：投影方式，海岸线
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4) #添加海岸线
    ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4) #添加海岸线
    ax_cf1 = ax1.contourf(cyclic_lons, lat, cyclic_data1, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-50, 55, 5), extend='both')  # 绘制等值线图
    ax_cf2 = ax2.contourf(cyclic_lons, lat, cyclic_data2, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-50, 55, 5), extend='both')  # 绘制等值线图
    fig.subplots_adjust(right=0.9)
    position = fig.add_axes([0.93, 0.25, 0.015, 0.5])  # 位置[左,下,右,上]
    cb = fig.colorbar(ax_cf2, shrink=0.6, cax=position, extend='both')
    # ax1.plot([30, 60],[60,30]) #
    # plt.savefig(r'F:\6_Scientific_Research\3_Teleconnection\2_Picture\Background\2_Temperature_Ano_19_20.png')
    plt.savefig(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\0_Background\1_ERA_500hPa_GHT_Ano_Land.png')
    plt.show()
drawing(Air_21_ano, Air_22_ano)