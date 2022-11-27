import xarray as xr
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.util import add_cyclic_point #进行循环
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要
import os

f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\R_P\cir_H_r_min_arr.nc', engine="netcdf4")
lat = np.arange(90, 22.5, -2.5)
lon = np.arange(0, 360, 2.5)
R = f1.r_min_r
R = R.fillna(0.4)
Point_df = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\R_P\Cir_H_Point.csv')
# 由于遥相关太多我们选择将其通过相关系数的不同进行分类
Point_DF = pd.DataFrame(columns=['Min_cor', 'A_lat', 'A_lon', 'B_lat', 'B_lon'])
Point_DF['Min_cor'] = Point_df['Min_cor']
for i in range(len(Point_df['Min_cor'])):
    Point_DF.loc[i, 'A_lat'] = lat[Point_df.loc[i, 'A_lat']]
    Point_DF.loc[i, 'A_lon'] = lon[Point_df.loc[i, 'A_lon']]
    Point_DF.loc[i, 'B_lat'] = lat[int(Point_df.loc[i, 'B_lat'])]
    Point_DF.loc[i, 'B_lon'] = lon[int(Point_df.loc[i, 'B_lon'])]
Point_DF = Point_DF[((Point_DF['Min_cor']<-0.6) & (Point_DF['Min_cor']>-0.7))]
Point_DF.reset_index(drop=True, inplace=True)


R, cyclic_lons = add_cyclic_point(R, coord=lon)
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
fig = plt.figure(figsize=(10, 3), dpi=400)
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=290))
ax1.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
ax1.set_yticks([0, 30, 60, 90])
ax1.xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.8)  # 添加海岸线

# 添加线段
for i in range(len(Point_DF.index)):
    A_lon = Point_DF.loc[i, 'A_lon']
    A_lat = Point_DF.loc[i, 'A_lat']
    B_lon = Point_DF.loc[i, 'B_lon']
    B_lat = Point_DF.loc[i, 'B_lat']
    # ax1.plot([A_lon,B_lon], [A_lat, B_lat], 'b', '.', transform=ccrs.PlateCarree(), linewidth=0.5)
# ax1.plot([292.5,260], [72.5, 65], 'b', '.', transform=ccrs.PlateCarree(), linewidth=0.5)

# # 添加点
# ax1.scatter(82.5, 130, c='y', s=5, transform=ccrs.PlateCarree())
ax_cf1 = ax1.contourf(cyclic_lons, lat, R, transform=ccrs.PlateCarree(), cmap='hot',
                      levels=np.arange(-0.8, -0.45, 0.05 ), extend='both')  # 绘制等值线图
cb1 = fig.colorbar(ax_cf1, shrink=0.55, orientation='horizontal')  # shrin收缩比例
# plt.savefig(r'F:\6_Scientific_Research\3_Teleconnection\2_Picture\2_Teleconnection\Cir_H_Teleconnectivity matrix_1.png')

plt.show()




