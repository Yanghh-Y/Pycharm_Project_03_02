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


#---读取数据---#
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\R_P\cir_W_De_Lattice_r_min_arr.nc', engine="netcdf4")
lat, lon = f1.lat, f1.lon
R = f1.r_min_r
R = R.fillna(0.4)
Point_df = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\R_P\Cir_W_De_Lattice_Point.csv')
# Point_df中其经纬度只是经纬度的index


#---绘图---#
R, cyclic_lons = add_cyclic_point(R, coord=lon)
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
fig = plt.figure(figsize=(10, 3), dpi=400)
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
ax1.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
ax1.set_yticks([0, 30, 60, 90])
ax1.xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.8)  # 添加海岸线

# # 添加线段
# mincor = np.arange(-0.9, -0.5, 0.01)
# for i in range(len(Point_df.index)):
#     n = 19
#     if Point_df.loc[i, 'Min_cor'] > mincor[n] and Point_df.loc[i, 'Min_cor'] < mincor[n+1]:
#     # if Point_df.loc[i, 'Min_cor'] > -0.9 and Point_df.loc[i, 'Min_cor'] < -0.89:
#         A_lon = lon[Point_df.loc[i, 'A_lon']]
#         A_lat = lat[Point_df.loc[i, 'A_lat']]
#         B_lon = lon[int(Point_df.loc[i, 'B_lon'])]
#         B_lat = lat[int(Point_df.loc[i, 'B_lat'])]
#         ax1.plot([A_lon,B_lon], [A_lat, B_lat], 'b', '.', transform=ccrs.PlateCarree(), linewidth=0.3)
#         if A_lon > 0 and A_lon < 30 and A_lat > 30 and A_lat < 60:
#             print(A_lat,A_lon,B_lat,B_lon)
#         if B_lon > 0 and B_lon < 30 and B_lat > 30 and B_lat < 60:
#             print(A_lat, A_lon, B_lat, B_lon)
#     # print(mincor[n],mincor[n+1])

# 添加点
# ax1.plot([140, 40], [82.5, 27.5], c='black', transform=ccrs.PlateCarree(), linewidth=0.5)


ax_cf1 = ax1.contourf(cyclic_lons, lat, R, transform=ccrs.PlateCarree(), cmap='hot',
                      levels=np.arange(-1.0, -0.4, 0.025 ), extend='both')  # 绘制等值线图
cb1 = fig.colorbar(ax_cf1, shrink=0.55, orientation='horizontal')  # shrin收缩比例
# plt.savefig(r'F:\6_Scientific_Research\3_Teleconnection\2_Picture\2_Teleconnection\Cir_W_Teleconnectivity matrix_1.png')

plt.show()




