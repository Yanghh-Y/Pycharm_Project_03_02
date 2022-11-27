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
Point_df = pd.DataFrame({
    "Name":['A1A2', 'B1B2', 'C1C2', 'D1D2', 'E1E2', 'F1F2', 'G1G2', 'H1H2', 'I1I2'],
    "Cor":[0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Point1_lat":[27.5, 27.5, 72.5, 90, 70, 30, 65, 72.5, 72.5],
    "Point1_lon":[167.5, 32.5, 107.5, 165, 340, 185, 205, 320, 22.5],
    "Point2_lat":[27.5, 27.5, 47.5, 90, 45, 45, 45, 70, 50],
    "Point2_lon":[347.5, 212.5, 100, 265, 355, 257.5, 197.5, 132.5, 17.5]
})


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
for i in range(len(Point_df.index)):
    A_lon = Point_df.loc[i, 'Point1_lon']
    A_lat = Point_df.loc[i, 'Point1_lat']
    B_lon = Point_df.loc[i, 'Point2_lon']
    B_lat = Point_df.loc[i, 'Point2_lat']
    ax1.plot([A_lon,B_lon], [A_lat, B_lat], 'b', '.', transform=ccrs.PlateCarree(), linewidth=0.3)
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




