import xarray as xr
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.util import add_cyclic_point #进行循环
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要


# --- Read data --- #
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\Min_R.nc')
f2 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\lat_min.nc')
f3 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\lon_min.nc')
lat, lon = np.array(f1.lat), np.array(f1.lon)
r = f1.Min_R
lat_min = f2.lat_min
lon_min = f3.lon_min


# --- Tel—point --- #
WTel = pd.DataFrame(columns=['Min_r', 'A_lat', 'A_lon', 'B_lat', 'B_lon'])
A_lat = [87.5, 87.5, 42.5, 65, 70, 25, 80, 70]
A_lon = [347.5, 32.5, 107.5, 77.5, 355, 155, 187.5, 197.5]
Min_r, B_lat, B_lon = [], [], []
for i in range(len(A_lat)):
    longitude = A_lon[i]
    latitude = A_lat[i]
    Min_r.append(float(r.sel(lat=latitude, lon=longitude).values))
    B_lat.append(lat[int(lat_min.sel(lat=latitude, lon=longitude).values)])
    B_lon.append(lon[int(lon_min.sel(lat=latitude, lon=longitude).values)])
WTel.Min_r, WTel.A_lat, WTel.A_lon, WTel.B_lat, WTel.B_lon = Min_r, A_lat, A_lon, B_lat, B_lon
WTel.to_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\WTel.csv', )


# --- Drawing contourf --- #
r, cyclic_lons = add_cyclic_point(r, coord=lon)
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
fig = plt.figure(figsize=(10, 3), dpi=400)
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
# # --- Draw-plot
# for n in range(10):
#          ax1.plot([A_lon[n],B_lon[n]], [A_lat[n], B_lat[n]], 'b', '.', transform=ccrs.PlateCarree(), linewidth=0.5)

# # - Plot-Select-Tel
# rr = f1.Min_R
# lat_mm = f2.lat_min
# lon_mm = f3.lon_min
# Point = pd.DataFrame(columns=['Min_r', 'A_lat', 'A_lon', 'B_lat', 'B_lon'])
# Min_r, A_lat, A_lon, B_lat, B_lon = [], [], [], [], []
# n = 0
# for i in range(27):
#     for j in range(144):
#         if r[i, j] >= -0.71 and r[i, j] <= -0.70:
#             # A_lat, A_lon = lat[i], lon[j]
#             # B_lat, B_lon = lat[int(lat_min[i, j])], lon[int(lon_min[i, j])]
#             # 存入Dataframe
#             Min_r.append(r[i, j])
#             A_lat.append(lat[i])
#             A_lon.append(lon[j])
#             B_lat.append(lat[int(lat_min[i, j])])
#             B_lon.append(lon[int(lon_min[i, j])])
#             # print(i, j, A_lat.values, A_lon.values, B_lat.values, B_lon.values)
#             ax1.plot([A_lon[n],B_lon[n]], [A_lat[n], B_lat[n]], 'b', '.', transform=ccrs.PlateCarree(), linewidth=0.5)
#             n += 1
# Point.Min_r, Point.A_lat, Point.A_lon, Point.B_lat, Point.B_lon = Min_r, A_lat, A_lon, B_lat, B_lon
# Point = Point.sort_values(by='Min_r', ascending=False)

# - ax1
ax1.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
ax1.set_yticks([0, 30, 60, 90])
ax1.xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.8)  # 添加海岸线
ax_cf1 = ax1.contourf(cyclic_lons, lat, r, transform=ccrs.PlateCarree(), cmap='hot', levels=np.arange(-0.75, -0.45, 0.05 ), extend='both')  # 绘制等值线图
cb1 = fig.colorbar(ax_cf1, shrink=0.55, orientation='horizontal')  # shrin收缩比例
plt.show()


