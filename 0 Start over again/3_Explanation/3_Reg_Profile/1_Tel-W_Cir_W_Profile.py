import xarray as xr
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

# -------- 读取数据 --------#
# 读取遥相关指数
TELIS = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\Index\Tel_Index_Normal.csv', usecols=['W_A1A2', 'W_B1B2', 'W_B3B4', 'W_C1C2', 'W_D1D2', 'W_E1E2', 'W_F1F2'])
TEL_name = TELIS.columns
# 读取遥相关两点经纬度
T_Point = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\WTel.csv')
# 读取CIR_H
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_W_197901-202208.nc')
# 读取风速
f2 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\us_W_197901-202208.nc')
f3 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\os_W_197901-202208.nc')

# 筛选并计算夏季平均
lon = f1.lon
lat = np.arange(90, -2.5, -2.5)
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选JJA
Cir_W = f1.data[Sum_Index, :, 0:37, :] # 筛选NH
v = f2.data[Sum_Index, :, 0:37, :]
w = f3.data[Sum_Index, :, 0:37, :] * (-100)
Cir_W_Sum, vs, ws = np.zeros((44, 17, 37, 144)), np.zeros((44, 17, 37, 144)), np.zeros((44, 17, 37, 144))
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_W_Sum[n, :, :, :] = (Cir_W[i, :, :, :] + Cir_W[i + 1, :, :, :] + Cir_W[i + 2, :, :, :]) / 3
        vs[n, :, :, :] = (v[i, :, :, :] + v[i + 1, :, :, :] + v[i + 2, :, :, :]) / 3
        ws[n, :, :, :] = (w[i, :, :, :] + w[i + 1, :, :, :] + w[i + 2, :, :, :]) / 3
        n = n + 1
# Cir_W_Sum[44, 17, 37, 144], [year, height, lat, lon]
# 后续的操作需要Cir_H_Sum是一个DataArray
height, lat, lon = Cir_W.height, Cir_W.lat, Cir_W.lon
year = range(1979, 2023)
Cir_H_Sum = xr.DataArray(Cir_W_Sum, coords=[year, height, lat, lon], dims=['year', 'height', 'lat', 'lon']) # 这里偷懒直接把H-W
vs = xr.DataArray(vs, coords=[year, height, lat, lon], dims=['year', 'height', 'lat', 'lon'])
ws = xr.DataArray(ws, coords=[year, height, lat, lon], dims=['year', 'height', 'lat', 'lon'])


# 计算回归
H, vr, wr, R, P = np.zeros((7, 17, 50)), np.zeros((7, 17, 50)), np.zeros((7, 17, 50)), np.zeros((7, 17, 50)), np.zeros((7, 17, 50))
lat_ver, lon_ver = [], []
for k in range(7):
    # 计算插值部分
    lat_ver.append(np.linspace(T_Point.A_lat[k], T_Point.B_lat[k], 50))
    lon_ver.append(np.linspace(T_Point.A_lon[k], T_Point.B_lon[k], 50))
    H_ver = Cir_H_Sum.interp(lat=lat_ver[k], lon=lon_ver[k]) # [44, 17, 50, 50] 将两点所组成的正方形区域内的格点圈了起来
    H_ver = np.array(H_ver).diagonal(axis1=2,axis2=3) # [44, 17, 50] 在这个正方形区域内求对角线上的数据
    v_ver = vs.interp(lat=lat_ver[k], lon=lon_ver[k]) # [44, 17, 50, 50] 将两点所组成的正方形区域内的格点圈了起来
    v_ver = np.array(v_ver).diagonal(axis1=2,axis2=3) # [44, 17, 50] 在这个正方形区域内求对角线上的数据
    w_ver = ws.interp(lat=lat_ver[k], lon=lon_ver[k]) # [44, 17, 50, 50] 将两点所组成的正方形区域内的格点圈了起来
    w_ver = np.array(w_ver).diagonal(axis1=2,axis2=3) # [44, 17, 50] 在这个正方形区域内求对角线上的数据

    # 计算回归部分
    for i in range(17):
        for j in range(50):
            H[k, i, j], _, R[k, i, j], P[k, i, j], _ = linregress(TELIS.iloc[:, k], H_ver[:, i, j])
            vr[k, i, j], _, _, _, _ = linregress(TELIS.iloc[:, k], v_ver[:, i, j])
            wr[k, i, j], _, _, _, _ = linregress(TELIS.iloc[:, k], w_ver[:, i, j])

#---绘图部分---#
height = f1.height
fig = plt.figure(figsize=(6, 14), dpi=600)
fig.subplots_adjust(hspace=0.6 ) # 子图间距
ax1 = fig.add_subplot(7, 1, 1)
ax2 = fig.add_subplot(7, 1, 2)
ax3 = fig.add_subplot(7, 1, 3)
ax4 = fig.add_subplot(7, 1, 4)
ax5 = fig.add_subplot(7, 1, 5)
ax6 = fig.add_subplot(7, 1, 6)
ax7 = fig.add_subplot(7, 1, 7)
axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
# 循环绘图函数
def drawprofile0(H, name, xmin, xmax, lev):
    fig.suptitle('', fontsize=16, x=0.53, y=1.05)
    for i in range(7):
        axs[i].set_title('(a) lev-lon  REG_Tel-' + name[i] + '-Cir_W', loc='left', fontsize=6)
        # 纵坐标设置
        axs[i].set_yscale('symlog')
        axs[i].set_yticks([1000, 500, 300, 200])
        axs[i].set_yticklabels(['1000', '500', '300', '200'], fontdict={'fontsize':6} )
        axs[i].invert_yaxis()
        axs[i].set_ylabel('Level (hPa)', fontsize=6)
        # 横坐标设置
        axs[i].set_xlim(xmin[i], xmax[i])
        axs[i].set_xlabel('Longitude', fontsize=6)
        w = axs[i].get_xticks() # 获取横坐标值列表，主要为了调节大小
        axs[i].set_xticklabels(w, fontdict={'fontsize': 6})
        ax_cf1 = axs[i].contourf(lon_ver[i],  lev[0:10],  H[i, 0:10, :], levels=np.arange(-0.08, 0.09, 0.01), extend = 'both', zorder=0, cmap='coolwarm')
        axs[i].contourf(lon_ver[i], lev[0:10], P[i, 0:10, :], levels=[0, 0.05, 1], hatches=['...', None], zorder=1,
                        colors="None")
        axs[i].quiver(lon_ver[i], lev[0:10], vr[i, 0:10, :], wr[i, 0:10, :])
        axs[i].xaxis.set_major_formatter(cticker.LongitudeFormatter())
# 设置colorbar
ax_cf1 = ax4.contourf(lon_ver[4],  height[0:10],  H[4, 0:10, :], levels=np.arange(-0.08, 0.09, 0.01), extend='both', zorder=0, cmap='coolwarm')
fig.subplots_adjust(bottom=0.13)
plt.rcParams['font.size'] = 6
position = fig.add_axes([0.15, 0.05, 0.7, 0.015])  # 位置[左,下,右,上]
cb = fig.colorbar(ax_cf1, shrink=0.6, cax=position, orientation='horizontal', extend='both')
drawprofile0(H, TEL_name, T_Point.A_lon, T_Point.B_lon, height)
fig.savefig(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\3_Explanation\3_Profile\reg_Tel-W_Cir-W_UW-W_profile.png')
plt.show()