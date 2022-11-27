import xarray as xr
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

# -------- 读取数据 -------- #
# 读取遥相关指数
TELIS = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\Index\Tel_Index_Normal.csv', usecols=['H_A1A2', 'H_B1B2', 'H_C1C2', 'H_D1D2', 'H_E1E2', 'H_F1F2', 'H_H1H2'])
TEL_name = TELIS.columns
# 读取遥相关两点经纬度
T_Point = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\HTel.csv')
# 读取CIR_H
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_H_197901-202208.nc')
# 读取风速
f2 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\vs_H_197901-202208.nc')
f3 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\os_H_197901-202208.nc')

# 筛选并计算夏季平均
lon = f1.lon
lat = np.arange(90, -2.5, -2.5)
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选JJA
Cir_H = f1.data[Sum_Index, :, 0:37, :] # 筛选NH
v = f2.data[Sum_Index, :, 0:37, :]
w = f3.data[Sum_Index, :, 0:37, :] * (-300)
Cir_H_Sum, vs, ws = np.zeros((44, 17, 37, 144)), np.zeros((44, 17, 37, 144)), np.zeros((44, 17, 37, 144))
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_H_Sum[n, :, :, :] = (Cir_H[i, :, :, :] + Cir_H[i + 1, :, :, :] + Cir_H[i + 2, :, :, :]) / 3
        vs[n, :, :, :] = (v[i, :, :, :] + v[i + 1, :, :, :] + v[i + 2, :, :, :]) / 3
        ws[n, :, :, :] = (w[i, :, :, :] + w[i + 1, :, :, :] + w[i + 2, :, :, :]) / 3
        n = n + 1
# Cir_H_Sum[44, 17, 37, 144], [year, height, lat, lon]
# 后续的操作需要Cir_H_Sum是一个DataArray
height, lat, lon = Cir_H.height, Cir_H.lat, Cir_H.lon
year = range(1979, 2023)
Cir_H_Sum = xr.DataArray(Cir_H_Sum, coords=[year, height, lat, lon], dims=['year', 'height', 'lat', 'lon'])
vs = xr.DataArray(vs, coords=[year, height, lat, lon], dims=['year', 'height', 'lat', 'lon'])
ws = xr.DataArray(ws, coords=[year, height, lat, lon], dims=['year', 'height', 'lat', 'lon'])


# -------- 进行插值操作+回归 -------- #
H, vr, wr, R, P = np.zeros((7, 17, 50)), np.zeros((7, 17, 50)), np.zeros((7, 17, 50)), np.zeros((7, 17, 50)), np.zeros((7, 17, 50))
lat_ver, lon_ver = [], []

# T_Point第一个遥相关是同一纬度的，只能从第二个遥相关开始
for k in range(6):
    # 计算插值部分
    lat_ver.append(np.linspace(T_Point.A_lat[1+k], T_Point.B_lat[1+k], 50))
    lon_ver.append(np.linspace(T_Point.A_lon[1+k], T_Point.B_lon[1+k], 50))
    H_ver = Cir_H_Sum.interp(lat=lat_ver[k], lon=lon_ver[k]) # [44, 17, 50, 50] 将两点所组成的正方形区域内的格点圈了起来
    H_ver = np.array(H_ver).diagonal(axis1=2,axis2=3) # [44, 17, 50] 在这个正方形区域内求对角线上的数据
    v_ver = vs.interp(lat=lat_ver[k], lon=lon_ver[k]) # [44, 17, 50, 50] 将两点所组成的正方形区域内的格点圈了起来
    v_ver = np.array(v_ver).diagonal(axis1=2,axis2=3) # [44, 17, 50] 在这个正方形区域内求对角线上的数据
    w_ver = ws.interp(lat=lat_ver[k], lon=lon_ver[k]) # [44, 17, 50, 50] 将两点所组成的正方形区域内的格点圈了起来
    w_ver = np.array(w_ver).diagonal(axis1=2,axis2=3) # [44, 17, 50] 在这个正方形区域内求对角线上的数据

    # 计算回归部分
    for i in range(17):
        for j in range(50):
            H[1+k, i, j], _, R[1+k, i, j], P[1+k, i, j], _ = linregress(TELIS.iloc[:, 1+k], H_ver[:, i, j])
            vr[1+k, i, j], _, _, _, _ = linregress(TELIS.iloc[:, 1+k], v_ver[:, i, j])
            wr[1+k, i, j], _, _, _, _ = linregress(TELIS.iloc[:, 1+k], w_ver[:, i, j])

# T_Point第一个遥相关是同一纬度，单独计算回归，lat[1] = 87.5 两点坐标：(87.5, 87.5), (87.5, 265)
s, r, p, v1, w1 = np.zeros((17, 144)), np.zeros((17, 144)), np.zeros((17, 144)), np.zeros((17, 144)), np.zeros((17, 144))
for i in range(17):
    for j in range(144):
        s[i, j], _, r[i, j], p[i, j], _ = linregress(TELIS.iloc[:, 0], Cir_H_Sum[:, i, 1, j]) # lat[1] 87.5
        v1[i, j], _, _, _, _ = linregress(TELIS.iloc[:, 0], vs[:, i, 1, j]) # lat[1] 87.5
        w1[i, j], _, _, _, _ = linregress(TELIS.iloc[:, 0], ws[:, i, 1, j]) # lat[1] 87.5
# vr = vr * 1000
# windspeed = np.sqrt(vr**2 + wr**2)
# windspeed1 = np.sqrt(v1**2 + w1**2)
# s 是 A1A2, 同纬度上
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
    fig.suptitle('', fontsize=16, x=0.53, y=1.05, )
    for i in range(7):
        if i != 0:
            axs[i].set_title('(a) lev-lat  REG_Tel-' + name[i] + '-Cir_H', loc='left', fontsize=6)
            # 纵坐标设置
            axs[i].set_yscale('symlog')
            axs[i].set_yticks([1000, 500, 300, 200])
            axs[i].set_yticklabels(['1000', '500', '300', '200'], fontdict={'fontsize':6} )
            axs[i].invert_yaxis()
            axs[i].set_ylabel('Level (hPa)', fontsize=6)
            # 横坐标设置
            axs[i].set_xlim(xmin[i], xmax[i])
            axs[i].set_xlabel('Latitude', fontsize=6)
            w = axs[i].get_xticks() # 获取横坐标值列表，主要为了调节大小
            axs[i].set_xticklabels(w, fontdict={'fontsize': 6})
            ax_cf1 = axs[i].contourf(lat_ver[i-1],  lev[0:10],  H[i, 0:10, :], levels=np.arange(-0.04, 0.044, 0.004), zorder=0, extend = 'both', cmap='coolwarm')
            axs[i].contourf(lat_ver[i-1],  lev[0:10],  P[i, 0:10, :], levels=[0, 0.05, 1], hatches=['...', None], zorder=1, colors="None")
            axs[i].quiver(lat_ver[i-1],  lev[0:10], vr[i, 0:10, :], wr[i, 0:10, :])
            axs[i].xaxis.set_major_formatter(cticker.LatitudeFormatter())
        else:
            axs[i].set_title('(a) lev-lon  REG_Tel_' + name[i] + '-Cir_H', loc='left', fontsize=6)
            # 纵坐标设置
            axs[i].set_yscale('symlog')
            axs[i].set_yticks([1000, 500, 300, 200])
            axs[i].set_yticklabels(['1000', '500', '300', '200'], fontdict={'fontsize':6})
            axs[i].invert_yaxis()
            axs[i].set_ylabel('Level (hPa)', fontsize=6)
            # 横坐标设置
            axs[i].set_xlim(T_Point.A_lon[0], T_Point.B_lon[0])
            axs[i].set_xlabel('Longitude', fontsize=6)
            w = axs[i].get_xticks()  # 获取横坐标值列表，主要为了调节大小
            axs[i].set_xticklabels(w, fontdict={'fontsize': 6})
            ax_cf1 = axs[i].contourf(lon[35: 107], lev[0:10], s[0:10, 35:107], levels=np.arange(-0.04, 0.044, 0.004), extend='both', zorder=0,  cmap='coolwarm')
            axs[i].quiver(lon[35: 107], lev[0:10], v1[0:10, 35:107], w1[0:10, 35:107])
            axs[i].contourf(lon[35: 107], lev[0:10], p[0:10, 35:107], levels=[0, 0.05, 1], hatches=['...', None], zorder=1, colors="None")
            axs[i].xaxis.set_major_formatter(cticker.LongitudeFormatter())
# 设置colorbar
drawprofile0(H, TEL_name, T_Point.A_lat, T_Point.B_lat, height)
ax_cf1 = ax4.contourf(lat_ver[4],  height[0:10],  H[4, 0:10, :], levels=np.arange(-0.04, 0.044, 0.004), extend = 'both', zorder=0, cmap='coolwarm')
fig.subplots_adjust(bottom=0.13)
plt.rcParams['font.size'] = 6
position = fig.add_axes([0.15, 0.05, 0.7, 0.015])  # 位置[左,下,右,上]
cb = fig.colorbar(ax_cf1, shrink=0.6, cax=position, orientation='horizontal', extend='both')
fig.savefig(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\3_Explanation\3_Profile\reg_Tel-H_Cir-H_VW-H_profile.png')
# drawprofile0(H, TEL_name, T_Point.A_lat, T_Point.B_lat, height)
plt.show()