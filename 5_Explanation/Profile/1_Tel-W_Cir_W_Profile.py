import xarray as xr
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

#--读取数据--#
# 读取遥相关指数
TELIS = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\WI.csv', usecols=['A1A2', 'A1A3', 'B1B2', 'B1B3', 'C1C2', 'C1C3', 'E1E2', 'E1E3'])
# 读取遥相关两点经纬度
T_Point = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\W_Tel_Manual.csv')
# 读取CIR_H
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_W_197901-202208.nc')
# 读取
# 筛选并计算夏季平均
lon = f1.lon
lat = np.arange(90, -2.5, -2.5)
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选JJA
Cir_W = f1.data[Sum_Index, :, 0:37, :] # 筛选NH
Cir_W_Sum = np.zeros((44, 17, 37, 144))
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_W_Sum[n, :, :, :] = (Cir_W[i, :, :, :] + Cir_W[i + 1, :, :, :] + Cir_W[i + 2, :, :, :]) / 3
        n = n + 1
# Cir_H_Sum[44, 17, 37, 144], [year, height, lat, lon]
# 后续的操作需要Cir_H_Sum是一个DataArray
height, lat, lon = Cir_W.height, Cir_W.lat, Cir_W.lon
year = range(1979, 2023)
Cir_H_Sum = xr.DataArray(Cir_W_Sum, coords=[year, height, lat, lon], dims=['year', 'height', 'lat', 'lon'])


#---进行插值操作+回归---#
# 遥相关指数标准化
def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    return x
TELIS['A1A2_std'], TELIS['A1A3_std'], TELIS['B1B2_std'], TELIS['B1B3_std'], TELIS['C1C2_std'], TELIS['C1C3_std'], TELIS['E1E2_std'], TELIS['E1E3_std']\
    = ZscoreNormalization(TELIS.A1A2), ZscoreNormalization(TELIS.A1A3), ZscoreNormalization(TELIS.B1B2), ZscoreNormalization(TELIS.B1B3), ZscoreNormalization(TELIS.C1C2), ZscoreNormalization(TELIS.C1C3), ZscoreNormalization(TELIS.E1E2), ZscoreNormalization(TELIS.E1E3)
# 计算回归
H, r, p = np.zeros((8, 17, 50)), np.zeros((8, 17, 50)), np.zeros((8, 17, 50))
lat_ver, lon_ver = [], []
for k in range(8):
    # 计算插值部分
    lat_ver.append(np.linspace(T_Point.Point1_lat[k], T_Point.Point2_lat[k], 50))
    lon_ver.append(np.linspace(T_Point.Point1_lon[k], T_Point.Point2_lon[k], 50))
    H_ver = Cir_H_Sum.interp(lat=lat_ver[k], lon=lon_ver[k]) # [44, 17, 50, 50] 将两点所组成的正方形区域内的格点圈了起来
    H_ver = np.array(H_ver).diagonal(axis1=2,axis2=3) # [44, 17, 50] 在这个正方形区域内求对角线上的数据
    # 计算回归部分
    for i in range(17):
        for j in range(50):
            H[k, i, j], _, r[k, i, j], p[k, i, j], _ = linregress(TELIS.iloc[:, 8+k], H_ver[:, i, j])
# # 计算回归
# s,r,p = np.zeros((17, 144)),np.zeros((17, 144)),np.zeros((17, 144))
# k = 1
# for i in range(17):
#     for j in range(144):
#         s[i, j], _, r[i, j], p[i, j], _ = linregress(TELIS.iloc[:, 6], Cir_H_Sum[:, i, 6, j])
#         # TEL_index 中的7个遥相关指数作为自变量，每一个点上的SAT序列作为因变量
# s 是 C1C2, 同纬度上
#---绘图部分---#
height = f1.height
fig = plt.figure(figsize=(4, 12), dpi=600)
fig.subplots_adjust(hspace=0.6 ) # 子图间距
ax1 = fig.add_subplot(8, 1, 1)
ax2 = fig.add_subplot(8, 1, 2)
ax3 = fig.add_subplot(8, 1, 3)
ax4 = fig.add_subplot(8, 1, 4)
ax5 = fig.add_subplot(8, 1, 5)
ax6 = fig.add_subplot(8, 1, 6)
ax7 = fig.add_subplot(8, 1, 7)
ax8 = fig.add_subplot(8, 1, 8)
axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
# 循环绘图函数
def drawprofile0(H, name, xmin, xmax, lev):
    fig.suptitle('', fontsize=16, x=0.53, y=1.05)
    for i in range(8):
        axs[i].set_title('(a) lev-lon  REG_Tel-W-' + name[i] + '-Cir_W', loc='left', fontsize=6)
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
        ax_cf1 = axs[i].contourf(lon_ver[i],  lev[0:10],  H[i, 0:10, :], levels=np.arange(-0.12, 0.13, 0.01), extend = 'both', zorder=0, cmap='coolwarm')
        axs[i].xaxis.set_major_formatter(cticker.LongitudeFormatter())
# 设置colorbar
ax_cf1 = ax4.contourf(lon_ver[4],  height[0:10],  H[4, 0:10, :], levels=np.arange(-0.12, 0.13, 0.01), extend='both', zorder=0, cmap='coolwarm')
fig.subplots_adjust(bottom=0.13)
plt.rcParams['font.size'] = 6
position = fig.add_axes([0.15, 0.05, 0.7, 0.015])  # 位置[左,下,右,上]
cb = fig.colorbar(ax_cf1, shrink=0.6, cax=position, orientation='horizontal', extend='both')
fig.savefig(r'F:\6_Scientific_Research\3_Teleconnection\2_Picture\4_Explanation\2_Profile\reg_Tel-W_Cir-W_profile.png')
drawprofile0(H, T_Point.Name, T_Point.Point1_lon, T_Point.Point2_lon, height)
plt.show()