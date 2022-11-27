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

# --------计算回归 --------- #
Hr, P, vr, wr = np.zeros((7, 17, 37, 144)), np.zeros((7, 17, 37, 144)), np.zeros((7, 17, 37, 144)), np.zeros((7, 17, 37, 144))
for k in range(7):
    print(k)
    TelI = TELIS.iloc[:, k]
    for h in range(17):
        for i in range(37):
            for j in range(144):
                Hr[k, h, i, j], _, _, P[k, h, i, j], _ = linregress(TelI, Cir_H_Sum[:, h, i, j])
                vr[k, h, i, j], _, _, _, _ = linregress(TelI, vs[:, h, i, j])
                wr[k, h, i, j], _, _, _, _ = linregress(TelI, ws[:, h, i, j])
  # ------ 切片出画图所需要的剖面 ------ #
itel = range(7)
P = xr.DataArray(P, coords=[itel, height, lat, lon], dims=['itel', 'height', 'lat', 'lon'])
Hr = xr.DataArray(Hr, coords=[itel, height, lat, lon], dims=['itel', 'height', 'lat', 'lon'])
vr = xr.DataArray(vr, coords=[itel, height, lat, lon], dims=['itel', 'height', 'lat', 'lon'])
wr = xr.DataArray(wr, coords=[itel, height, lat, lon], dims=['itel', 'height', 'lat', 'lon'])
A_H_Prof, A_vr_Prof, A_wr_Prof, A_P_Prof, A_lat_list= [], [], [], [], []
B_H_Prof, B_vr_Prof, B_wr_Prof, B_P_Prof, B_lat_list= [], [], [], [], []
for k in range(7) :
    print(k)
    A_lat = T_Point.A_lat[k]
    A_lon = T_Point.A_lon[k]
    B_lat = T_Point.B_lat[k]
    B_lon = T_Point.B_lon[k]
    if k == 0 :
        # Point_A
        A_lat_range = lat[(lat >= (A_lat-5)) & (lat <= 90)]
        A1_P_Prof = P.sel(itel = k, lat = A_lat_range, lon = A_lon)
        A1_H_prof = Hr.sel(itel = k, lat = A_lat_range, lon = A_lon)
        A1_vr_prof = vr.sel(itel = k, lat = A_lat_range, lon = A_lon)
        A1_wr_prof = wr.sel(itel = k, lat = A_lat_range, lon = A_lon)
        A_P_Prof.append(A1_P_Prof)
        A_H_Prof.append(A1_H_prof)
        A_vr_Prof.append(A1_vr_prof)
        A_wr_Prof.append(A1_wr_prof)
        A_lat_list.append(A_lat_range)
        # Point_B
        B_lat_range = lat[(lat >= (A_lat-5)) & (lat <= 90)]
        B1_P_Prof = P.sel(itel = k, lat = A_lat_range, lon = A_lon)
        B1_H_prof = Hr.sel(itel = k, lat = B_lat_range, lon = B_lon)
        B1_vr_prof = vr.sel(itel = k, lat = B_lat_range, lon = A_lon)
        B1_wr_prof = wr.sel(itel = k, lat = B_lat_range, lon = A_lon)
        B_P_Prof.append(B1_P_Prof)
        B_vr_Prof.append(B1_vr_prof)
        B_wr_Prof.append(B1_wr_prof)
        B_H_Prof.append(B1_H_prof)
        B_lat_list.append(B_lat_range)
    else:
        A_lat_range = lat[(lat >= (A_lat - 5)) & (lat <= (A_lat + 5))]
        B_lat_range = lat[(lat >= (B_lat - 5)) & (lat <= (B_lat + 5))]
        Ai_H_prof = Hr.sel(itel = k, lat = A_lat_range, lon = A_lon)
        Bi_H_prof = Hr.sel(itel = k, lat = B_lat_range, lon = B_lon)
        Ai_P_prof = P.sel(itel = k, lat = A_lat_range, lon = A_lon)
        Bi_P_prof = P.sel(itel = k, lat = B_lat_range, lon = B_lon)
        Ai_vr_prof = vr.sel(itel = k, lat = A_lat_range, lon = A_lon)
        Bi_vr_prof = vr.sel(itel = k, lat = B_lat_range, lon = B_lon)
        Ai_wr_prof = wr.sel(itel = k, lat = A_lat_range, lon = A_lon)
        Bi_wr_prof = wr.sel(itel = k, lat = B_lat_range, lon = B_lon)
        A_H_Prof.append(Ai_H_prof)
        B_H_Prof.append(Bi_H_prof)
        A_P_Prof.append(Ai_P_prof)
        B_P_Prof.append(Bi_P_prof)
        A_vr_Prof.append(Ai_vr_prof)
        B_vr_Prof.append(Bi_vr_prof)
        A_wr_Prof.append(Ai_wr_prof)
        B_wr_Prof.append(Bi_wr_prof)
        A_lat_list.append(A_lat_range)
        B_lat_list.append(B_lat_range)



# ------- draw Picture ------ #
fig = plt.figure(figsize=(6, 14), dpi=600)
fig.subplots_adjust(hspace=0.6, wspace=0.3) # 子图间距
ax1 = fig.add_subplot(7, 2, 1)
ax2 = fig.add_subplot(7, 2, 2)
ax3 = fig.add_subplot(7, 2, 3)
ax4 = fig.add_subplot(7, 2, 4)
ax5 = fig.add_subplot(7, 2, 5)
ax6 = fig.add_subplot(7, 2, 6)
ax7 = fig.add_subplot(7, 2, 7)
ax8 = fig.add_subplot(7, 2, 8)
ax9 = fig.add_subplot(7, 2, 9)
ax10 = fig.add_subplot(7, 2, 10)
ax11 = fig.add_subplot(7, 2, 11)
ax12 = fig.add_subplot(7, 2, 12)
ax13 = fig.add_subplot(7, 2, 13)
ax14 = fig.add_subplot(7, 2, 14)
axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14]

def draw_profile():
    n, m = 0, 0
    for i in range(14):
        # 纵坐标设置
        axs[i].set_yscale('symlog')
        axs[i].set_yticks([1000, 500, 300, 200])
        axs[i].set_yticklabels(['1000', '500', '300', '200'], fontdict={'fontsize': 6})
        axs[i].invert_yaxis()
        axs[i].set_ylabel('Level (hPa)', fontsize=6)
        # 横坐标设置
        # axs[i].set_xlim(A_lat_list[i].min(), A_lat_list[i].max())
        # axs[i].set_xlabel('Latitude', fontsize=6)
        # w = axs[i].get_xticks()  # 获取横坐标值列表，主要为了调节大小
        # axs[i].set_xticklabels(w, fontdict={'fontsize': 6})
        if i%2 == 0:
            axs[i].set_xlim(A_lat_list[n].min(), A_lat_list[n].max())
            axs[i].set_xlabel('Latitude', fontsize=6)
            w = axs[i].get_xticks()  # 获取横坐标值列表，主要为了调节大小
            axs[i].set_xticklabels(w, fontdict={'fontsize': 6})

            drawH, drawv, draww, drawP= A_H_Prof[n], A_vr_Prof[n], A_wr_Prof[n], A_P_Prof[n]
            ax_cf1 = axs[i].contourf(A_lat_list[n],  height[0:10],  drawH[0:10, :], levels=np.arange(-0.04, 0.044, 0.004), zorder=0, extend = 'both', cmap='coolwarm')
            axs[i].contourf(A_lat_list[n],  height[0:10],  drawP[0:10, :], levels=[0, 0.05, 1], hatches=['...', None], zorder=1, colors="None")
            axs[i].quiver(A_lat_list[n],  height[0:10], drawv[0:10, :], draww[0:10, :])
            n = n+1
        else:
            axs[i].set_xlim(B_lat_list[m].min(), B_lat_list[m].max())
            axs[i].set_xlabel('Latitude', fontsize=6)
            w = axs[i].get_xticks()  # 获取横坐标值列表，主要为了调节大小
            axs[i].set_xticklabels(w, fontdict={'fontsize': 6})

            drawH, drawv, draww, drawP= B_H_Prof[m], B_vr_Prof[m], B_wr_Prof[m], B_P_Prof[m]
            ax_cf1 = axs[i].contourf(B_lat_list[m],  height[0:10],  drawH[0:10, :], levels=np.arange(-0.04, 0.044, 0.004), zorder=0, extend = 'both', cmap='coolwarm')
            axs[i].contourf(B_lat_list[m],  height[0:10],  drawP[0:10, :], levels=[0, 0.05, 1], hatches=['...', None], zorder=1, colors="None")
            axs[i].quiver(B_lat_list[m],  height[0:10], drawv[0:10, :], draww[0:10, :])
            m = m+1
    plt.show()

draw_profile()
fig.savefig(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\3_Explanation\3_Profile\reg_Tel-H_Cir-H_VW-H_profile_Point.png')
