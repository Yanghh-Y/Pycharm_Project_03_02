import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal # 用于去线性趋势
import sys
from sklearn.metrics import mean_squared_error, r2_score # 用于评价模型的函数
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point #进行循环
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要
from Mask_Ocean_Land import mask_land

# --- Read - Data --- #
# ----------Z
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_R_197901-202208.nc')
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_R = f1.data[Sum_Index, 5, 0:37, :] # 筛选
Cir_R_Sum = np.zeros((44, 37, 144))
lat, lon = Cir_R.lat, Cir_R.lon
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_R_Sum[n, :, :] = (Cir_R[i, :, :] + Cir_R[i + 1, :, :] + Cir_R[i + 2, :, :]) / 3
        n = n + 1
year = range(1979, 2023)
Cir_R_Sum = xr.DataArray(Cir_R_Sum, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])
Cir_R_CL = Cir_R_Sum.sel(year=range(1979, 2001)).mean('year')


# --- Calculating - Cir_R_ANO --- #F
Cir_R_ANO = np.zeros((44, 37, 144))
Cir_R_detrend = scipy.signal.detrend(Cir_R_Sum, axis=0, type='linear', overwrite_data=False)
Cir_R_year_mean = Cir_R_Sum.mean('year')
for iyear in range(44):
    Cir_R_detrend[iyear, :, :] = Cir_R_detrend[iyear, :, :] + Cir_R_year_mean[:, :]
    Cir_R_ANO[iyear, :, :] = Cir_R_detrend[iyear, :, :] - Cir_R_CL[:, :]
Cir_R_Recon_21,Cir_R_Recon_22 = np.zeros((37, 144)), np.zeros((37, 144))
# --- 提取线性分量 --- #
Cir_R_linear = Cir_R_Sum - Cir_R_detrend
Cir_R_linear_2021 = Cir_R_linear[-2, :, :]
Cir_R_linear_2022 = Cir_R_linear[-1, :, :]

# ---------- TEL-Index
TELIS_H = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\Tel_H_I.csv',
                      usecols=['A1A2', 'A3A4', 'B1B2', 'C1C2', 'C3C4', 'D1D2', 'E1E2', 'F1F2', 'G1G2', 'H1H2'])
TELIS_W = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\Tel_W_I.csv',
                      usecols=['A1A2', 'A3A4', 'B1B2', 'B3B4', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
TELI = pd.concat([TELIS_H, TELIS_W], axis=1)
col = ['H_A1A2', 'H_A3A4', 'H_B1B2', 'H_C1C2', 'H_C3C4', 'H_D1D2', 'H_E1E2', 'H_F1F2', 'H_G1G2', 'H_H1H2', 'W_A1A2', 'W_A3A4', 'W_B1B2', 'W_B3B4', 'W_C1C2', 'W_D1D2', 'W_E1E2',
       'W_F1F2']
TELI.columns = col
TELI = TELI.drop(columns=['H_A3A4', 'H_C3C4', 'H_G1G2', 'W_A3A4'])



# --- 训练模型 --- #
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression  # 线性回归
for ilat in range(37):
    for ilon in range(144):
        X_0= np.array(TELI)
        Y_0 = np.array(Cir_R_ANO[:, ilat, ilon]) # 某一格点上的温度异常序列
        X = X_0[:, :]
        Y = Y_0[:]
        # tscv = TimeSeriesSplit(n_splits=3, max_train_size=40, test_size=10)
        # for train_index, test_index in tscv.split(X):
        #     print("TRAIN:", train_index, "TEST:", test_index)
        linreg = LinearRegression()
        lin_model = linreg.fit(X, Y)
        Cir_R_Recon_21[ilat, ilon] = linreg.predict(X_0[42,:].reshape(1, -1))
        Cir_R_Recon_22[ilat, ilon] = linreg.predict(X_0[43,:].reshape(1, -1))


# --- Drawing - Contourf --- #
# -- Data - Pre-Process -- #
Cir_R_Recon_21 = Cir_R_Recon_21 + Cir_R_linear_2021
Cir_R_Recon_22 = Cir_R_Recon_22 + Cir_R_linear_2022
Cir_R_Recon_21_mask = mask_land(Cir_R_Recon_21, label='ocean', lonname='lon')
Cir_R_Recon_22_mask = mask_land(Cir_R_Recon_22, label='ocean', lonname='lon')
Cir_R_Recon_21c, cyclic_lons = add_cyclic_point(Cir_R_Recon_21_mask, coord=lon)
Cir_R_Recon_22c, cyclic_lons = add_cyclic_point(Cir_R_Recon_22_mask, coord=lon)
# -- Fuction - Drawing -- #
def drawing_Z(cyclic_data1, cyclic_data2, name1, name2):
    # 设置字体为楷体,显示负号
    mpl.rcParams['font.sans-serif'] = ['KaiTi']
    mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=0))
    # 图名
    ax1.set_title('H+W_Muliti_Reg-'+str(name1)+'-Cir_R')
    ax2.set_title('H+W_Muliti_Reg-'+str(name2)+'-Cir_R')
    # 刻度线形式
    ax1.set_xticks([-180, -150, -120, -90, -60, -30,0, 30, 60, 90, 120, 150, 180]) #指定要显示的经纬度
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
    ax1.add_feature(cfeature.LAKES, edgecolor='black')  # 边缘为黑色
    ax2.add_feature(cfeature.LAKES, edgecolor='black')  # 边缘为黑色
    ax_cf1 = ax1.contourf(cyclic_lons, lat, cyclic_data1, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-0.05, 0.055, 0.005), extend='both')  # 绘制等值线图
    ax_cf2 = ax2.contourf(cyclic_lons, lat, cyclic_data2, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-0.05, 0.055, 0.005), extend='both')  # 绘制等值线图
    fig.subplots_adjust(right=0.9)
    position = fig.add_axes([0.93, 0.25, 0.015, 0.5])  # 位置[左,下,右,上]
    cb = fig.colorbar(ax_cf2, shrink=0.6, cax=position, extend='both')
    plt.savefig(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\2_Reg_Recon-Pred\Muliti-Reg-Reconstruct-Cir_R-2021-2022.png')
    plt.show()
drawing_Z(Cir_R_Recon_21c, Cir_R_Recon_22c, name1='Reconstruct-21', name2='Reconstruct-22')








