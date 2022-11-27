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
# ----------SAT
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA5_T_U_V_1.nc')
SAT = f1.t.isel(expver=0, level=1).groupby('time.year').mean(dim='time', skipna=True)
lat = SAT.latitude
lon = SAT.longitude
SAT_CL = SAT.loc[1979:2000].mean('year') # 气候态平均
# --- Calculating - SAT_ANO --- #
SAT_ANO = np.zeros((44, 91, 360))
SAT_detrend = scipy.signal.detrend(SAT, axis=0, type='linear', overwrite_data=False)
SAT_year_mean = SAT.mean('year')
for iyear in range(44):
    SAT_detrend[iyear, :, :] = SAT_detrend[iyear, :, :] + SAT_year_mean[:, :]
    SAT_ANO[iyear, :, :] = SAT_detrend[iyear, :, :] - SAT_CL[:, :]
SAT_Recon_21,SAT_Recon_22 = np.zeros((91, 360)), np.zeros((91, 360))
# --- 提取线性分量 --- #
SAT_linear = SAT - SAT_detrend
SAT_linear_2021 = SAT_linear[-2, :, :]
SAT_linear_2022 = SAT_linear[-1, :, :]

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
for ilat in range(91):
    for ilon in range(360):
        X_0= np.array(TELI)
        Y_0 = np.array(SAT_ANO[:, ilat, ilon]) # 某一格点上的温度异常序列
        X = X_0[:, :]
        Y = Y_0[:]
        # tscv = TimeSeriesSplit(n_splits=3, max_train_size=40, test_size=10)
        # for train_index, test_index in tscv.split(X):
        #     print("TRAIN:", train_index, "TEST:", test_index)
        linreg = LinearRegression()
        lin_model = linreg.fit(X, Y)
        SAT_Recon_21[ilat, ilon] = linreg.predict(X_0[42,:].reshape(1, -1))
        SAT_Recon_22[ilat, ilon] = linreg.predict(X_0[43,:].reshape(1, -1))


# --- Drawing - Contourf --- #
# -- Data - Pre-Process -- #
SAT_Recon_21 = SAT_Recon_21 + SAT_linear_2021
SAT_Recon_22 = SAT_Recon_22 + SAT_linear_2022
SAT_Recon_21_mask = mask_land(SAT_Recon_21, label='ocean', lonname='longitude')
SAT_Recon_22_mask = mask_land(SAT_Recon_22, label='ocean', lonname='longitude')
SAT_Recon_21c, cyclic_lons = add_cyclic_point(SAT_Recon_21_mask, coord=lon)
SAT_Recon_22c, cyclic_lons = add_cyclic_point(SAT_Recon_22_mask, coord=lon)
# -- Fuction - Drawing -- #
def drawing_SAT(cyclic_data1, cyclic_data2, name1, name2):
    # 设置字体为楷体,显示负号
    mpl.rcParams['font.sans-serif'] = ['KaiTi']
    mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=0))
    # 图名
    ax1.set_title('H+W_Muliti_Reg-'+str(name1)+'-SAT')
    ax2.set_title('H+W_Muliti_Reg-'+str(name2)+'-SAT')
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
    ax_cf1 = ax1.contourf(cyclic_lons, lat, cyclic_data1, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-2.4, 2.8, 0.4), extend='both')  # 绘制等值线图
    ax_cf2 = ax2.contourf(cyclic_lons, lat, cyclic_data2, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-2.4, 2.8, 0.4), extend='both')  # 绘制等值线图
    fig.subplots_adjust(right=0.9)
    position = fig.add_axes([0.93, 0.25, 0.015, 0.5])  # 位置[左,下,右,上]
    cb = fig.colorbar(ax_cf2, shrink=0.6, cax=position, extend='both')
    plt.savefig(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\2_Reg_Recon-Pred\Muliti-Reg-Reconstruct-SAT-2021-2022.png')
    plt.show()
drawing_SAT(SAT_Recon_21c, SAT_Recon_22c, name1='Reconstruct-21', name2='Reconstruct-22')








