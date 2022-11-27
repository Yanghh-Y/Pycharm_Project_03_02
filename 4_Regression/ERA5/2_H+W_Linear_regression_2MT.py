import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal # 用于去线性趋势
import matplotlib as mpl
import matplotlib.pyplot as plt
from global_land_mask import globe
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point #进行循环
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要

#---MASK OCEAN Function---#
def mask_land(ds, label='land', lonname='lon'):
    if lonname == 'lon':
        lat = ds.lat.data
        lon = ds.lon.data
        if np.any(lon > 180):
            lon = lon - 180
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
            temp = []
            temp = mask[:, 0:(len(lon) // 2)].copy()
            mask[:, 0:(len(lon) // 2)] = mask[:, (len(lon) // 2):]
            mask[:, (len(lon) // 2):] = temp
        else:
            lons, lats = np.meshgrid(lon, lat)# Make a grid
            mask = globe.is_ocean(lats, lons)# Get whether the points are on ocean.
        ds.coords['mask'] = (('lat', 'lon'), mask)
    elif lonname == 'longitude':
        lat = ds.latitude.data
        lon = ds.longitude.data
        if np.any(lon > 180):
            lon = lon - 180
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
            temp = []
            temp = mask[:, 0:(len(lon) // 2)].copy()
            mask[:, 0:(len(lon) // 2)] = mask[:, (len(lon) // 2):]
            mask[:, (len(lon) // 2):] = temp
        else:
            lons, lats = np.meshgrid(lon, lat)
            mask = globe.is_ocean(lats, lons)
        lons, lats = np.meshgrid(lon, lat)
        mask = globe.is_ocean(lats, lons)
        ds.coords['mask'] = (('latitude', 'longitude'), mask)
    if label == 'land':
        ds = ds.where(ds.mask == True)
    elif label == 'ocean':
        ds = ds.where(ds.mask == False)
    return ds


#---读取2m气温进行回归---#
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA5_2MT_1959-2022_JJA_25.nc')
SAT = f1.t2m.isel(expver=0).groupby('time.year').mean(dim='time', skipna=True)
lat = SAT.latitude
lon = SAT.longitude
SAT = SAT[20:, :, :] # 切片出1979-20222
# SAT气候态平均
SAT_CL = SAT.loc[1971:2000].mean('year')
# SAT去线性趋势
SAT_ANO = np.zeros((44, 91, 360))
SAT_detrend = scipy.signal.detrend(SAT, axis=0, type='linear', overwrite_data=False)
SAT_year_mean = SAT.mean('year')
for iyear in range(44):
    SAT_detrend[iyear, :, :] = SAT_detrend[iyear, :, :] + SAT_year_mean[:, :]
    SAT_ANO[iyear, :, :] = SAT_detrend[iyear, :, :] - SAT_CL[:, :]
# 将17,18年的线性分量提取出来
SAT_linear = SAT - SAT_detrend
SAT_linear_2021 = SAT_linear[-2, :, :]
SAT_linear_2022 = SAT_linear[-1, :, :]
# 读取遥相关指数
TELIS_H = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\WI.csv',
                      usecols=['A1A2', 'A1A3', 'B1B2', 'B1B3', 'C1C2', 'C1C3', 'E1E2', 'E1E3'])
TELIS_W = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\HI.csv',
                      usecols=['B1B2', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
TELIS = pd.concat([TELIS_H, TELIS_W], axis=1)
col = ['W_A1A2', 'W_A1A3', 'W_B1B2', 'W_B1B3', 'W_C1C2', 'W_C1C3', 'W_E1E2', 'W_E1E3', 'B1B2', 'C1C2', 'D1D2', 'E1E2',
       'F1F2']
TELIS.columns = col

SAT_recon_21, SAT_recon_22 = np.zeros((91, 360)), np.zeros((91, 360))
SAT_pred_21, SAT_pred_22 = np.zeros((91, 360)), np.zeros((91, 360))

# 训练模型阶段
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression  #线性回归
from sklearn.metrics import mean_squared_error, r2_score
for ilat in range(91):
    for ilon in range(360):
        # 划分了数据集为训练集和测试集
        # # Reconstracted
        # X_0= np.array(TELIS)
        # Y_0 = np.array(SAT_ANO[:, ilat, ilon]) # 某一格点上的温度异常序列
        #
        # X = X_0[:, :]
        # Y = Y_0[:]
        # tscv = TimeSeriesSplit(n_splits=3, max_train_size=40, test_size=10)
        # for train_index, test_index in tscv.split(X):
        #     print("TRAIN:", train_index, "TEST:", test_index)
        # linreg = LinearRegression()
        # lin_model = linreg.fit(X, Y)
        #
        # SAT_recon_21[ilat, ilon] = linreg.predict(X_0[42,:].reshape(1, -1))
        # SAT_recon_22[ilat, ilon] = linreg.predict(X_0[43,:].reshape(1, -1))


        # Predict
        X_0= np.array(TELIS)
        Y_0 = np.array(SAT_ANO[:, ilat, ilon]) # 某一格点上的温度异常序列
        X = X_0[0:41, :]
        Y = Y_0[0:41]
        X_Test = X_0[42:44, :]
        Y_Test = Y_0[42:44]
        tscv = TimeSeriesSplit(n_splits=3, max_train_size=40, test_size=10)
        for train_index, test_index in tscv.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
        linreg = LinearRegression()
        lin_model = linreg.fit(X, Y)
        y_pred = lin_model.predict(X_Test)
        SAT_pred_21[ilat, ilon] = y_pred[0]
        SAT_pred_22[ilat, ilon] = y_pred[1]

lat = f1.latitude
lon = f1.longitude
SAT_recon_21 = SAT_recon_21 + SAT_linear_2021
SAT_recon_22 = SAT_recon_22 + SAT_linear_2022
SAT_pred_21 = SAT_pred_21 + SAT_linear_2021
SAT_pred_22 = SAT_pred_22 + SAT_linear_2022
SAT_recon_21 = mask_land(SAT_recon_21,label='ocean', lonname='longitude')
SAT_recon_22 = mask_land(SAT_recon_22,label='ocean', lonname='longitude')
SAT_pred_21 = mask_land(SAT_pred_21,label='ocean', lonname='longitude')
SAT_pred_22 = mask_land(SAT_pred_22,label='ocean', lonname='longitude')


SAT_recon_21, cyclic_lons = add_cyclic_point(SAT_recon_21, coord=lon)
SAT_recon_22, cyclic_lons = add_cyclic_point(SAT_recon_22, coord=lon)
SAT_pred_21, cyclic_lons = add_cyclic_point(SAT_pred_21, coord=lon)
SAT_pred_22, cyclic_lons = add_cyclic_point(SAT_pred_22, coord=lon)

# 定义绘制预测SAT的hanshu
def drawing_SAT(cyclic_data1, cyclic_data2, Name1, Name2):
    # 设置字体为楷体,显示负号
    mpl.rcParams['font.sans-serif'] = ['KaiTi']
    mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=0))
    # 图名
    ax1.set_title('ERA5_H+W_MLR_'+str(Name1)+'2021-2MT')
    ax2.set_title('ERA5_H+W_MLR_'+str(Name2)+'2022-2MT')
    # ax1.add_patch(rectangle171)  # patch绘制方法
    # ax1.add_patch(rectangle172)  # patch绘制方法
    # ax1.add_patch(rectangle173)  # patch绘制方法
    # ax2.add_patch(rectangle171)  # patch绘制方法
    # ax2.add_patch(rectangle172)  # patch绘制方法
    # ax2.add_patch(rectangle173)  # patch绘制方法
    # ax1.add_patch(rectangle181)  # patch绘制方法
    # ax1.add_patch(rectangle182)  # patch绘制方法
    # ax1.add_patch(rectangle183)  # patch绘制方法
    # ax2.add_patch(rectangle181)  # patch绘制方法
    # ax2.add_patch(rectangle182)  # patch绘制方法
    # ax2.add_patch(rectangle183)  # patch绘制方法

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
    ax1.add_feature(cfeature.LAKES, edgecolor='black', lw=0.4)  # 边缘为黑色
    ax2.add_feature(cfeature.LAKES, edgecolor='black', lw=0.4)  # 边缘为黑色
    ax_cf1 = ax1.contourf(cyclic_lons, lat, cyclic_data1, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-2.4, 2.8, 0.4), extend='both')  # 绘制等值线图
    ax_cf2 = ax2.contourf(cyclic_lons, lat, cyclic_data2, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-2.4, 2.8, 0.4), extend='both')  # 绘制等值线图
    fig.subplots_adjust(right=0.9)
    position = fig.add_axes([0.93, 0.25, 0.015, 0.5])  # 位置[左,下,右,上]
    cb = fig.colorbar(ax_cf2, shrink=0.6, cax=position, extend='both')
    plt.savefig(r'F:\6_Scientific_Research\3_Teleconnection\2_Picture\3_Pred\2_era5\'ERA5_H+W_MLR_'+str(Name1)+'-2021-2022-2MT.png')
    plt.show()
# drawing_SAT(SAT_pred_19, SAT_pred_20, 19, 20)
# drawing_SAT(SAT_pred_21, SAT_pred_22, 21, 22)
# drawing_SAT(SAT_recon_21, SAT_recon_22, 'Reconstruction', 'Reconstruction')
drawing_SAT(SAT_pred_21, SAT_pred_22, 'Predict', 'Predict')