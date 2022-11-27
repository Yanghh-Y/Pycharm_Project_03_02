import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal # 用于去线性趋势
import sys
sys.path.append(r'E:\Pycharm\Pycharm_Project_02\0_python_bidirectional_stepwise_selection-master\BidirectionalStep')
import BidirectionalStepwiseSelection as ss # 写的多元线性回归的函数
from sklearn.metrics import mean_squared_error, r2_score # 用于评价模型的函数
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point #进行循环
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要

TELIS_H = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\WI.csv', usecols=['A1A2', 'A1A3', 'B1B2', 'B1B3', 'C1C2', 'C1C3', 'E1E2', 'E1E3'])
TELIS_W = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\HI.csv', usecols=['B1B2', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
TELIS = pd.concat([TELIS_H, TELIS_W],axis=1)
col = ['W_A1A2', 'W_A1A3', 'W_B1B2', 'W_B1B3', 'W_C1C2', 'W_C1C3', 'W_E1E2', 'W_E1E3', 'B1B2', 'C1C2', 'D1D2', 'E1E2',
       'F1F2']
TELIS.columns = col

f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\air.mon.mean.nc')
SAT = f1.air.loc[(f1.time.dt.month.isin([6, 7, 8]))].loc['1979-01-01':'2022-08-01'].sel(level=1000).sel(lat=np.arange(0, 92.5, 2.5)).groupby('time.year').mean(dim='time', skipna=True)
lat = SAT.lat
lon = SAT.lon
# SAT气候态平均
SAT_CL = SAT.loc[1971:2000].mean('year')
# SAT去线性趋势
SAT_ANO = np.zeros((44, 37, 144))
SAT_detrend = scipy.signal.detrend(SAT, axis=0, type='linear', overwrite_data=False)
SAT_year_mean = SAT.mean('year')
for iyear in range(44):
    SAT_detrend[iyear, :, :] = SAT_detrend[iyear, :, :] + SAT_year_mean[:, :]
    SAT_ANO[iyear, :, :] = SAT_detrend[iyear, :, :] - SAT_CL[:, :]
# 将17,18年的线性分量提取出来
SAT_linear = SAT - SAT_detrend
SAT_linear_2019 = SAT_linear[-4, :, :]
SAT_linear_2020 = SAT_linear[-3, :, :]
SAT_linear_2021 = SAT_linear[-2, :, :]
SAT_linear_2022 = SAT_linear[-1, :, :]

#--Training-model--#
# time-series-split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression  #线性回归
from sklearn.metrics import mean_squared_error, r2_score
# 准备开始训练模型
# scores_r2,scores_rmse = np.zeros((37, 144)), np.zeros((37, 144))
SAT_pred_19,SAT_pred_20 = np.zeros((37, 144)), np.zeros((37, 144))
SAT_pred_21,SAT_pred_22 = np.zeros((37, 144)), np.zeros((37, 144))

for ilat in range(37):
    for ilon in range(144):
        # 划分了数据集为训练集和测试集
        X_0= np.array(TELIS)
        Y_0 = np.array(SAT_ANO[:, ilat, ilon]) # 某一格点上的温度异常序列
        X = X_0[0:39, :]
        Y = Y_0[0:39]
        X_Test = X_0[40:44, :]
        Y_Test = Y_0[40:44]
        tscv = TimeSeriesSplit(n_splits=3, max_train_size=40, test_size=10)
        for train_index, test_index in tscv.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
        linreg = LinearRegression()
        lin_model = linreg.fit(X, Y)
        # scores_cross[ilat, ilon] = cross_val_score(lin_model,X,Y,cv=tscv, scoring='r2').mean()
        y_pred = lin_model.predict(X_Test)
        SAT_pred_19[ilat, ilon] = y_pred[0]
        SAT_pred_20[ilat, ilon] = y_pred[1]
        SAT_pred_21[ilat, ilon] = y_pred[2]
        SAT_pred_22[ilat, ilon] = y_pred[3]
        # scores_rmse[ilat, ilon] = mean_squared_error(Y_Test, y_pred, sample_weight=None, multioutput='uniform_average')
        # scores_r2[ilat, ilon] = r2_score(Y_Test, y_pred, sample_weight=None, multioutput='uniform_average')



lat = np.arange(0, 92.5, 2.5)
lon = f1.lon
SAT_pred_19 = SAT_pred_19 + SAT_linear_2019
SAT_pred_20 = SAT_pred_20 + SAT_linear_2020
SAT_pred_21 = SAT_pred_21 + SAT_linear_2021
SAT_pred_22 = SAT_pred_22 + SAT_linear_2022
# scores_rmse, cyclic_lons = add_cyclic_point(scores_rmse, coord=lon)
# scores_r2, cyclic_lons = add_cyclic_point(scores_r2, coord=lon)
# scores_rmse_All, cyclic_lons = add_cyclic_point(scores_rmse_All, coord=lon)
# scores_r2_All, cyclic_lons = add_cyclic_point(scores_r2_All, coord=lon)
SAT_pred_19, cyclic_lons = add_cyclic_point(SAT_pred_19, coord=lon)
SAT_pred_20, cyclic_lons = add_cyclic_point(SAT_pred_20, coord=lon)
SAT_pred_21, cyclic_lons = add_cyclic_point(SAT_pred_21, coord=lon)
SAT_pred_22, cyclic_lons = add_cyclic_point(SAT_pred_22, coord=lon)
# # 定义绘图函数：绘制r2和RMSE之中colorbar有两个
# def drawing_model(cyclic_data1, cyclic_data2):
#     # 设置字体为楷体,显示负号
#     mpl.rcParams['font.sans-serif'] = ['KaiTi']
#     mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
#     fig = plt.figure(figsize=(8, 6), dpi=400)
#     ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
#     ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=0))
#     # 图名
#     ax1.set_title('Multiple_linear_regression_RMSE')
#     ax2.set_title('Multiple_linear_regression_R2')
#     ax1.add_patch(rectangle171)  # patch绘制方法
#     ax1.add_patch(rectangle172)  # patch绘制方法
#     ax1.add_patch(rectangle173)  # patch绘制方法
#     ax2.add_patch(rectangle171)  # patch绘制方法
#     ax2.add_patch(rectangle172)  # patch绘制方法
#     ax2.add_patch(rectangle173)  # patch绘制方法
#     ax1.add_patch(rectangle181)  # patch绘制方法
#     ax1.add_patch(rectangle182)  # patch绘制方法
#     ax1.add_patch(rectangle183)  # patch绘制方法
#     ax2.add_patch(rectangle181)  # patch绘制方法
#     ax2.add_patch(rectangle182)  # patch绘制方法
#     ax2.add_patch(rectangle183)  # patch绘制方法
#     # 刻度线形式
#     ax1.set_xticks([-180, -150, -120, -90, -60, -30,0, 30, 60, 90, 120, 150, 180]) #指定要显示的经纬度
#     ax1.set_yticks([0, 30, 60, 90])
#     ax1.xaxis.set_major_formatter(LongitudeFormatter()) #刻度格式转换为经纬度样式
#     ax1.yaxis.set_major_formatter(LatitudeFormatter())
#     ax2.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
#     ax2.set_yticks([0, 30, 60, 90])
#     ax2.xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
#     ax2.yaxis.set_major_formatter(LatitudeFormatter())
#     # 调整图的内容：投影方式，海岸线
#     ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4) #添加海岸线
#     ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4) #添加海岸线
#     ax1.add_feature(cfeature.LAKES, edgecolor='black', lw=0.4)  # 边缘为黑色
#     ax2.add_feature(cfeature.LAKES, edgecolor='black', lw=0.4)  # 边缘为黑色
#     ax_cf1 = ax1.contourf(cyclic_lons, lat, cyclic_data1, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(0, 4, 0.5), extend='both')  # 绘制等值线图
#     ax_cf2 = ax2.contourf(cyclic_lons, lat, cyclic_data2, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-1, 1.25, 0.25), extend='both')  # 绘制等值线图
#     cb1 = fig.colorbar(ax_cf1, shrink=0.6, orientation='horizontal', extend='both', ax=ax1)
#     cb2 = fig.colorbar(ax_cf2, shrink=0.6, orientation='horizontal', extend='both',ax=ax2)
#     # path = 'F://Pictures//1_Graduation_Project//3_Hot_Summer//Model_com//' + pathname + '.png'
#     # plt.savefig(r'F:\Pictures\1_Graduation_Project\3_Hot_Summer\Model_com\111.png')
#     plt.show()
# drawing_model(scores_rmse, scores_r2)







# 定义绘制预测SAT的hanshu
def drawing_SAT(cyclic_data1, cyclic_data2, Name1, Name2):
    # 设置字体为楷体,显示负号
    mpl.rcParams['font.sans-serif'] = ['KaiTi']
    mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=0))
    # 图名
    ax1.set_title('H+W_Multiple_linear'+str(Name1))
    ax2.set_title('H+W_Multiple_linear'+str(Name2))
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
    plt.savefig(r'F:\6_Scientific_Research\3_Teleconnection\2_Picture\3_Pred\1_ncep\H+W_Linear_regression'+str(Name1)+'_'+str(Name2)+'.png')
    plt.show()
drawing_SAT(SAT_pred_19, SAT_pred_20, 19, 20)
drawing_SAT(SAT_pred_21, SAT_pred_22, 21, 22)
