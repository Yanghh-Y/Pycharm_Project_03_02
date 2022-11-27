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


#--read data--#
#--read data--#
TELIS = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\HI.csv', usecols=['B1B2', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
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

# 数据准备处理
scores_r2,scores_rmse = np.zeros((37, 144)), np.zeros((37, 144))
SAT_pred_19,SAT_pred_20 = np.zeros((37, 144)), np.zeros((37, 144))
SAT_pred_21,SAT_pred_22 = np.zeros((37, 144)), np.zeros((37, 144))
scores_r2_All,scores_rmse_All = np.zeros((37, 144)), np.zeros((37, 144))
for ilat in range(37):
    for ilon in range(144):
        print(ilat, ilon)
        TELI = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\HI.csv', usecols=['B1B2', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
        TELI['y']=SAT_ANO[:, ilat, ilon]
        X = TELI.drop(columns= "y")
        y = TELI.y
        # 准备训练逐步拟合回归模型
        final_vars, iterations_logs, reg_model = ss.BidirectionalStepwiseSelection(X, y, model_type ="linear", elimination_criteria = "aic", senter=0.05, sstay=0.05)
        yFit = reg_model.fittedvalues
        SAT_pred_19[ilat, ilon], SAT_pred_20[ilat, ilon] = yFit[40], yFit[41]
        SAT_pred_21[ilat, ilon], SAT_pred_22[ilat, ilon] = yFit[42], yFit[43]
        # scores_rmse[ilat, ilon] = mean_squared_error(y[50: ], yFit[50: ], sample_weight=None, multioutput='uniform_average')
        # scores_r2[ilat, ilon] = r2_score(y[50: ], yFit[50: ], sample_weight=None, multioutput='uniform_average')
        # scores_rmse_All[ilat, ilon] = mean_squared_error(y, yFit, sample_weight=None, multioutput='uniform_average')
        # scores_r2_All[ilat, ilon] = r2_score(y, yFit, sample_weight=None, multioutput='uniform_average')
# 将数据存下
# df_scores_RMSE = pd.DataFrame(scores_rmse)
# df_scores_RMSE.to_csv(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Model\SFR_DF_RMSE.csv')
# df_scores_r2 = pd.DataFrame(scores_r2)
# df_scores_r2.to_csv(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Model\SFR_DF_R2.csv')
# df_SAT_pred_17 = pd.DataFrame(SAT_pred_17)
# df_SAT_pred_18 = pd.DataFrame(SAT_pred_18)
# df_SAT_pred_17.to_csv(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Model\SFR_SAT_17.csv')
# df_SAT_pred_18.to_csv(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Model\SFR_SAT_18.csv')

#--绘图系列--#
# 这里预测的SAT17和SAT18是去除了线性趋势以后的，所以这里我们应该还要加上之前去除的线性趋势
# 数据准备
lat = np.arange(0, 92.5, 2.5)
lon = f1.lon
SAT_pred_19 = SAT_pred_19 + SAT_linear_2019
SAT_pred_20= SAT_pred_20 + SAT_linear_2020
SAT_pred_19, cyclic_lons = add_cyclic_point(SAT_pred_19, coord=lon)
SAT_pred_20, cyclic_lons = add_cyclic_point(SAT_pred_20, coord=lon)
SAT_pred_21 = SAT_pred_21 + SAT_linear_2021
SAT_pred_22 = SAT_pred_22 + SAT_linear_2022
SAT_pred_21, cyclic_lons = add_cyclic_point(SAT_pred_21, coord=lon)
SAT_pred_22, cyclic_lons = add_cyclic_point(SAT_pred_22, coord=lon)

# scores_rmse, cyclic_lons = add_cyclic_point(scores_rmse, coord=lon)
# scores_r2, cyclic_lons = add_cyclic_point(scores_r2, coord=lon)
# scores_rmse_All, cyclic_lons = add_cyclic_point(scores_rmse_All, coord=lon)
# scores_r2_All, cyclic_lons = add_cyclic_point(scores_r2_All, coord=lon)


# # 定义绘图函数：绘制r2和RMSE之中colorbar有两个
# def drawing_model(cyclic_data1, cyclic_data2):
#     # 设置字体为楷体,显示负号
#     mpl.rcParams['font.sans-serif'] = ['KaiTi']
#     mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
#     fig = plt.figure(figsize=(8, 6), dpi=400)
#     ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
#     ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=0))
#     # 图名
#     ax1.set_title('Stepwise_linear'+str(cyclic_data1))
#     ax2.set_title('Stepwise_linear'+str(cyclic_data2))
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
#     ax1.add_feature(cfeature.LAKES, edgecolor='black')  # 边缘为黑色
#     ax2.add_feature(cfeature.LAKES, edgecolor='black')  # 边缘为黑色
#     ax_cf1 = ax1.contourf(cyclic_lons, lat, cyclic_data1, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(0, 4, 0.5), extend='both')  # 绘制等值线图
#     ax_cf2 = ax2.contourf(cyclic_lons, lat, cyclic_data2, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=np.arange(-1, 1.25, 0.25), extend='both')  # 绘制等值线图
#     cb1 = fig.colorbar(ax_cf1, shrink=0.6, orientation='horizontal', extend='both', ax=ax1)
#     cb2 = fig.colorbar(ax_cf2, shrink=0.6, orientation='horizontal', extend='both',ax=ax2)
#     # plt.savefig(r'F:\Pictures\1_Graduation_Project\3_Hot_Summer\Model\Stepwise_linear_regression_RMSE_R2.png')
#     plt.show()
# # drawing_model(scores_rmse, scores_r2)

# 定义绘制预测SAT的hanshu
def drawing_SAT(cyclic_data1, cyclic_data2, name1, name2):
    # 设置字体为楷体,显示负号
    mpl.rcParams['font.sans-serif'] = ['KaiTi']
    mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=0))
    # 图名
    ax1.set_title('Stepwise_linear'+str(name1))
    ax2.set_title('Stepwise_linear'+str(name2))
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
    # plt.savefig(r'F:\Pictures\1_Graduation_Project\3_Hot_Summer\Model\Stepwise_linear_regression_SAT_Pred.png')
    plt.savefig(r'F:\6_Scientific_Research\3_Teleconnection\2_Picture\3_Pred\1_ncep\H_Stepwise_linear_regression'+str(name1)+'_'+str(name2)+'.png')

    plt.show()
drawing_SAT(SAT_pred_19, SAT_pred_20, 19, 20)
drawing_SAT(SAT_pred_21, SAT_pred_22, 21, 22)











