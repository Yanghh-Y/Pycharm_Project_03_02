import scipy
from scipy import signal
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.util import add_cyclic_point #进行循环
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要

#---Read_Data---#
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_H_197901-202208.nc')
# 筛选夏季平均
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_H = f1.data[Sum_Index, 5, 0:37, :] # 筛选
Cir_H_Sum = np.zeros((44, 37, 144))
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_H_Sum[n, :, :] = (Cir_H[i, :, :] + Cir_H[i + 1, :, :] + Cir_H[i + 2, :, :]) / 3
        n = n + 1
# 去线性趋势
H = np.zeros((44, 37, 144))
Cir_H_detrend = scipy.signal.detrend(Cir_H_Sum, axis=0, type='linear', overwrite_data=False)
Cir_H_year_mean = Cir_H_Sum.mean(axis=0)
for iyear in range(44):
    H[iyear, :, :] = Cir_H_detrend[iyear, :, :] + Cir_H_year_mean[:, :]
lon = np.arange(0, 360, 2.5)
lat = np.arange(90, -2.5, -2.5)
year = np.arange(1979, 2023, 1)
H = xr.DataArray(H, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])


#---Read_Data---#
f2 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_W_197901-202208.nc')
# 筛选夏季平均
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_W = f2.data[Sum_Index, 5, 0:37, :] # 筛选
Cir_W_Sum = np.zeros((44, 37, 144))
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_W_Sum[n, :, :] = (Cir_W[i, :, :] + Cir_W[i + 1, :, :] + Cir_W[i + 2, :, :]) / 3
        n = n + 1
# 去线性趋势
W = np.zeros((44, 37, 144))
Cir_W_detrend = scipy.signal.detrend(Cir_W_Sum, axis=0, type='linear', overwrite_data=False)
Cir_W_year_mean = Cir_W_Sum.mean(axis=0)
for iyear in range(44):
    W[iyear, :, :] = Cir_W_detrend[iyear, :, :] + Cir_W_year_mean[:, :]
lon = np.arange(0, 360, 2.5)
lat = np.arange(90, -2.5, -2.5)
year = np.arange(1979, 2023, 1)
W = xr.DataArray(W, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])


# 读取端点信息
Df_H = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\H_Teleconnection.csv')
Df_W = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\W_Teleconnection.csv')
HI = pd.DataFrame(columns=['B1B2', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
cl = HI.columns
for i in range(len(cl)):
    HI[cl[i]] = (H.loc[:, Df_H.loc[i, 'Point1_lat'], Df_H.loc[i, 'Point1_lon']] - H.loc[:, Df_H.loc[i, 'Point2_lat'], Df_H.loc[i, 'Point2_lon']])*0.5
WI = pd.DataFrame(columns=['A1A2', 'A1A3', 'B1B2', 'B1B3', 'C1C2', 'C1C3', 'E1E2', 'E1E3'])
cl = WI.columns
for i in range(len(cl)):
    WI[cl[i]] = (W.loc[:, Df_W.loc[i, 'Point1_lat'], Df_W.loc[i, 'Point1_lon']] - W.loc[:, Df_W.loc[i, 'Point2_lat'], Df_W.loc[i, 'Point2_lon']])*0.5
HI.to_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\HI.csv')
WI.to_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\WI.csv')





