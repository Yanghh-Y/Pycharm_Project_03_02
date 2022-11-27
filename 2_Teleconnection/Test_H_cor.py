import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.util import add_cyclic_point #进行循环
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要
import os

Point_DW = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\H_Teleconnection.csv')
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_H_197901-202208.nc')
# 筛选夏季平均
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_W = f1.data[Sum_Index, 5, 0:37, :] # 筛选
# 去纬度平均
Cir_W_Sum = np.zeros((44, 37, 144))
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_W_Sum[n, :, :] = (Cir_W[i, :, :] + Cir_W[i + 1, :, :] + Cir_W[i + 2, :, :]) / 3
        n = n + 1
lat, lon = Cir_W.lat, Cir_W.lon
year = np.arange(1979, 2023)
Cir_W_Sum = xr.DataArray(Cir_W_Sum, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])
r = np.zeros((9))
for i in range(len(Point_DW.Cor)):
    WD1 = Cir_W_Sum.sel(lat=Point_DW.Point1_lat[i], lon=Point_DW.Point1_lon[i])
    WD2 = Cir_W_Sum.sel(lat=Point_DW.Point2_lat[i], lon=Point_DW.Point2_lon[i])
    r[i], _ = pearsonr(WD1, WD2)
