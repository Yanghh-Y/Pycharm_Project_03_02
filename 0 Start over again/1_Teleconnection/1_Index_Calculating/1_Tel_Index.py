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


# ---Read_Data-CIR_H--- #
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_H_197901-202208.nc')
# 筛选夏季平均
# 夏季平均的CIR_H_Sum[44, 37, 144]
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_H = f1.data[Sum_Index, 5, 0:27, :] # 筛选
Cir_H_Sum = np.zeros((44, 27, 144))
lat, lon, year = Cir_H.lat, Cir_H.lon, range(1979, 2023)
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_H_Sum[n, :, :] = (Cir_H[i, :, :] + Cir_H[i + 1, :, :] + Cir_H[i + 2, :, :]) / 3
        n = n + 1
Cir_H_Sum = xr.DataArray(Cir_H_Sum, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])


# ---Read_Data-CIR_W--- #
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_W_197901-202208.nc')
# 筛选夏季平均
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_W = f1.data[Sum_Index, 5, 0:27, :] # 筛选
Cir_W = Cir_W - Cir_W.mean(dim='lon') # 去纬度平均
Cir_W_Sum = np.zeros((44, 27, 144))
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_W_Sum[n, :, :] = (Cir_W[i, :, :] + Cir_W[i + 1, :, :] + Cir_W[i + 2, :, :]) / 3
        n = n + 1
Cir_W_Sum = xr.DataArray(Cir_W_Sum, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])



# --- Calculat Tel Index --- #
Df_H = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\HTel.csv')
Df_W = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\WTel.csv')
Tel_H_I = pd.DataFrame(columns=['A1A2', 'A3A4', 'B1B2', 'C1C2', 'C3C4', 'D1D2', 'E1E2', 'F1F2', 'G1G2', 'H1H2'])
for i in range(10):
    Tel_H_I.iloc[:, i] = (Cir_H_Sum.sel(lat=Df_H.A_lat[i], lon=Df_H.A_lon[i]) - Cir_H_Sum.sel(lat=Df_H.B_lat[i], lon=Df_H.B_lon[i])) * 0.5
Tel_W_I = pd.DataFrame(columns=['A1A2', 'A3A4', 'B1B2', 'B3B4', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
for i in range(8):
    Tel_W_I.iloc[:, i] = (Cir_W_Sum.sel(lat=Df_W.A_lat[i], lon=Df_W.A_lon[i]) - Cir_W_Sum.sel(lat=Df_W.B_lat[i], lon=Df_W.B_lon[i])) * 0.5



Tel_W_I.to_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\Tel_W_I.csv')
Tel_H_I.to_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\Tel_H_I.csv')





