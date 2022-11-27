import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


#---创建一个DataFrame存储DW-Tel的相关系数和点的位置---#
Point_DW = pd.DataFrame({
    "Name":['A1A2', 'B1B2', 'C1C2', 'D1D2', 'E1E2', 'F1F2', 'G1G2', 'H1H2', 'I1I2'],
    "Cor":[0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Point1_lat":[27.5, 27.5, 72.5, 90, 70, 30, 65, 72.5, 72.5],
    "Point1_lon":[167.5, 32.5, 107.5, 165, 340, 185, 205, 320, 22.5],
    "Point2_lat":[27.5, 27.5, 47.5, 90, 45, 45, 45, 70, 50],
    "Point2_lon":[347.5, 212.5, 100, 265, 355, 257.5, 197.5, 132.5, 17.5]
})
# 读取Cir_W，处理成DW，计算COR
#---Read_Data---#
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_W_197901-202208.nc')
# 筛选夏季平均
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_W = f1.data[Sum_Index, 5, 0:37, :] # 筛选
# 去纬度平均
Cir_W = Cir_W - Cir_W.mean(dim='lon')
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
