import pandas as pd
import xarray as xr
import numpy as np
from scipy.stats import pearsonr


# -------- Read Cir_R --------#
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_R_197901-202208.nc')
# 筛选夏季平均
# 夏季平均的CIR_R_Sum[44, 37, 144]
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_R = f1.data[Sum_Index, 5, 0:27, :] # 筛选
Cir_R_Sum = np.zeros((44, 27, 144))
lat, lon = Cir_R.lat, Cir_R.lon
year = range(1979, 2023)
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_R_Sum[n, :, :] = (Cir_R[i, :, :] + Cir_R[i + 1, :, :] + Cir_R[i + 2, :, :]) / 3
        n = n + 1
Cir_R_Sum = xr.DataArray(Cir_R_Sum, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])


# --------- Calculating --------#
RTel = pd.DataFrame(columns=['Min_r', 'A_lat', 'A_lon', 'B_lat', 'B_lon'])
RTel.A_lat = [65, 40, 32.5, 40, 40, 57.5, 65, 70]
RTel.A_lon = [307.5, 22.5, 325, 300, 297.5, 260, 250, 127.5]
RTel.B_lat = [52.5, 57.5, 57.5, 60, 62.5, 75, 85, 70]
RTel.B_lon = [355, 2.5, 342.5, 322.5, 295, 197.5, 162.5, 217.5]
for i in range(8):
    RTel.iloc[i, 0], p = pearsonr(Cir_R_Sum.sel(lat=RTel.A_lat[i], lon=RTel.A_lon[i]), Cir_R_Sum.sel(lat=RTel.B_lat[i], lon=RTel.B_lon[i]))

# --- Calculat Tel Index --- #
Df_H = RTel
Tel_R_I = pd.DataFrame(columns=['A1A2', 'A3A4', 'B1B2', 'C1C2', 'C3C4', 'D1D2', 'E1E2', 'F1F2'])
for i in range(8):
    Tel_R_I.iloc[:, i] = (Cir_R_Sum.sel(lat=Df_H.A_lat[i], lon=Df_H.A_lon[i]) - Cir_R_Sum.sel(lat=Df_H.B_lat[i], lon=Df_H.B_lon[i])) * 0.5
Tel_R_I.C1C2 = (Tel_R_I.C1C2 + Tel_R_I.C3C4) * 0.5
Te = Tel_R_I.drop('C3C4', axis=1)

Te.to_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\R\Tel_R_I.csv')




