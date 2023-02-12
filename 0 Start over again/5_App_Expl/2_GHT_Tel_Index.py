import xarray as xr
import pandas as pd
import numpy as np
import Longitude_Transform
from eofs.standard import Eof
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader
from matplotlib.path import Path
from cartopy.mpl.patch import geos_to_path

# --------- Read Data --------- #
latrange = np.arange(0, 92.5, 2.5)
filepath = r'Z:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA5_T_U_V_25_1.nc'
lon_name = 'longitude'
fi = Longitude_Transform.translon180_360(filepath, lon_name)
hgt_sum = fi.z.sel(latitude=latrange).loc[(fi.time.dt.month.isin([6, 7, 8]))].groupby('time.year').mean(dim='time', skipna=True) / 9.8
v_sum = fi.v.sel(latitude=latrange).loc[(fi.time.dt.month.isin([6, 7, 8]))].groupby('time.year').mean(dim='time', skipna=True)


# -------- Tel_index_Cal -------- #
# ---- CGT—Index ----- #
CGT_lat = np.arange(35, 42.5, 2.5)
CGT_lon = np.arange(60, 70, 2.5)
CGT_lev = 200
CGTI = hgt_sum.sel(level=CGT_lev, longitude=CGT_lon, latitude=CGT_lat).mean(dim='latitude').mean(dim='longitude')
CGTI = np.array((CGTI - CGTI.mean())/CGTI.std())


# ---- SRP-Index ---- #
SRP_lat = np.arange(20, 62.5, 2.5)
SRP_lon = np.arange(0, 152.5, 2.5)
SRP_lev = 200
SRP_V_mean = v_sum.sel(level=SRP_lev, longitude=SRP_lon, latitude=SRP_lat).mean(dim='latitude').mean(dim='longitude')
SRP_V = v_sum.sel(level=SRP_lev, longitude=SRP_lon, latitude=SRP_lat) - SRP_V_mean
# -- EOF -- #
# 计算纬度权重
SRP_V_arr = np.array(SRP_V)
lat1 = np.array(SRP_lat)
coslat = np.cos(np.deg2rad(lat1))
wgts = np.sqrt(coslat)[..., np.newaxis]
# 创建EOF分解器
solver = Eof(SRP_V_arr, weights=wgts)
# 获取前三个模态，获取对应的PC序列和解释方差
eof = solver.eofsAsCorrelation(neofs=4)
pc = solver.pcs(npcs=4, pcscaling=1)
var = solver.varianceFraction()
# SRPI
SRPI = pc[:, 0]
SRPI = (SRPI - SRPI.mean()) / SRPI.std()

# ---- SAHI ---- #
SAH1_lat = np.arange(20, 30, 2.5)
SAH1_lon = np.arange(85, 117.5, 2.5)
SAH2_lat = np.arange(27.5, 37.5, 2.5)
SAH2_lon = np.arange(50, 82.5, 2.5)
SAH_lev = 200
SAHI = hgt_sum.sel(level=SAH_lev, longitude=SAH1_lon, latitude=SAH1_lat).mean(dim='latitude').mean(dim='longitude') \
    - hgt_sum.sel(level=SAH_lev, longitude=SAH2_lon, latitude=SAH2_lat).mean(dim='latitude').mean(dim='longitude')
SAHI = np.array((SAHI - SAHI.mean()) / SAHI.std())

# ---- WPNAI ---- #
WPNA_lat = np.arange(0, 92.5, 2.5)
WPNA_lon = np.arange(0, 360, 2.5)
WPNA_lev = 200
WPNA_hgt = hgt_sum.sel(level=WPNA_lev, longitude=WPNA_lon, latitude=WPNA_lat)
# -- EOF -- #
# 计算纬度权重
WPNA_hgt_arr = np.array(WPNA_hgt)
lat1 = np.array(WPNA_lat)
coslat = np.cos(np.deg2rad(lat1))
wgts = np.sqrt(coslat)[..., np.newaxis]
# 创建EOF分解器
solver = Eof(WPNA_hgt_arr, weights=wgts)
# 获取前三个模态，获取对应的PC序列和解释方差
eof = solver.eofsAsCorrelation(neofs=3)
pc = solver.pcs(npcs=3, pcscaling=1)
var = solver.varianceFraction()
# WPNAI
WPNAI = pc[:, 0]
WPNAI = (WPNAI - WPNAI.mean()) / WPNAI.std()

# ---- EAPI ---- #
EAPI_hgt = hgt_sum - hgt_sum.mean(dim='latitude').mean(dim='longitude')
EAPI = 0.25 * EAPI_hgt.sel(level=500, longitude=125, latitude=20) - \
    0.5 * EAPI_hgt.sel(level=500, longitude=125, latitude=40) + \
    0.25 * EAPI_hgt.sel(level=500, longitude=125, latitude=60)
EAPI = np.array((EAPI - EAPI.mean()) / EAPI.std())

# ---- EU ---- #
EUI_hgt = hgt_sum - hgt_sum.mean(dim='latitude').mean(dim='longitude')
# -- EUI SCAND -- #
EUI = -0.25 * EUI_hgt.sel(level=500, longitude=20, latitude=55) + \
      0.5 * EUI_hgt.sel(level=500, longitude=75, latitude=55) - \
      0.25 * EUI_hgt.sel(level=500, longitude=145, latitude=40)
EUI = np.array((EUI - EUI.mean()) / EUI.std())
# -- EUII WRUS -- #
EUII = -0.25 * EUI_hgt.sel(level=500, longitude=10, latitude=60) + \
      0.5 * EUI_hgt.sel(level=500, longitude=50, latitude=50) - \
      0.25 * EUI_hgt.sel(level=500, longitude=110, latitude=45)
EUII = np.array((EUII - EUII.mean()) / EUII.std())

# ---- AEAI ---- #
AEA_hgt = hgt_sum.sel(level=500) - hgt_sum.sel(level=500).mean(dim='latitude').mean(dim='longitude')
AEAI = (1/6) * (AEA_hgt.sel(longitude=320, latitude=32.5) + AEA_hgt.sel(longitude=25, latitude=57.5) + AEA_hgt.sel(longitude=107.5, latitude=42.5)) \
    - 0.25 * (AEA_hgt.sel(longitude=342.5, latitude=52.5) + AEA_hgt.sel(longitude=70, latitude=72.5))
AEAI = np.array((AEAI - AEAI.mean()) / AEAI.std())

# ---- BBCI ---- #
BBC_lat = np.arange(50, 82.5, 2.5)
BBC_lon1 = np.arange(340, 360, 2.5)
BBC_lon2 = np.arange(0, 152.5, 2.5)
BBC_lon = np.concatenate([BBC_lon2, BBC_lon1],axis=0)
BBC_lev = 250
BBC_v = v_sum.sel(level=BBC_lev, latitude=BBC_lat, longitude=BBC_lon)
# -- EOF -- #
# 计算纬度权重
BBC_v_arr = np.array(BBC_v)
lat1 = np.array(BBC_lat)
coslat = np.cos(np.deg2rad(lat1))
wgts = np.sqrt(coslat)[..., np.newaxis]
# 创建EOF分解器
solver = Eof(BBC_v_arr, weights=wgts)
# 获取前三个模态，获取对应的PC序列和解释方差
eof = solver.eofsAsCorrelation(neofs=3)
pc = solver.pcs(npcs=3, pcscaling=1)
var = solver.varianceFraction()
# -- BBCI -- #
BBCI = pc[:, 0]
BBCI = (BBCI - BBCI.mean()) / BBCI.std()
# -- BOCI -- #
BOCI = pc[:, 1]
BOCI = (BOCI - BOCI.mean()) / BOCI.std()


# -------- DataFrame --------- #
df1 = pd.DataFrame({
    'CGTI': CGTI, 'SRPI': SRPI, 'SAHI': SAHI, 'WPNAI': WPNAI, 'EAPI': EAPI, 'EUI': EUI, 'EUII': EUII, 'AEAI': AEAI, 'BBCI': BBCI, 'BOCI': BOCI   })
df1.to_csv(r'Z:\6_Scientific_Research\3_Teleconnection\1_Data\Index\HGT_TEL_INDEX_CAL.csv')








