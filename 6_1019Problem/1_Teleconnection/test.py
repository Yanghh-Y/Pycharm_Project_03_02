import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.util import add_cyclic_point #进行循环
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要
from Longitude_Transform import translon180_360

filepath = (r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA-GHT-2.5.nc')
lon_name = 'longitude'
f2 = translon180_360(filepath, lon_name)
z = f2.z.sel(expver=1, level=500).groupby('time.year').mean(dim='time', skipna=True)
z = z.sel(year=range(1979,2023))
z = z/9.8
z_detrend = scipy.signal.detrend(z, axis=0, type='linear', overwrite_data=False)

f3 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA-GHT-2.5.nc')
# 选时间，高度，求平均
hgt = f3.z.sel(expver=1, level=500).groupby('time.year').mean(dim='time', skipna=True)
hgt = hgt.sel(year=range(1979,2023))
hgt = hgt/9.8
hgt_detrend = scipy.signal.detrend(hgt, axis=0, type='linear', overwrite_data=False)
