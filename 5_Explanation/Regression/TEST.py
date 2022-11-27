import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal
from scipy.stats.mstats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import linregress
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point #进行循环
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要
from matplotlib.patches import  Rectangle

import matplotlib.ticker as mticker


#--读取数据--#
# 读取遥相关指数
TELIS = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\HI.csv', usecols=['B1B2', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
TEL_name = TELIS.columns
# Read-GHT
# hgt-1950-2019-summer-mean
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA-GHT-2.5.nc')
lat = f1.latitude
lon = f1.longitude
hgt = f1.z.sel(expver=1, level=500).groupby('time.year').mean(dim='time', skipna=True)
hgt = hgt.sel(year=range(1979,2023)) # max58286, 52151
