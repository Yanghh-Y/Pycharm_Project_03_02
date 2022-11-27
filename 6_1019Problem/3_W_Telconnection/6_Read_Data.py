import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.util import add_cyclic_point #进行循环
from matplotlib import patches
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要

#--读取数据--#
# 相关系数以及P值筛选
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\R_P\Cir_W_De_Lattice_r_all_lon_lat.nc' ,engine='netcdf4')
allr = f1.r.sel(ilat=np.arange(90, 22.5, -2.5), dlat=np.arange(90, 22.5, -2.5))
