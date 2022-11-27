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
TELIS = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\WI.csv', usecols=['A1A2', 'A1A3', 'B1B2', 'B1B3', 'C1C2', 'C1C3', 'E1E2', 'E1E3'])
TEL_name = TELIS.columns
# Read-GHT
# hgt-1950-2019-summer-mean
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA-GHT-2.5.nc')
lat = f1.latitude
lon = f1.longitude
# 选时间，高度，求平均
hgt = f1.z.sel(expver=1, level=500).groupby('time.year').mean(dim='time', skipna=True)
hgt = hgt.sel(year=range(1979,2023))
hgt = hgt/9.8
# 去线性趋势
hgt_detrend = scipy.signal.detrend(hgt, axis=0, type='linear', overwrite_data=False)
hgt_year_mean = hgt.mean('year')
for iyear in range(44):
    hgt_detrend[iyear, :, :] = hgt_detrend[iyear, :, :] + hgt_year_mean[:, :]
# 遥相关指数标准化
def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    return x
TELIS['A1A2_std'], TELIS['A1A3_std'], TELIS['B1B2_std'], TELIS['B1B3_std'], TELIS['C1C2_std'], TELIS['C1C3_std'], TELIS['E1E2_std'], TELIS['E1E3_std']\
    = ZscoreNormalization(TELIS.A1A2), ZscoreNormalization(TELIS.A1A3), ZscoreNormalization(TELIS.B1B2), ZscoreNormalization(TELIS.B1B3), ZscoreNormalization(TELIS.C1C2), ZscoreNormalization(TELIS.C1C3), ZscoreNormalization(TELIS.E1E2), ZscoreNormalization(TELIS.E1E3)

# 计算回归
s,r,p = np.zeros((8, 37, 144)),np.zeros((8, 37, 144)),np.zeros((8, 37, 144))
for k in range(8): # 这一纬度是存放不同的遥相关指数对
    for i in range(37):
        for j in range(144):
            s[k, i, j], _, r[k, i, j], p[k ,i, j], _ = linregress(TELIS.iloc[:, 8+k], hgt_detrend[:, i, j])
            # TEL_index 中的7个遥相关指数作为自变量，每一个点上的SAT序列作为因变量

s, clons = add_cyclic_point(s, coord=lon)
p, clons = add_cyclic_point(p, coord=lon)
abcde = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']


# 定义绘制回归分布的等值线图
fig = plt.figure(figsize=(9,18), dpi=600)
fig.subplots_adjust(hspace=0.45) # 子图间距
ax1 = fig.add_subplot(8, 1, 1, projection=ccrs.PlateCarree(central_longitude=290))
ax2 = fig.add_subplot(8, 1, 2, projection=ccrs.PlateCarree(central_longitude=290))
ax3 = fig.add_subplot(8, 1, 3, projection=ccrs.PlateCarree(central_longitude=290))
ax4 = fig.add_subplot(8, 1, 4, projection=ccrs.PlateCarree(central_longitude=290))
ax5 = fig.add_subplot(8, 1, 5, projection=ccrs.PlateCarree(central_longitude=290))
ax6 = fig.add_subplot(8, 1, 6, projection=ccrs.PlateCarree(central_longitude=290))
ax7 = fig.add_subplot(8, 1, 7, projection=ccrs.PlateCarree(central_longitude=290))
ax8 = fig.add_subplot(8, 1, 8, projection=ccrs.PlateCarree(central_longitude=290))
axs2 = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
def drawingcount(Data, P,abc):
    for nrow in range(8):
        # 图题
        axs2[nrow].set_title('reg(W-'+TEL_name[(nrow)]+',GHT)')
        axs2[nrow].set_title(abc[nrow], loc='left')
        # 刻度形式
        axs2[nrow].set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
        axs2[nrow].set_yticks([0, 30, 60, 90])
        axs2[nrow].xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
        axs2[nrow].yaxis.set_major_formatter(LatitudeFormatter())
        # 调整图的内容：投影方式，海岸线
        axs2[nrow].add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4) # 添加海岸线
        ax_cf1 = axs2[nrow].contourf(clons, lat, Data[nrow,:,:], transform=ccrs.PlateCarree(), zorder=0, cmap='coolwarm', levels=np.arange(-16, 16, 2), extend='both')
        axs2[nrow].contourf(clons, lat, P[nrow,:,:], levels=[0, 0.05, 1], hatches=['...', None], zorder=1, colors="none", transform=ccrs.PlateCarree())
    fig.subplots_adjust(bottom=0.1)
    position = fig.add_axes([0.15, 0.05, 0.7, 0.015])  # 位置[左,下,右,上]
    cb = fig.colorbar(ax_cf1, shrink=0.6, cax=position, orientation='horizontal', extend='both')
    fig.savefig(r'F:\6_Scientific_Research\3_Teleconnection\2_Picture\4_Explanation\reg_W-tel_GHT-290.png')
    plt.show()

drawingcount(s, p, abcde)
