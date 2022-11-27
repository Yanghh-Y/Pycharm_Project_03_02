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
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA-T-UV.nc')
lat = f1.latitude
lon = f1.longitude
# 选时间，高度，求平均
Tem = f1.t.sel(expver=1, level=500).groupby('time.year').mean(dim='time', skipna=True)
Tem = Tem.sel(year=range(1979, 2023))
# hgt = hgt/9.8
# 去线性趋势
Tem_detrend = scipy.signal.detrend(Tem, axis=0, type='linear', overwrite_data=False)
Tem_year_mean = Tem.mean('year')
for iyear in range(44):
    Tem_detrend[iyear, :, :] = Tem_detrend[iyear, :, :] + Tem_year_mean[:, :]
# 遥相关指数标准化
def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    return x
TELIS['B1B2_std'], TELIS['C1C2_std'], TELIS['D1D2_std'], TELIS['E1E2_std'], TELIS['F1F2_std'],\
    = ZscoreNormalization(TELIS.B1B2), ZscoreNormalization(TELIS.C1C2), ZscoreNormalization(TELIS.D1D2), ZscoreNormalization(TELIS.E1E2), ZscoreNormalization(TELIS.F1F2)
# TEL_index.to_csv(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Teleconnectivity\Tel_index_STD.csv')

# 计算回归
s,r,p = np.zeros((5, 91, 360)),np.zeros((5, 91, 360)),np.zeros((5, 91, 360))
for k in range(5): # 这一纬度是存放不同的遥相关指数对
    for i in range(91):
        for j in range(360):
            s[k, i, j], _, r[k, i, j], p[k ,i, j], _ = linregress(TELIS.iloc[:, 5+k], Tem_detrend[:, i, j])
            # TEL_index 中的7个遥相关指数作为自变量，每一个点上的SAT序列作为因变量

s, clons = add_cyclic_point(s, coord=lon)
p, clons = add_cyclic_point(p, coord=lon)
abcde = ['(a)', '(b)', '(c)', '(d)', '(e)']
# 定义绘制回归分布的等值线图
fig = plt.figure(figsize=(9,18), dpi=600)
fig.subplots_adjust(hspace=0.3) # 子图间距
ax1 = fig.add_subplot(5, 1, 1, projection=ccrs.PlateCarree(central_longitude=290))
ax2 = fig.add_subplot(5, 1, 2, projection=ccrs.PlateCarree(central_longitude=290))
ax3 = fig.add_subplot(5, 1, 3, projection=ccrs.PlateCarree(central_longitude=290))
ax4 = fig.add_subplot(5, 1, 4, projection=ccrs.PlateCarree(central_longitude=290))
ax5 = fig.add_subplot(5, 1, 5, projection=ccrs.PlateCarree(central_longitude=290))
# ax6 = fig.add_subplot(7, 1, 6, projection=ccrs.PlateCarree(central_longitude=0))
# ax7 = fig.add_subplot(7, 1, 7, projection=ccrs.PlateCarree(central_longitude=0))
axs2 = [ax1, ax2, ax3, ax4, ax5]
def drawingcount(Data, P,abc):
    for nrow in range(5):
        # 图题
        axs2[nrow].set_title('reg(H-'+TEL_name[(nrow)]+',Tem)')
        axs2[nrow].set_title(abc[nrow], loc='left')
        # 刻度形式
        axs2[nrow].set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
        axs2[nrow].set_yticks([0, 30, 60, 90])
        axs2[nrow].xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
        axs2[nrow].yaxis.set_major_formatter(LatitudeFormatter())
        # 调整图的内容：投影方式，海岸线
        axs2[nrow].add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4) # 添加海岸线
        ax_cf1 = axs2[nrow].contourf(clons, lat, Data[nrow,:,:], transform=ccrs.PlateCarree(), zorder=0, cmap='coolwarm', levels=np.arange(-0.65, 0.7, 0.05), extend='both')
        axs2[nrow].contourf(clons, lat, P[nrow,:,:], levels=[0, 0.05, 1], hatches=['...', None], zorder=1, colors="none", transform=ccrs.PlateCarree())
    fig.subplots_adjust(bottom=0.1)
    position = fig.add_axes([0.15, 0.05, 0.7, 0.015])  # 位置[左,下,右,上]
    cb = fig.colorbar(ax_cf1, shrink=0.6, cax=position, orientation='horizontal', extend='both')
    fig.savefig(r'F:\6_Scientific_Research\3_Teleconnection\2_Picture\4_Explanation\reg_H-tel_Tem-290.png')
    plt.show()

drawingcount(s, p, abcde)
