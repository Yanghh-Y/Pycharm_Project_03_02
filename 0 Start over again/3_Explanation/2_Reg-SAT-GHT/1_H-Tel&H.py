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
TELIS = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\Index\Tel_Index_Normal.csv', usecols=['H_A1A2', 'H_B1B2', 'H_C1C2', 'H_D1D2', 'H_E1E2', 'H_F1F2', 'H_H1H2'])
TEL_name = TELIS.columns
# read-H
# hgt-1950-2019-summer-mean
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Stream\Stream_nc\cir_H_197901-202208.nc')
# 筛选夏季平均
lon = f1.lon
lat = np.arange(90, -2.5, -2.5)
Sum_Index = [i for i in range(524) if i%12 in [5, 6, 7]] # 筛选
Cir_H = f1.data[Sum_Index, 5, 0:37, :] # 筛选
Cir_H_Sum = np.zeros((44, 37, 144))
n = 0
for i in range(132): # 计算夏季平均
    if i%3 == 0:
        Cir_H_Sum[n, :, :] = (Cir_H[i, :, :] + Cir_H[i + 1, :, :] + Cir_H[i + 2, :, :]) / 3
        n = n + 1


# 计算回归
s,r,p = np.zeros((7, 37, 144)),np.zeros((7, 37, 144)),np.zeros((7, 37, 144))
for k in range(7): # 这一纬度是存放不同的遥相关指数对
    for i in range(37):
        for j in range(144):
            s[k, i, j], _, r[k, i, j], p[k ,i, j], _ = linregress(TELIS.iloc[:, k], Cir_H_Sum[:, i, j])
            # TEL_index 中的7个遥相关指数作为自变量，每一个点上的SAT序列作为因变量
s, clons = add_cyclic_point(s, coord=lon)
p, clons = add_cyclic_point(p, coord=lon)
abcde = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

# 定义绘制回归分布的等值线图
fig = plt.figure(figsize=(9,18), dpi=400)
fig.subplots_adjust(hspace=0.3) # 子图间距
ax1 = fig.add_subplot(7, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
ax2 = fig.add_subplot(7, 1, 2, projection=ccrs.PlateCarree(central_longitude=0))
ax3 = fig.add_subplot(7, 1, 3, projection=ccrs.PlateCarree(central_longitude=0))
ax4 = fig.add_subplot(7, 1, 4, projection=ccrs.PlateCarree(central_longitude=0))
ax5 = fig.add_subplot(7, 1, 5, projection=ccrs.PlateCarree(central_longitude=0))
ax6 = fig.add_subplot(7, 1, 6, projection=ccrs.PlateCarree(central_longitude=0))
ax7 = fig.add_subplot(7, 1, 7, projection=ccrs.PlateCarree(central_longitude=0))
axs2 = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
def drawingcount(Data, P, abc):
    for nrow in range(7):
        # 图题
        axs2[nrow].set_title('reg(H-'+str(TEL_name[nrow])+',Cir_H)')
        axs2[nrow].set_title(abc[nrow], loc='left')
        # 刻度形式
        axs2[nrow].set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
        axs2[nrow].set_yticks([0, 30, 60, 90])
        axs2[nrow].xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
        axs2[nrow].yaxis.set_major_formatter(LatitudeFormatter())
        # 调整图的内容：投影方式，海岸线
        axs2[nrow].add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4) # 添加海岸线
        ax_cf1 = axs2[nrow].contourf(clons, lat, Data[nrow,:,:], transform=ccrs.PlateCarree(), zorder=0, cmap='coolwarm', levels=np.arange(-0.05, 0.06, 0.01), extend='both')
        axs2[nrow].contourf(clons, lat, P[nrow,:,:], levels=[0, 0.05, 1], hatches=['...', None], zorder=1, colors="none", transform=ccrs.PlateCarree())
    fig.subplots_adjust(bottom=0.1)
    position = fig.add_axes([0.15, 0.05, 0.7, 0.015])  # 位置[左,下,右,上]
    cb = fig.colorbar(ax_cf1, shrink=0.6, cax=position, orientation='horizontal', extend='both')
    fig.savefig(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\3_Explanation\2_Reg\reg_H-tel_Cir_H.png')
    plt.show()

drawingcount(s, p, abcde)
