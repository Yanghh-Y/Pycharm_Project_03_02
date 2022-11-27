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
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point #进行循环
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要
from matplotlib.patches import  Rectangle

# 遥相关指数
TELIS = pd.read_csv(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Teleconnectivity\Tel_index_STD.csv', usecols=['NAI_STD', 'EUI_STD', 'EAI_STD', 'WAI_STD', 'CNAI_STD', 'ANAI_STD',
       'BSI_STD'])
# 计算T—N WAVE 的三个变量
# z300，u300，v300 为2017,2018年夏季的300hPa位势高度场，U、V风场的月平均
# z_tmp，u_tmp，v_tmp为1979-2018年夏季月的气候态
f_z = xr.open_dataset(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Ncep\hgt.mon.mean.nc')
f_u = xr.open_dataset(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Ncep\uwnd.mon.mean.nc')
f_v = xr.open_dataset(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Ncep\vwnd.mon.mean.nc')
# 17,18年夏季ZUV
latrange = np.arange(0, 92.5, 2.5)
z300 = f_z.hgt.sel(level=300).loc[(f_z.time.dt.month.isin([6, 7, 8]))].loc['1949-01-01':'2018-12-01']
u300 = np.array(f_u.uwnd.sel(level=300).loc[(f_u.time.dt.month.isin([6, 7, 8]))].loc['2017-01-01':'2018-12-01'].mean('time'))
v300 = np.array(f_v.vwnd.sel(level=300).loc[(f_v.time.dt.month.isin([6, 7, 8]))].loc['2017-01-01':'2018-12-01'].mean('time'))

# 气候态ZUV
ztmp = np.array(f_z.hgt.sel(level=300).loc[(f_z.time.dt.month.isin([6, 7, 8]))].loc['1949-01-01':'2018-12-01'].mean('time'))/9.8
utmp = np.array(f_u.uwnd.sel(level=300).loc[(f_u.time.dt.month.isin([6, 7, 8]))].loc['1949-01-01':'2018-12-01'].mean('time'))
vtmp = np.array(f_v.vwnd.sel(level=300).loc[(f_v.time.dt.month.isin([6, 7, 8]))].loc['1949-01-01':'2018-12-01'].mean('time'))

# lat和lon
lat, lon = f_z.lat, f_z.lon

# 一些常数
# 要把经纬度转换成角度量，所以做（*np.pi/180.0）处理
# 因为最终要计算Fx,Fy，所以统一数组shape，使用.reshape((1,-1))或(-1,1)处理
# 只有经度维的使用((1,-1))，只有纬度维的使用((-1,1))
a = 6400000 #地球半径
omega = 7.292e-5 # 自转角速度
lev = 300/1000 # p/p0

dlon=(np.gradient(lon)*np.pi/180.0).reshape((1,-1))
dlat=(np.gradient(lat)*np.pi/180.0).reshape((-1,1))
coslat = (np.cos(np.array(lat)*np.pi/180)).reshape((-1,1))
sinlat = (np.sin(np.array(lat)*np.pi/180)).reshape((-1,1))

#计算科氏力
f = np.array(2*omega*np.sin(lat*np.pi/180.0)).reshape((-1,1))
#计算|U|
wind = np.sqrt(utmp**2+vtmp**2)
#计算括号外的参数，a^2可以从括号内提出
c = (lev)*coslat/(2*a*a*wind)
# 回归到遥相关指数的Z300
# 计算与指数相关的streamf
z300 = z300.groupby('time.year').mean(dim='time', skipna=True)
z300 = z300/9.8
s, r, Z300 = np.zeros((73, 144)), np.zeros((73, 144)), np.zeros((73, 144))
for i in range(73):
       for j in range(144):
              s[i, j], _, r[i, j], Z300[i, j], _ = linregress(TELIS.iloc[:, 6], z300[:, i, j])
#Φ`
za = Z300 - ztmp.mean(1).reshape((-1,1))
#Ψ`
g = 9.8
streamf = g*za/f

# # 计算与指数相关的streamf
# s,r,streamf = np.zeros((7, 37, 144)),np.zeros((7, 37, 144)),np.zeros((7, 37, 144))
# for i in range(37):
#        for j in range(144):
#               s[i, j], _, r[i, j], streamf[i, j], _ = linregress(TELIS.BSI_STD, streamff[i,j])

# 计算各个部件，难度在于二阶导，变量的名字应该可以很容易看出我是在计算哪部分
dzdlon = np.gradient(streamf, axis = 1)/dlon
ddzdlonlon = np.gradient(dzdlon, axis = 1)/dlon
dzdlat = np.gradient(streamf, axis = 0)/dlat
ddzdlatlat = np.gradient(dzdlat, axis = 0)/dlat
ddzdlatlon = np.gradient(dzdlat, axis = 1)/dlon
# 这是X,Y分量共有的部分
x_tmp = dzdlon*dzdlon-streamf*ddzdlonlon
y_tmp = dzdlon*dzdlat-streamf*ddzdlatlon
# 计算两个分量
fx = c * ((utmp/coslat/coslat)*x_tmp+vtmp*y_tmp/coslat)
fy = c * ((utmp/coslat)*y_tmp+vtmp*x_tmp)


fig = plt.figure(figsize=(12,8))
proj = ccrs.PlateCarree(central_longitude=180)
leftlon, rightlon, lowerlat, upperlat = (0,180,0,90)
img_extent = [leftlon, rightlon, lowerlat, upperlat]
ax = fig.add_axes([0.1, 0.1, 0.8, 0.6],projection = proj)
ax.set_extent(img_extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.set_xticks(np.arange(leftlon,rightlon+60,60), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(lowerlat,upperlat+30,30), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
lat_formatter = cticker.LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
# ax.quiver(lon[::2],lat[::2],u_cli[0,::2,::2],v_cli[0,::2,::2],transform=ccrs.PlateCarree(),scale=150,color='r')
ax.contourf(lon, lat, streamf, transform=ccrs.PlateCarree(), cmap='coolwarm')
ax.quiver(lon[::2], lat[::2], fx[::2,::2], fy[::2,::2], wind[::2,::2], transform=ccrs.PlateCarree(), scale=100, cmap=plt.cm.jet)
plt.show()
















