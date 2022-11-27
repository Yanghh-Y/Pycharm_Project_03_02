import numpy as np
import xarray as xr
import pandas as pd
import scipy
from scipy import signal
from scipy.stats import linregress
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature #用于添加地理属性的库
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import earth_avg_radius
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point #进行循环
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要



#---遥相关指数---#
teli = pd.read_csv(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Teleconnectivity\Tel_index_STD.csv', usecols=['NAI_STD', 'EUI_STD', 'EAI_STD', 'WAI_STD', 'CNAI_STD', 'ANAI_STD',
       'BSI_STD'])


#---一些常数---#
p = 200 * units('hPa')  # 也有用300hPa的
mon = 1  # 目标月
time_target = f'2015-{mon:02d}-08'  # 目标日期
p0 = 1000 * units('hPa')
latrange = np.arange(0, 92.5, 2.5)

#---读取数据---#
f_z = xr.open_dataset(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Ncep\hgt.mon.mean.nc')
f_u = xr.open_dataset(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Ncep\uwnd.mon.mean.nc')
f_v = xr.open_dataset(r'F:\7_Data\0_Science_Program\1_Graduation_Program\Ncep\vwnd.mon.mean.nc')
# 读取200hPa的GHT
hgt = f_z.hgt.sel(level=200, lat=latrange).loc[(f_z.time.dt.month.isin([6, 7, 8]))].loc['1950-01-01':'2019-12-01'].groupby('time.year').mean(dim='time', skipna=True)
# 去线性趋势
hgt_detrend = scipy.signal.detrend(hgt, axis=0, type='linear', overwrite_data=False)
hgt_year_mean = hgt.mean('year')
for iyear in range(70):
    hgt_detrend[iyear, :, :] = hgt_detrend[iyear, :, :] + hgt_year_mean[:, :]
phgt,rhgt,hgt_reg = np.zeros((37, 144)),np.zeros((37, 144)),np.zeros((37, 144))
for i in range(37):
       for j in range(144):
              hgt_reg[i, j], _, rhgt[i, j], phgt[i, j], _ = linregress(teli.iloc[:, 6], hgt_detrend[:, i, j])

# latrange = f_z.lat
#---目标场：1949-2018年的夏季---#
z_sum = f_z.hgt.sel(level=p, lat=latrange).loc[(f_z.time.dt.month.isin([6, 7, 8]))].loc['1950-01-01':'2019-12-01'].groupby('time.year').mean(dim='time', skipna=True) * units('m')
u_sum = f_u.uwnd.sel(level=p, lat=latrange).loc[(f_u.time.dt.month.isin([6, 7, 8]))].loc['1950-01-01':'2019-12-01'].groupby('time.year').mean(dim='time', skipna=True) * units('m/s')
v_sum = f_v.vwnd.sel(level=p, lat=latrange).loc[(f_u.time.dt.month.isin([6, 7, 8]))].loc['1950-01-01':'2019-12-01'].groupby('time.year').mean(dim='time', skipna=True) * units('m/s')
# 回归结果变成ndarray
# hgt_reg 是去趋势过后的回归结果， z_reg是未去趋势过后的回归结果
lon, lat = z_sum.lon, z_sum.lat
hgt_reg = xr.DataArray(hgt_reg, coords=[lat, lon], dims=['lat', 'lon'])
hgt_reg = hgt_reg * units('m')
# 位势高度转位势
Φ_sum = mpcalc.height_to_geopotential(z_sum) # [70, 37, 144]
Φ_sum = Φ_sum.mean(dim='year') # [37, 144]
# Φ_reg_sum = mpcalc.height_to_geopotential(hgt_reg) # [37, 144] #
# Φ_reg_sum = Φ_reg_sum * 48648
# # 目标场uv
# u_sum = u_sum.mean(dim='year')
# v_sum = v_sum.mean(dim='year')

# 背景场：1949-2018年的全年
z_year = f_z.hgt.sel(level=p, lat=latrange).loc['1950-01-01':'2019-12-01'].groupby('time.year').mean(dim='time', skipna=True) * units('m')
u_year = f_u.uwnd.sel(level=p, lat=latrange).loc['1950-01-01':'2019-12-01'].groupby('time.year').mean(dim='time', skipna=True) * units('m/s')
v_year = f_v.vwnd.sel(level=p, lat=latrange).loc['1950-01-01':'2019-12-01'].groupby('time.year').mean(dim='time', skipna=True) * units('m/s')
# 与遥相关指数回归 z_reg_sum [37, 144]
pz, rz, pu, ru, pv, rv = np.zeros((37, 144)), np.zeros((37, 144)), np.zeros((37, 144)), np.zeros((37, 144)), np.zeros((37, 144)), np.zeros((37, 144))
z_reg_year, u_reg_year, v_reg_year = np.zeros((37, 144)), np.zeros((37, 144)), np.zeros((37, 144))
for i in range(37):
       for j in range(144):
              z_reg_year[i, j], _, rz[i, j], pz[i, j], _ = linregress(teli.iloc[:, 6], z_year[:, i, j]) # teli[:, 6]为BSI_std
              u_reg_year[i, j], _, ru[i, j], pu[i, j], _ = linregress(teli.iloc[:, 6], u_year[:, i, j]) # teli[:, 6]为BSI_std
              v_reg_year[i, j], _, rv[i, j], pv[i, j], _ = linregress(teli.iloc[:, 6], v_year[:, i, j]) # teli[:, 6]为BSI_std
lon, lat = z_year.lon, z_year.lat
z_reg_year = xr.DataArray(z_reg_year, coords=[lat, lon], dims=['lat', 'lon'])
u_reg_year = xr.DataArray(u_reg_year, coords=[lat, lon], dims=['lat', 'lon'])
v_reg_year = xr.DataArray(v_reg_year, coords=[lat, lon], dims=['lat', 'lon'])
hgt_reg = hgt_reg * units('m')
z_reg_year = z_reg_year * units('m')
u_reg_year = u_reg_year * units('m/s')
v_reg_year = v_reg_year * units('m/s')

# 位势高度转位势
Φ_year = mpcalc.height_to_geopotential(z_reg_year)
# Φ_year = Φ_year.mean(dim='time') # [37, 144]

# 背景场：Φ_year，u_year，v_year。 回归目标场：Φ_reg_sum，u_reg_sum，v_reg_sum。 目标场：Φ_sum，u_sum，v_sum


#---一些准备工作---#
# 经纬度转为弧度制
lon_deg = Φ_year['lon'].copy()
lat_deg = Φ_year['lat'].copy()
lon_rad = np.deg2rad(lon_deg) * units('1')
lat_rad = np.deg2rad(lat_deg) * units('1')
# 科氏参数
f = mpcalc.coriolis_parameter(Φ_sum['lat'])
# 目标月和目标气压基本流场的气候态
wind_year = mpcalc.wind_speed(u_reg_year, v_reg_year)
cosφ = np.cos(lat_rad)
# 位势的纬向偏差 !!!!这里用的是SUM
Φ_prime = Φ_sum - Φ_year.mean(dim='lon')#.reshape(-1,1)
# 将需要对弧度制纬度求偏导的量的坐标都换成弧度制经纬度
Φ_prime = Φ_prime.assign_coords({'lon': lon_rad, 'lat': lat_rad})
f = f.assign_coords({'lat': lat_rad})
cosφ = cosφ.assign_coords({'lat': lat_rad})
u_climatic = u_reg_year.assign_coords({'lon': lon_rad, 'lat': lat_rad})
v_climatic = v_reg_year.assign_coords({'lon': lon_rad, 'lat': lat_rad})
wind_climatic = wind_year.assign_coords({'lon': lon_rad, 'lat': lat_rad})
# 准地转流函数相对于气候场的扰动
Ψ_prime = Φ_prime / f

#---偏导过程---#
dΨ_prime_dλ = Ψ_prime.differentiate('lon')
dΨ_prime_dφ = Ψ_prime.differentiate('lat')
ddΨ_prime_ddλ = dΨ_prime_dλ.differentiate('lon')
ddΨ_prime_ddφ = dΨ_prime_dφ.differentiate('lat')
ddΨ_prime_dλφ = dΨ_prime_dλ.differentiate('lat')


#---计算部分---#
# T-N波作用通量的水平分量公共部分
temp1 = p / p0 * cosφ / (2 * wind_climatic * earth_avg_radius**2)
temp2 = dΨ_prime_dλ * dΨ_prime_dφ - Ψ_prime * ddΨ_prime_dλφ
# T-N波作用通量的水平分量
fx = temp1 * (u_climatic / cosφ**2 * (dΨ_prime_dλ**2 - Ψ_prime * ddΨ_prime_ddλ) + v_climatic / cosφ * temp2)
fy = temp1 * (u_climatic / cosφ * temp2 + v_climatic * (dΨ_prime_dφ**2 - Ψ_prime * ddΨ_prime_ddφ))
# 把弧度制经纬度再换成角度制便于画图
lon = lon_deg.values
lon[lon>180] -= 360
fx = fx.assign_coords({'lon': lon, 'lat': lat_deg}).sortby(['lon', 'lat'])
fy = fy.assign_coords({'lon': lon, 'lat': lat_deg}).sortby(['lon', 'lat'])
# 转换为array数组
# lon = np.array(lon)
# lat = np.array(lat)
fx = np.array(fx)
fy = np.array(fy)
wind_climatic = np.array(wind_climatic)


#---绘图部分---#
fig = plt.figure(figsize=(5,5), dpi=600)
lat = hgt.lat
lon = hgt.lon
# fig.subplots_adjust(hspace=0.3) # 子图间距
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
# ax.set_title('reg(H-' + TEL_name[(nrow)] + ',GHT)')
# ax.set_title(abc[nrow], loc='left')
# 刻度形式
ax.set_xticks([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180])  # 指定要显示的经纬度
ax.set_yticks([0, 30, 60, 90])
ax.xaxis.set_major_formatter(LongitudeFormatter())  # 刻度格式转换为经纬度样式
ax.yaxis.set_major_formatter(LatitudeFormatter())
# 调整图的内容：投影方式，海岸线
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4)  # 添加海岸线
ax_cf1 = ax.contourf(lon, lat, hgt_reg, transform=ccrs.PlateCarree(), zorder=0, cmap='coolwarm',
                             levels=np.arange(-16, 18, 2), extend='both')
ax.quiver(lon[::2], lat[::2], fx[::2,::2], fy[::2,::2], wind_climatic[::2,::2], transform=ccrs.PlateCarree(), scale=100000, cmap=plt.cm.jet)
# axs2[nrow].contourf(clons, lat, P[nrow, :, :], levels=[0, 0.05, 1], hatches=['...', None], zorder=1, colors="none",
#                     transform=ccrs.PlateCarree())
fig.subplots_adjust(bottom=0.1)
position = fig.add_axes([0.15, 0.05, 0.7, 0.015])  # 位置[左,下,右,上]
cb = fig.colorbar(ax_cf1, shrink=0.6, cax=position, orientation='horizontal', extend='both')
plt.show()
