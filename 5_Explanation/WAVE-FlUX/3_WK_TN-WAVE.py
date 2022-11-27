import numpy as np
import pandas as pd
import xarray as xr

#以下为数据读取部分，最终所得z300，u300，v300为2014年11月的300hPa位势高度场，UV风场的月平均；
#z_c，u_c，v_c为1979-2018年11月的气候态；

f_z300 = xr.open_dataset('E:/DATASETS/NCEP_NCAR/hgt.mon.mean.nc')
z_c = np.mean(f_z300['hgt'][394:754:12,8,:,:],axis=0)

f_u300 = xr.open_dataset('E:/DATASETS/NCEP_NCAR/uwnd.mon.mean.nc')
u_c = np.mean(f_u300['uwnd'][394:754:12,8,:,:],axis=0)

f_v300 = xr.open_dataset('E:/DATASETS/NCEP_NCAR/vwnd.mon.mean.nc')
v_c = np.mean(f_v300['vwnd'][394:754:12,8,:,:],axis=0)

z300 = f_z300.hgt.loc['2014-11-01',250,:,:]

u300 = f_u300.uwnd.loc['2014-11-01',250,:,:]

v300 = f_v300.vwnd.loc['2014-11-01',250,:,:]

lat = f_u300.lat
lon = f_u300.lon

#计算高度距平
za = z300 - z_c

a=6370000            #地球半径
omega=7.292e-5       #自转角速度
p_p0 = 300/1000      #p/p0/lev
g = 9.8

dlon=(np.gradient(lon)*np.pi/180.0).reshape((1,-1))
dlat=(np.gradient(lat)*np.pi/180.0).reshape((-1,1))

coslat = (np.cos(np.array(lat)*np.pi/180)).reshape((-1,1))

#计算科氏力
f=np.array(2*omega*np.sin(lat*np.pi/180.0)).reshape((-1,1))
#计算|U|
wind = np.sqrt(u_c**2+v_c**2)
#计算括号外的参数，a^2可以从括号内提出
lev = 300/1000
c=(lev)*coslat/(2*a*a*wind)

#Ψ`
g = 9.8
streamf = g*za/f
#计算各个部件，难度在于二阶导，变量的名字应该可以很容易看出我是在计算哪部分
dzdlon = np.gradient(streamf,axis = 1)/dlon
ddzdlonlon = np.gradient(dzdlon,axis = 1)/dlon
dzdlat = np.gradient(streamf,axis = 0)/dlat
ddzdlatlat = np.gradient(dzdlat,axis = 0)/dlat
ddzdlatlon = np.gradient(dzdlat,axis = 1)/dlon
#这是X,Y分量共有的部分
x_tmp = dzdlon*dzdlon-streamf*ddzdlonlon
xy_tmp = dzdlon*dzdlat-streamf*ddzdlatlon
y_tmp = dzdlat*dzdlat-streamf*ddzdlatlat
#计算两个分量
fx = c * ((u_c/coslat/coslat)*x_tmp+v_c*xy_tmp/coslat)
fy = c * ((u_c/coslat)*xy_tmp+v_c*y_tmp)