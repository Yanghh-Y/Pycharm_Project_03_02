import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.util import add_cyclic_point #进行循环
from matplotlib import patches
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要


#--读取数据--#
# 相关系数以及P值筛选
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\6_1019Problem\R_P\Cir_R_r_all_lon_lat_1.nc' ,engine='netcdf4')
f2 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\6_1019Problem\R_P\Cir_R_p_all_lon_lat_1.nc')
r = f1.r.sel(ilat=np.arange(25, 92.5, 2.5), dlat=np.arange(25, 92.5, 2.5))
p = f2.p.sel(ilat=np.arange(25, 92.5, 2.5), dlat=np.arange(25, 92.5, 2.5))
r_arr = np.array(r)
p_arr = np.array(p)
r_min_arr = np.zeros((27, 144))
p_sel_arr = np.zeros((27,144))
lat_arr = np.zeros((27,144))
lon_arr = np.zeros((27,144))
# 筛选最小的相关系数，同时将每个点对应的坐标存储下来
for ilat in range(27):
    for ilon in range(144):
        r_min_arr[ilat, ilon] = r_arr[ilat, ilon, :, :].min()
        location = np.where(r_arr[ilat, ilon, :, :]==r_min_arr[ilat, ilon]) #取(ilat，ilon)所对应最小的相关系数的那个坐标
        if len(location[0])>1: # 如果不止一个点的最小相关系数是一样的
            print((ilat, ilon), r_min_arr[ilat, ilon])
            for i in range(len(location[0])):
                print(location[0][i], location[1][i])
            lat_arr[ilat, ilon] = location[0][0]
            lon_arr[ilat, ilon] = location[1][0]
            p_sel_arr[ilat, ilon] = p_arr[ilat, ilon, location[0][0], location[1][0]]
        else:
            lat_arr[ilat, ilon] = location[0]
            lon_arr[ilat, ilon] = location[1]
            p_sel_arr[ilat, ilon] = p_arr[ilat, ilon, location[0], location[1]]

# 筛选P值，当显著性<99.99%，我们直接将其相关性定为0
# for ilat in range(27):
#     for ilon in range(144):
#         # if r_min_arr[ilat, ilon] > -0.35 or p_sel_arr[ilat, ilon] > 0.01 :
#         if p_sel_arr[ilat, ilon] > 0.05 : # p值<0.01<=>通过显著性99%的显著性检验
#             r_min_arr[ilat, ilon] = np.nan


# 预留一个计算好的最小相关系数矩阵用于后面筛选其经纬度
R_lat_lon = r_min_arr
# 这里将选择好的数据存为nc文件
lon = np.arange(0, 360, 2.5)
lat = np.arange(90, 22.5, -2.5)
r_min_arr = xr.DataArray(r_min_arr, coords=[lat, lon], dims=['lat', 'lon'])
ds = xr.Dataset({'r_min_r': r_min_arr})
ds.to_netcdf(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\6_1019Problem\R_P\cir_R_r_min_arr.nc')


# ---将最小相关性进行排序---#
# 排序算法：
def bubble_sort(lst):
    n = len(lst)

    A_lat,A_lon,B_lat,B_lon = [], [], [], [] # 将两个点的四个坐标存储下来
    for i in range(n): # 冒泡法排序
        for j in range(1, n - i):
            if lst[j - 1] > lst[j]:
                lst[j - 1], lst[j] = lst[j], lst[j - 1]
    for i in range(len(lst)): # 存储两个点的index
        position = np.where(R_lat_lon == lst[i])
        # A点对应的lat_lon的index
        A_lat.append(position[0][0])
        A_lon.append(position[1][0])
        B_lat.append(lat_arr[A_lat[i], A_lon[i]])
        B_lon.append(lon_arr[A_lat[i], A_lon[i]])
    dic = {'Min_cor': lst,
           'A_lat': A_lat,
           'A_lon': A_lon,
           'B_lat': B_lat,
           'B_lon': B_lon}
    Point_df = pd.DataFrame(dic)
    return Point_df

# 去除nan值
e = list(np.array(r_min_arr).ravel())
e = [a_ for a_ in e if a_ == a_]
# 找出
Point_df = bubble_sort(e)
Point_df.to_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\6_1019Problem\R_P\Cir_R_Point.csv', )

# 数据属性
lat = r_min_arr.lat
lon = r_min_arr.lon
cyclic_r, cyclic_lons = add_cyclic_point(r_min_arr, coord=lon)

#--绘图--#
def drawing(cyclic_lons, lat, cyclic_data):
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())  # 添加子图，设置投影方式，添加中心经度
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    # 设置范围
    leftlon, rightlon, lowerlat, upperlat = (-180, 180, 25, 90)
    img_extent = [leftlon, rightlon, lowerlat, upperlat]
    ax.set_extent(img_extent, ccrs.PlateCarree())  # 通过圆柱投影限制地图范围，便于设置地图参数
    #  设置网格点属性
    theta = np.linspace(0, 2 * np.pi, 120)
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    center, radius = [0.5, 0.5], 0.5
    circle = patches.Path(verts * radius + center)
    # 指定xlocs/ylocs会产生经线/纬线的数量
    dmeridian = 90  # 经线间距
    dparallel = 15  # 纬线间距
    num_merid = int(360 / dmeridian + 1)  # 经线数量
    num_parra = int(65 / dparallel + 1)  # 纬线数量
    # 设置标签以及网格线
    gl = ax.gridlines(crs=ccrs.PlateCarree(), \
                      xlocs=np.linspace(-180, 180, num_merid), \
                      ylocs=np.linspace(25, 90, num_parra), \
                      linestyle="--", linewidth=1, color='k', alpha=0.5)
    # 标签对齐
    va = 'center'  # also bottom, top
    ha = 'center'  # right, left
    degree_symbol = u'\u00B0'
    # 经线标签的位置
    # lond = np.linspace(0, 360, num_merid)
    # latd = 24*np.ones(len(lond))
    # for (alon, alat) in zip(lond, latd):  # zip函数将lond latd打包为元组的列表
    #     projx1, projy1 = ax.projection.transform_point(alon, alat, ccrs.Geodetic())
    #     if alon > 0 and alon < 180:
    #         ha = 'left'
    #         va = 'center'
    #     if alon > 180 and alon < 360:
    #         ha = 'right'
    #         va = 'center'
    #     if np.abs(alon - 180) < 0.01:
    #         ha = 'center'
    #         va = 'bottom'
    #     if alon == 0.:
    #         ha = 'center'
    #         va = 'top'
    #     if (alon < 360.):
    #         txt = ' {0} '.format(str(int(alon))) + degree_symbol
    #     ax.text(projx1, projy1, txt, va=va, ha=ha, color='black', alpha=0.5, fontsize=19)

    # 设置axes边界，为圆形边界，否则为正方形的极地投影
    ax.set_boundary(circle, transform=ax.transAxes)
    # 设置字体为楷体,显示负号
    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号
    # 设置图名和边框
    picture_title = "Teleconnectivity matrix"
    # 添加线段
    for i in range(len(Point_df.index)):
        A_lon = lon[Point_df.loc[i, 'A_lon']]
        A_lat = lat[Point_df.loc[i, 'A_lat']]
        B_lon = lon[int(Point_df.loc[i, 'B_lon'])]
        B_lat = lat[int(Point_df.loc[i, 'B_lat'])]
        # ax.plot([A_lon, B_lon], [A_lat, B_lat], 'k', '.',transform=ccrs.PlateCarree(), linewidth=0.1)
    ax.set_title(picture_title, fontsize=15, pad=25)  # 图名title
    # 填充等值线和添加等值线、colobar
    ax_cf1 = ax.contourf(cyclic_lons, lat, cyclic_data, zorder=0, extend='both', transform=ccrs.PlateCarree(), cmap='hot', levels=np.arange(-0.35, 0.05, 0.05 ))  # 绘制等值线图
    # 添加colorbar
    cb1 = fig.colorbar(ax_cf1, shrink=0.55, orientation='horizontal')# shrin收缩比例
    ax1 = cb1.ax  # 调出colorbar的ax属性，将colorbar视为一个ax对象来详细设置相关属性
    ax1.set_title('Correlation coefficient', fontsize=7)
    ax1.tick_params(which='major', direction='in', labelsize=4, length=7.5)
    # ax1.tick_params(which='minor',direction='in',length=7.5,width=0.3)
    ax1.xaxis.set_minor_locator(mticker.MultipleLocator(2.5))  # 显示x轴副刻度
    # plt.savefig(r'F:\6_Scientific_Research\3_Teleconnection\2_Picture\2_Teleconnection\Cir_R_Teleconnectivity matrix.png')
    plt.show()

drawing(cyclic_lons, lat, cyclic_r)