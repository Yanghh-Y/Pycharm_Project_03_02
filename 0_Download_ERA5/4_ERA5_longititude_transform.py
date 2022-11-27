import xarray as xr
import sys
sys.path.append(r'E:\Pycharm\Pycharm_Project_03\0_Download_ERA5\Longitude_Transform')
from Longitude_Transform import translon180_360



#---read z---#
filepath = (r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA-GHT-2.5.nc')
# f1的经度是(-180，177.5)(首尾都包含) 转换成(0, 360)
lon_name = 'longitude'  # whatever name is in the data
f1 = translon180_360(filepath, lon_name)






