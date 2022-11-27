import cdsapi

c = cdsapi.Client()
# 不适用IDM的时候的下载
# c.retrieve(
#     'reanalysis-era5-pressure-levels-monthly-means',
#     {
#         'product_type': 'monthly_averaged_reanalysis',
#         'variable': [
#             'temperature', 'u_component_of_wind', 'v_component_of_wind',
#         ],
#         'pressure_level': [
#             '500', '1000',
#         ],
#         'year': [
#             '1979', '1980', '1981',
#             '1982', '1983', '1984',
#             '1985', '1986', '1987',
#             '1988', '1989', '1990',
#             '1991', '1992', '1993',
#             '1994', '1995', '1996',
#             '1997', '1998', '1999',
#             '2000', '2001', '2002',
#             '2003', '2004', '2005',
#             '2006', '2007', '2008',
#             '2009', '2010', '2011',
#             '2012', '2013', '2014',
#             '2015', '2016', '2017',
#             '2018', '2019', '2020',
#             '2021', '2022',
#         ],
#         'month': [
#             '06', '07', '08',
#         ],
#         'time': '00:00',
#         'area': [90, -180, 0,180,],# North, West, South, East. Default: global
#
#         'grid': [1,1], # Resolution
#         'format': 'netcdf',
#     },
#     'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\1_TEM_GHT_1979-2022_JJA_NH_1-1.nc')

# 使用IDM下载器
from subprocess import call
def idmDownloader(task_url, folder_path, file_name):
    """
    IDM下载器
    :param task_url: 下载任务地址
    :param folder_path: 存放文件夹
    :param file_name: 文件名
    :return:
    """
    # IDM安装目录
    idm_engine = "C:\\Program Files (x86)\\Internet Download Manager\\IDMan.exe"
    # 将任务添加至队列
    call([idm_engine, '/d', task_url, '/p', folder_path, '/f', file_name, '/a'])
    # 开始任务队列
    call([idm_engine, '/s'])

# 数据信息字典
c = cdsapi.Client()
dic = {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            'temperature', 'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': [
            '500', '1000',
        ],
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021', '2022',
        ],
        'month': [
            '06', '07', '08',
        ],
        'time': '00:00',
        'area': [90, -180, 0,180,],# North, West, South, East. Default: global

        'grid': [1,1], # Resolution
        'format': 'netcdf',
    },


r = c.retrieve('reanalysis-era5-pressure-levels-monthly-means', dic, )  # 文件下载器
url = r.location  # 获取文件下载地址
path = 'F:\\6_Scientific_Research\\3_Teleconnection\\1_Data\\ERA5'  # 存放文件夹
filename =  '1_TEM_GHT_1979-2022_JJA_NH_1-1.nc' # 文件名
idmDownloader(url, path, filename)  # 添加进IDM中下载
