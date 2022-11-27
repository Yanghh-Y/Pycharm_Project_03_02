import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            'geopotential', 'temperature', 'u_component_of_wind',
            'v_component_of_wind',
        ],
        'year': [
            '1959', '1960', '1961',
            '1962', '1963', '1964',
            '1965', '1966', '1967',
            '1968', '1969', '1970',
            '1971', '1972', '1973',
            '1974', '1975', '1976',
            '1977', '1978', '1979',
            '1980', '1981', '1982',
            '1983', '1984', '1985',
            '1986', '1987', '1988',
            '1989', '1990', '1991',
            '1992', '1993', '1994',
            '1995', '1996', '1997',
            '1998', '1999', '2000',
            '2001', '2002', '2003',
            '2004', '2005', '2006',
            '2007', '2008', '2009',
            '2010', '2011', '2012',
            '2013', '2014', '2015',
            '2016', '2017', '2018',
            '2019', '2020', '2021',
            '2022',
        ],
        'month': [
            '06', '07', '08',
        ],
        'time': '00:00',
        'area': [
            90, -180, 0,
            180,
        ],
        'grid': [1,1], # Resolution
        'format': 'netcdf',
    },
    'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA5_2MT_1979-2022_JJA_25.nc')