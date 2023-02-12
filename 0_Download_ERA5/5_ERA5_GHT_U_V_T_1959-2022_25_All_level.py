import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels-monthly-means',
    {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            'geopotential', 'temperature', 'u_component_of_wind',
            'v_component_of_wind',
        ],
        'pressure_level': [
            '200', '250', '300', '500',
            '850', '1000',
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
            # '01', '02', '03',
            # '04', '05', '06',
            # '07', '08', '09',
            # '10', '11', '12',
            '6', '7', '8',
        ],
        'area': [
            90, -180, 0,
            180,
        ],
        'time': '00:00',
        'grid': [2.5, 2.5],  # Resolution
        'format': 'netcdf',
    },
    r'Z:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA5_T_U_V_25_1.nc')
