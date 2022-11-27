import xarray as xr

def translon180_360(filepath, lon_name):
    ds = xr.open_dataset(filepath)
    # f1的经度是(-180，177.5)(首尾都包含) 转换成(0, 360)
    ds['longitude_adjusted'] = xr.where(ds[lon_name] < 0, ds[lon_name] % 360, ds[lon_name])
    ds = (
        ds
        .swap_dims({lon_name: 'longitude_adjusted'})
        .sel(**{'longitude_adjusted': sorted(ds.longitude_adjusted)})
        .drop(lon_name))
    ds = ds.rename({'longitude_adjusted': lon_name})
    return ds

def translon360_180(filepath, lon_name):
    ds = xr.open_dataset(filepath)
    ds['longitude_adjusted'] = xr.where(ds[lon_name] < 0, ds[lon_name] % 360, ds[lon_name])
    ds = (
        ds
        .swap_dims({lon_name: 'longitude_adjusted'})
        .sel(**{'longitude_adjusted': sorted(ds.longitude_adjusted)})
        .drop(lon_name))
    ds = ds.rename({'longitude_adjusted': lon_name})
    return ds