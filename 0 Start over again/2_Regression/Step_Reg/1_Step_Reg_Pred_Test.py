import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal # 用于去线性趋势
import sys
sys.path.append(r'/0_python_bidirectional_stepwise_selection-master/BidirectionalStep')
import BidirectionalStepwiseSelection as ss # 写的多元线性回归的函数
from sklearn.metrics import mean_squared_error, r2_score # 用于评价模型的函数
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point #进行循环
import cartopy.feature as cfeature #用于添加地理属性的库
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter #添加经纬度需要
from Mask_Ocean_Land import mask_land

# --- Read - Data --- #
# ----------T2M
f1 = xr.open_dataset(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA5_2MT_1959-2022_JJA_25.nc')
T2M = f1.t2m.isel(expver=0).groupby('time.year').mean(dim='time', skipna=True)
lat = T2M.latitude
lon = T2M.longitude
T2M = T2M[20:, :, :] # 切片出1979-20222
T2M_CL = T2M.loc[1971:2000].mean('year')
# --- 提取线性分量 --- #
T2M_ANO = np.zeros((44, 91, 360))
T2M_detrend = scipy.signal.detrend(T2M, axis=0, type='linear', overwrite_data=False)
T2M_year_mean = T2M.mean('year')
for iyear in range(44):
    T2M_detrend[iyear, :, :] = T2M_detrend[iyear, :, :] + T2M_year_mean[:, :]
    T2M_ANO[iyear, :, :] = T2M_detrend[iyear, :, :] - T2M_CL[:, :]
T2M_Recon_21,T2M_Recon_22 = np.zeros((91, 360)), np.zeros((91, 360))
# --- 提取线性分量 --- #
T2M_linear = T2M - T2M_detrend
T2M_linear_2021 = T2M_linear[-2, :, :]
T2M_linear_2022 = T2M_linear[-1, :, :]

# ---------- TEL-Index
TELIS_H = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\Tel_H_I.csv',
                      usecols=['A1A2', 'A3A4', 'B1B2', 'C1C2', 'C3C4', 'D1D2', 'E1E2', 'F1F2', 'G1G2', 'H1H2'])
TELIS_W = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\Tel_W_I.csv',
                      usecols=['A1A2', 'A3A4', 'B1B2', 'B3B4', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
TELI = pd.concat([TELIS_H, TELIS_W], axis=1)
col = ['H_A1A2', 'H_A3A4', 'H_B1B2', 'H_C1C2', 'H_C3C4', 'H_D1D2', 'H_E1E2', 'H_F1F2', 'H_G1G2', 'H_H1H2', 'W_A1A2', 'W_A3A4', 'W_B1B2', 'W_B3B4', 'W_C1C2', 'W_D1D2', 'W_E1E2',
       'W_F1F2']
TELI.columns = col
Factor = TELI.copy(deep=True)
Factor['intercept'] = 1
factors_2022 = Factor.iloc[42, :]

# Train - Model
ilat = 30
ilon = 30
TELIS = TELI.copy(deep=True)
TELIS['y'] = T2M_ANO[:, ilat, ilon]
InPut_T = TELIS.iloc[:43, :]
X = InPut_T.drop(columns="y")
y = InPut_T.y
# 准备训练逐步拟合回归模型
final_vars, iterations_logs, reg_model = ss.BidirectionalStepwiseSelection(X, y, model_type="linear",
                                                                           elimination_criteria="aic", senter=0.05,
                                                                           sstay=0.05)
yFit = reg_model.fittedvalues


# 2022年的预测影子
# factors_2022 = TELI.iloc[-1, :]
# factors_2022['intercept'] = [1]
predict = 0
for i in range(len(final_vars)):
    pre_mid = reg_model.params[final_vars[i]] * factors_2022[final_vars[i]]
    predict = predict + pre_mid
    print(i, predict)

# for i in len(final_vars):

# ax = np.array(TELI.iloc[0, 0:3])
# T2M2022 = reg_model.predict(ax)
# T2M_2021, T2M_2022 = yFit[42], yFit[43]
