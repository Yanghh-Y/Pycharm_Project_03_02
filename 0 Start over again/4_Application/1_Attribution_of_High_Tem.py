import xarray as xr
import pandas as pd
import numpy as np
import scipy
from scipy import signal # 用于去线性趋势
import BidirectionalStepwiseSelection as ss # 写的多元线性回归的函数
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch # 定义图例

# -------- READ DATA --------- #
f1 = xr.open_dataset(r'Z:\6_Scientific_Research\3_Teleconnection\1_Data\ERA5\ERA5_T_U_V_25.nc')
SAT = f1.t.sel(level=1000).groupby('time.year').mean(dim='time', skipna=True)
SAT_CL = SAT.loc[1971:2000].mean('year') # 气候态平均
# ---- Calculating - SAT_ANO ---- #
SAT_ANO = np.zeros((44, 37, 144))
SAT_detrend = scipy.signal.detrend(SAT, axis=0, type='linear', overwrite_data=False)
SAT_year_mean = SAT.mean('year')
for iyear in range(44):
    SAT_detrend[iyear, :, :] = SAT_detrend[iyear, :, :] + SAT_year_mean[:, :]
    SAT_ANO[iyear, :, :] = SAT_detrend[iyear, :, :] - SAT_CL[:, :]
SAT_Recon_21,SAT_Recon_22 = np.zeros((37, 144)), np.zeros((37, 144))
# ---- 提取线性分量 ---- #
SAT_linear = SAT - SAT_detrend
SAT_linear_2021 = SAT_linear[-2, :, :]
SAT_linear_2022 = SAT_linear[-1, :, :]
# ---- region ---- #
lat = SAT.latitude
lon = SAT.longitude
year = range(1979, 2023)
SAT_ANO = xr.DataArray(SAT_ANO, coords=[year, lat, lon], dims=['year', 'lat', 'lon'])
SAT_linear_2021 = xr.DataArray(SAT_linear_2021, coords=[lat, lon], dims=['lat', 'lon'])
SAT_linear_2022 = xr.DataArray(SAT_linear_2022, coords=[lat, lon], dims=['lat', 'lon'])
# -- 2021 -- #
# 11
lat_range_11 = lat[(lat>=37.5) & (lat<=(37.5+12.5))]
lon_range_11 = lon[(lon>=-125) & (lon<=(-125+27.5))]
SAT_ANO_11 = SAT_ANO.sel(lat=lat_range_11, lon=lon_range_11).mean(dim='latitude').mean(dim='longitude')
SAT_linear_11 = SAT_linear_2021.sel(lat=lat_range_11, lon=lon_range_11).mean(dim='latitude').mean(dim='longitude')
# 12
lat_range_12 = lat[(lat>=47.5) & (lat<=(47.5+15))]
lon_range_12 = lon[(lon>=30) & (lon<=(30+30))]
SAT_ANO_12 = SAT_ANO.sel(lat=lat_range_12, lon=lon_range_12).mean(dim='latitude').mean(dim='longitude')
SAT_linear_12 = SAT_linear_2021.sel(lat=lat_range_12, lon=lon_range_12).mean(dim='latitude').mean(dim='longitude')
# 13
lat_range_13 = lat[(lat>=60) & (lat<=(60+12.5))]
lon_range_13 = lon[(lon>=105) & (lon<=(105+40))]
SAT_ANO_13 = SAT_ANO.sel(lat=lat_range_13, lon=lon_range_13).mean(dim='latitude').mean(dim='longitude')
SAT_linear_13 = SAT_linear_2021.sel(lat=lat_range_13, lon=lon_range_13).mean(dim='latitude').mean(dim='longitude')
# -- 2022 -- #
# 21
lat_range_21 = lat[(lat>=62.5) & (lat<=(62.5+10))]
lon_range_21 = lon[(lon>=-115) & (lon<=(-115+30))]
SAT_ANO_21 = SAT_ANO.sel(lat=lat_range_21, lon=lon_range_21).mean(dim='latitude').mean(dim='longitude')
SAT_linear_21 = SAT_linear_2022.sel(lat=lat_range_21, lon=lon_range_21).mean(dim='latitude').mean(dim='longitude')
# 22
lat_range_22 = lat[(lat>=30) & (lat<=(30+17.5))]
lon_range_22 = lon[(lon>=-7.5) & (lon<=(-7.5+27.5))]
SAT_ANO_22 = SAT_ANO.sel(lat=lat_range_22, lon=lon_range_22).mean(dim='latitude').mean(dim='longitude')
SAT_linear_22 = SAT_linear_2022.sel(lat=lat_range_22, lon=lon_range_22).mean(dim='latitude').mean(dim='longitude')
# 23
lat_range_23 = lat[(lat>=20) & (lat<=(20+15))]
lon_range_23 = lon[(lon>=100) & (lon<=(100+20))]
SAT_ANO_23 = SAT_ANO.sel(lat=lat_range_23, lon=lon_range_23).mean(dim='latitude').mean(dim='longitude')
SAT_linear_23 = SAT_linear_2022.sel(lat=lat_range_23, lon=lon_range_23).mean(dim='latitude').mean(dim='longitude')
# list
Ylist = [SAT_ANO_11, SAT_ANO_12, SAT_ANO_13, SAT_ANO_21, SAT_ANO_22, SAT_ANO_23]



# -------- Train Model -------- #
# ---- All - TEL - Index ---- #
TELIS_H = pd.read_csv(r'Z:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\Tel_H_I.csv',
                      usecols=['A1A2', 'A3A4', 'B1B2', 'C1C2', 'C3C4', 'D1D2', 'E1E2', 'F1F2', 'G1G2', 'H1H2'])
TELIS_W = pd.read_csv(r'Z:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\Tel_W_I.csv',
                      usecols=['A1A2', 'A3A4', 'B1B2', 'B3B4', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
TELI = pd.concat([TELIS_H, TELIS_W], axis=1)
col = ['H_A1A2', 'H_A3A4', 'H_B1B2', 'H_C1C2', 'H_C3C4', 'H_D1D2', 'H_E1E2', 'H_F1F2', 'H_G1G2', 'H_H1H2', 'W_A1A2', 'W_A3A4', 'W_B1B2', 'W_B3B4', 'W_C1C2', 'W_D1D2', 'W_E1E2',
       'W_F1F2']
TELI.columns = col
TELI = TELI.drop(columns=['H_A3A4', 'H_C3C4', 'H_G1G2', 'W_A3A4'])
# ---- Select - Tel - Index ---- #
TELI2021 = TELI[['H_C1C2', 'H_H1H2', 'W_A1A2', 'W_B3B4', 'W_D1D2']]
TELI2022 = TELI[['H_A1A2', 'H_C1C2', 'H_D1D2', 'H_F1F2', 'W_A1A2', 'W_B1B2', 'W_B3B4', 'W_C1C2', 'W_D1D2']]
factors_2021 = TELI.iloc[-2, :]
factors_2022 = TELI.iloc[-1, :]
# ---- Train model ---- #
All_01_final_vars_list, All_01_iterations_logs_list, All_01_reg_model_list = [], [], []
All_05_final_vars_list, All_05_iterations_logs_list, All_05_reg_model_list = [], [], []
Sel_01_final_vars_list, Sel_01_iterations_logs_list, Sel_01_reg_model_list = [], [], []
Sel_05_final_vars_list, Sel_05_iterations_logs_list, Sel_05_reg_model_list = [], [], []
All_Fit_01 ,All_Fit_05 = [],[]
# -- Model -- #
for i in range(6):
    X_All = TELI.copy(deep=True)
    if i < 3 :
        X_All_2021 = X_All.iloc[0:43, :]
        X_Sel_2021 = TELI2021.iloc[0:43, :]
        Y_2021 = Ylist[i][0:43]
        All_01_final_vars, All_01_iterations_logs, All_01_reg_model = ss.BidirectionalStepwiseSelection(X_All_2021, Y_2021,model_type="linear",elimination_criteria="aic",senter=0.1, sstay=0.1)
        All_05_final_vars, All_05_iterations_logs, All_05_reg_model = ss.BidirectionalStepwiseSelection(X_All_2021, Y_2021,model_type="linear",elimination_criteria="aic",senter=0.5, sstay=0.5)
        Sel_01_final_vars, Sel_01_iterations_logs, Sel_01_reg_model = ss.BidirectionalStepwiseSelection(X_Sel_2021, Y_2021,model_type="linear",elimination_criteria="aic",senter=0.1, sstay=0.1)
        Sel_05_final_vars, Sel_05_iterations_logs, Sel_05_reg_model = ss.BidirectionalStepwiseSelection(X_Sel_2021, Y_2021,model_type="linear",elimination_criteria="aic",senter=0.5, sstay=0.5)
        All_01_yFit = All_01_reg_model.fittedvalues[42]
        All_05_yFit = All_05_reg_model.fittedvalues[42]


        All_01_final_vars_list.append(All_01_final_vars)
        All_05_final_vars_list.append(All_05_final_vars)
        Sel_01_final_vars_list.append(Sel_01_final_vars)
        Sel_05_final_vars_list.append(Sel_05_final_vars)
        All_01_iterations_logs_list.append(All_01_iterations_logs)
        All_05_iterations_logs_list.append(All_05_iterations_logs)
        Sel_01_iterations_logs_list.append(Sel_01_iterations_logs)
        Sel_05_iterations_logs_list.append(Sel_05_iterations_logs)
        All_01_reg_model_list.append(All_01_reg_model.params)
        All_05_reg_model_list.append(All_05_reg_model.params)
        Sel_01_reg_model_list.append(Sel_01_reg_model.params)
        Sel_05_reg_model_list.append(Sel_05_reg_model.params)
        All_Fit_01.append(All_01_yFit)
        All_Fit_05.append(All_05_yFit)

    if i >= 3:
        X_All_2022 = X_All.iloc[:, :]
        X_Sel_2022 = TELI2022.iloc[:, :]
        Y_2022 = Ylist[i]
        All_01_final_vars, All_01_iterations_logs, All_01_reg_model = ss.BidirectionalStepwiseSelection(X_All_2022, Y_2022,model_type="linear",elimination_criteria="aic",senter=0.1, sstay=0.1)
        All_05_final_vars, All_05_iterations_logs, All_05_reg_model = ss.BidirectionalStepwiseSelection(X_All_2022, Y_2022,model_type="linear",elimination_criteria="aic",senter=0.5, sstay=0.5)
        Sel_01_final_vars, Sel_01_iterations_logs, Sel_01_reg_model = ss.BidirectionalStepwiseSelection(X_Sel_2022, Y_2022,model_type="linear",elimination_criteria="aic",senter=0.1, sstay=0.1)
        Sel_05_final_vars, Sel_05_iterations_logs, Sel_05_reg_model = ss.BidirectionalStepwiseSelection(X_Sel_2022, Y_2022,model_type="linear",elimination_criteria="aic",senter=0.5, sstay=0.5)
        All_01_yFit = All_01_reg_model.fittedvalues[43]
        All_05_yFit = All_05_reg_model.fittedvalues[43]

        All_01_final_vars_list.append(All_01_final_vars)
        All_05_final_vars_list.append(All_05_final_vars)
        Sel_01_final_vars_list.append(Sel_01_final_vars)
        Sel_05_final_vars_list.append(Sel_05_final_vars)
        All_01_iterations_logs_list.append(All_01_iterations_logs)
        All_05_iterations_logs_list.append(All_05_iterations_logs)
        Sel_01_iterations_logs_list.append(Sel_01_iterations_logs)
        Sel_05_iterations_logs_list.append(Sel_05_iterations_logs)
        All_01_reg_model_list.append(All_01_reg_model.params)
        All_05_reg_model_list.append(All_05_reg_model.params)
        Sel_01_reg_model_list.append(Sel_01_reg_model.params)
        Sel_05_reg_model_list.append(Sel_05_reg_model.params)
        All_Fit_01.append(All_01_yFit)
        All_Fit_05.append(All_05_yFit)

# -------- Attribution --------- #
Region_Tel_01, Region_Tel_05 ,Attribution_01, Attribution_05= [], [], [], []
SAT_L_Name = ['SAT_linear_11', 'SAT_linear_12', 'SAT_linear_13', 'SAT_linear_21', 'SAT_linear_22', 'SAT_linear_23']
SAT_L = [SAT_linear_11, SAT_linear_12, SAT_linear_13, SAT_linear_21, SAT_linear_22, SAT_linear_23]
Resd_Name = ['R_11', 'R_12', 'R_13', 'R_21', 'R_22', 'R_23']
TELI['intercept'] = 1
for i in range(6):
    # 每个区域的遥相关名称
    Index_05 = list(All_05_reg_model_list[i].index.values)
    Index_05.append(SAT_L_Name[i])
    Index_05.append(Resd_Name[i])
    Index_01 = list(All_01_reg_model_list[i].index.values)
    Index_01.append(SAT_L_Name[i])
    Index_01.append(Resd_Name[i])
    Region_Tel_01.append(Index_01)
    Region_Tel_05.append(Index_05)


    # 05每个区域的遥相关贡献量
    Attri_region = []
    for n in range(len(All_05_reg_model_list[i].index)):
        if i < 3 :
            Attribution = All_05_reg_model_list[i][n] * TELI[All_05_reg_model_list[i].index[n]][42]
            Attri_region.append(Attribution)
        if i > 2 :
            Attribution = All_05_reg_model_list[i][n] * TELI[All_05_reg_model_list[i].index[n]][43]
            Attri_region.append(Attribution)
    # 线性分量
    Attri_region.append(float(SAT_L[i]))
    # 残差量
    if i < 3 :
        res = Ylist[i][42] - All_Fit_05[i]
        Attri_region.append(float(res))
    if i > 2:
        res = Ylist[i][43] - All_Fit_05[i]
        Attri_region.append(float(res))
    Attribution_05.append(Attri_region)

    # 01每个区域的遥相关贡献量
    Attri_region = []
    for n in range(len(All_01_reg_model_list[i].index)):
        if i < 3 :
            Attribution = All_01_reg_model_list[i][n] * TELI[All_01_reg_model_list[i].index[n]][42]
            Attri_region.append(Attribution)
        if i > 2 :
            Attribution = All_01_reg_model_list[i][n] * TELI[All_01_reg_model_list[i].index[n]][43]
            Attri_region.append(Attribution)
    # 线性分量
    Attri_region.append(float(SAT_L[i]))
    # 残差量
    if i < 3 :
        res = Ylist[i][42] - All_Fit_01[i]
        Attri_region.append(float(res))
    if i > 2:
        res = Ylist[i][43] - All_Fit_01[i]
        Attri_region.append(float(res))
    # 将每个区域的各个TEl的贡献存成list在放进list
    Attribution_01.append(Attri_region)

Tel_01 = pd.DataFrame(Region_Tel_01)
Tel_05 = pd.DataFrame(Region_Tel_05)
Attr_01 = pd.DataFrame(Attribution_01)
Attr_05 = pd.DataFrame(Attribution_05)


# -------- 2021 -------- #
fig1,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(6,3), dpi=450, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
ax1.spines['right'].set_linewidth(0) # 框线宽度为0，则在视觉效果上为无框线
ax1.spines['right'].set_color('None') # 框线无颜色，在在视觉效果上为无框
ax2.spines['right'].set_linewidth(0)
ax2.spines['right'].set_color('None')
ax2.spines['left'].set_linewidth(0)
ax2.spines['left'].set_color('None')
ax3.spines['left'].set_linewidth(0)
ax3.spines['left'].set_color('None')
ax3.yaxis.set_major_locator(mticker.NullLocator())
ax2.yaxis.set_major_locator(mticker.NullLocator())
colordict = {'W_E1E2':'salmon', 'W_F1F2':'maroon', 'W_C1C2':'sienna', 'W_D1D2':'peru', \
    'H_B1B2':'burlywood','H_C1C2':'skyblue', 'H_F1F2':'greenyellow', 'H_H1H2':'tan', 'H_D1D2':'gold', 'H_E1E2':'cyan', \
    'R_11':'grey', 'R_12':'grey', 'R_13':'grey', 'R_21':'grey', 'R_22':'grey', 'R_23':'grey'}
colordf = pd.DataFrame(colordict, index=[0])
# ax1
#                H_F1F2-beige
# W_E1E2-salmon, H_B1B2-burlywood, R_11-grey
bary11_up_n = [Tel_05[2][0], Tel_05[1][0], Tel_05[2][0]]
bary11_up = [0, Attr_05[1][0], 0]
bary11_n = [Tel_05[0][0], Tel_05[2][0], Tel_05[4][0]]
bary11 = [Attr_05[0][0], Attr_05[2][0], Attr_05[4][0]]
color11_up_n = list(colordf[bary11_up_n].values[0])
color11_n = list(colordf[bary11_n].values[0])
ax1.set_ylim(0, 2)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
# ax1.set_xticklabels(['W', 'H', 'Resi'])
ax1.bar(['W', 'H', 'Resi'], bary11, width=1, color=color11_n)
ax1.bar(['W', 'H', 'Resi'], bary11_up, bottom=bary11, width=1, color=color11_up_n)

# ax2
#        H_D1D2
# W_D1D2 H_H1H2 R_12
bary12_up_n = [Tel_05[2][0], Tel_05[2][1], Tel_05[2][0]]
bary12_up = [0, Attr_05[2][1], 0]
bary12_n = [Tel_05[3][1], Tel_05[1][1], Tel_05[5][1]]
bary12 = [Attr_05[3][1], Attr_05[1][1], Attr_05[5][1]]
color12_up_n = list(colordf[bary12_up_n].values[0])
color12_n = list(colordf[bary12_n].values[0])
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
# ax1.set_xticklabels(['W', 'H', 'Resi'])
ax2.bar(['W', 'H', 'Resi'], bary12, width=1, color=color12_n)
ax2.bar(['W', 'H', 'Resi'], bary12_up, bottom=bary12, width=1, color=color12_up_n)

# ax3
# W_E1E2
# W_F1F2 H_D1D2 R_13
bary13_up_n = [Tel_05[2][2], Tel_05[2][2], Tel_05[2][2]]
bary13_up = [Attr_05[2][2], 0, 0]
bary13_n = [Tel_05[0][2], Tel_05[1][2], Tel_05[4][2]]
bary13 = [Attr_05[0][2], abs(Attr_05[1][2]), Attr_05[4][2]]
color13_up_n = list(colordf[bary13_up_n].values[0])
color13_n = list(colordf[bary13_n].values[0])
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
# ax1.set_xticklabels(['W', 'H', 'Resi'])
ax3.bar(['W', 'H', 'Resi'], bary13, width=1, color=color13_n)
ax3.bar(['W', 'H', 'Resi'], bary13_up, bottom=bary13, width=1, color=color13_up_n)

# legend
colordict = {'W_E1E2':'salmon', 'W_F1F2':'maroon', 'W_C1C2':'sienna', 'W_D1D2':'peru', \
    'H_B1B2':'burlywood','H_C1C2':'skyblue', 'H_F1F2':'greenyellow', 'H_H1H2':'tan', 'H_D1D2':'gold', 'H_E1E2':'cyan', \
    'Resi':'grey'}
legend_elements = []
for key in colordict:
    legend_elements.append(Patch(facecolor=colordict[key], edgecolor=None, label=str(key)))
ax3.legend(handles=legend_elements, loc=1, ncol=3, fontsize='x-small')
fig1.savefig(r'Z:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\4_Application\2021_005.png')
plt.show()








# -------- 2022 -------- #
fig1,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(6,3), dpi=450, sharey=True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
ax1.spines['right'].set_linewidth(0) # 框线宽度为0，则在视觉效果上为无框线
ax1.spines['right'].set_color('None') # 框线无颜色，在在视觉效果上为无框
ax2.spines['right'].set_linewidth(0)
ax2.spines['right'].set_color('None')
ax2.spines['left'].set_linewidth(0)
ax2.spines['left'].set_color('None')
ax3.spines['left'].set_linewidth(0)
ax3.spines['left'].set_color('None')
ax3.yaxis.set_major_locator(mticker.NullLocator())
ax2.yaxis.set_major_locator(mticker.NullLocator())
colordict = {'W_E1E2':'salmon', 'W_F1F2':'maroon', 'W_C1C2':'sienna', 'W_D1D2':'peru', \
    'H_B1B2':'burlywood','H_C1C2':'skyblue', 'H_F1F2':'greenyellow', 'H_H1H2':'tan', 'H_D1D2':'gold', 'H_E1E2':'cyan', \
    'R_11':'grey', 'R_12':'grey', 'R_13':'grey', 'R_21':'grey', 'R_22':'grey', 'R_23':'grey'}
colordf = pd.DataFrame(colordict, index=[0])
# ax1
# W_E1E2
# W_F1F2, H_H1H2
# W_D1D2, H_F1F2, R_21
bary21_upup_n = [Tel_05[3][3], Tel_05[1][0], Tel_05[2][0]]
bary21_upup = [abs(Attr_05[3][3]), 0, 0]
bary21_up_n = [Tel_05[2][3], Tel_05[5][3], Tel_05[2][0]]
bary21_up = [Attr_05[2][3], abs(Attr_05[5][3]), 0]
bary21_n = [Tel_05[4][3], Tel_05[1][3], Tel_05[7][3]]
bary21 = [Attr_05[4][3], Attr_05[1][3], Attr_05[7][3]]
color21_upup_n = list(colordf[bary21_upup_n].values[0])
color21_up_n = list(colordf[bary21_up_n].values[0])
color21_n = list(colordf[bary21_n].values[0])
ax1.set_ylim(0, 2.0)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
# ax1.set_xticklabels(['W', 'H', 'Resi'])
ax1.bar(['W', 'H', 'Resi'], bary21, width=1, color=color21_n)
ax1.bar(['W', 'H', 'Resi'], bary21_up, bottom=bary21, width=1, color=color21_up_n)
ax1.bar(['W', 'H', 'Resi'], bary21_upup, bottom=(np.array(bary21_up)+np.array(bary21)), width=1, color=color21_upup_n)

# ax2    H_H1H2
#        H_E1E2
# W_C1C2 H_C1C2 R_22
bary22_upup_n = [Tel_05[2][0], Tel_05[2][4], Tel_05[2][0]]
bary22_upup = [0, abs(Attr_05[2][4]), 0]
bary22_up_n = [Tel_05[2][0], Tel_05[4][4], Tel_05[2][0]]
bary22_up = [0, abs(Attr_05[4][4]), 0]
bary22_n = [Tel_05[1][4], Tel_05[3][4], Tel_05[6][4]]
bary22 = [Attr_05[1][4], Attr_05[3][4], Attr_05[6][4]]
color22_upup_n = list(colordf[bary22_upup_n].values[0])
color22_up_n = list(colordf[bary22_up_n].values[0])
color22_n = list(colordf[bary22_n].values[0])
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
# ax1.set_xticklabels(['W', 'H', 'Resi'])
ax2.bar(['W', 'H', 'Resi'], bary22, width=1, color=color22_n)
ax2.bar(['W', 'H', 'Resi'], bary22_up, bottom=bary22, width=1, color=color22_up_n)
ax2.bar(['W', 'H', 'Resi'], bary22_upup, bottom=(np.array(bary22_up)+np.array(bary22)), width=1, color=color22_upup_n)

# ax3
#        H_D1D2
# W_C1C2 H_F1F2 R_33
bary23_up_n = [Tel_05[2][0], Tel_05[0][5], Tel_05[2][0]]
bary23_up = [0, abs(Attr_05[0][5]), 0]

bary23_n = [Tel_05[2][5], Tel_05[1][5], Tel_05[4][5]]
bary23 = [Attr_05[2][5], Attr_05[1][5], Attr_05[4][5]]

color23_up_n = list(colordf[bary22_up_n].values[0])
color23_n = list(colordf[bary22_n].values[0])

ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
# ax1.set_xticklabels(['W', 'H', 'Resi'])
ax3.bar(['W', 'H', 'Resi'], bary23, width=1, color=color23_n)
ax3.bar(['W', 'H', 'Resi'], bary23_up, bottom=bary23, width=1, color=color23_up_n)
# legend
colordict = {'W_E1E2':'salmon', 'W_F1F2':'maroon', 'W_C1C2':'sienna', 'W_D1D2':'peru', \
    'H_B1B2':'burlywood','H_C1C2':'skyblue', 'H_F1F2':'greenyellow', 'H_H1H2':'tan', 'H_D1D2':'gold', 'H_E1E2':'cyan', \
    'Resi':'grey'}
legend_elements = []
for key in colordict:
    legend_elements.append(Patch(facecolor=colordict[key], edgecolor=None, label=str(key)))
ax3.legend(handles=legend_elements, loc=1, ncol=3, fontsize='x-small')
plt.show()


fig1.savefig(r'Z:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\4_Application\2022_005.png')

plt.show()


