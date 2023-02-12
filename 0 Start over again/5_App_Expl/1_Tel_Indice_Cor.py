import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import scipy



# -------- Read Data -------- #

# --- TEL --- #
TELIS = pd.read_csv(r'Z:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\Index\Tel_Index.csv', usecols=['H_A1A2', 'H_B1B2', 'H_C1C2', 'H_D1D2', 'H_E1E2', 'H_F1F2', 'H_H1H2', 'W_A1A2', 'W_B1B2', 'W_B3B4', 'W_C1C2', 'W_D1D2','W_E1E2', 'W_F1F2'])
Tel_cl = TELIS.columns

# --- Atm Ocean --- #
Atm_Ind = pd.read_csv(r'Z:\6_Scientific_Research\3_Teleconnection\1_Data\Index\M_Atm_Nc.txt', sep='\s+')
Atm_Tel = Atm_Ind.iloc[:, 67:78]
Atm_cl = Atm_Tel.columns
Atm_list = []
for i in range(863):
    if i % 12 == 5:
        # print(Atm_Ind.iloc[i, :].name)
        abc = (Atm_Tel.iloc[i, :]+Atm_Tel.iloc[i+1, :]+Atm_Tel.iloc[i+2, :])/3
        Atm_list.append(abc)
Atm_arr = np.array(Atm_list)
Atm = pd.DataFrame(Atm_arr[28:, :], columns=Atm_cl)

# Oce_Ind = pd.read_csv(r'Z:\6_Scientific_Research\3_Teleconnection\1_Data\Index\M_Oce_Er.txt', sep='\s+')
# Oce_cl = Oce_Ind.columns
# Oce_list = []
# for i in range(863):
#     if i % 12 == 5:
#         print(Oce_Ind.iloc[i, :].name)
#         abc = (Oce_Ind.iloc[i, :] + Oce_Ind.iloc[i+1, :] + Oce_Ind.iloc[i+2, :])/3
#         Oce_list.append(abc)
# Oce_arr = np.array(Oce_list)
# Oce = pd.DataFrame(Oce_arr[28:, :], columns=Oce_cl)

# --- Cor --- #

rA = np.zeros((14, 11))
tA = np.zeros((14, 11))
for i in range (14):
    for j in range(11):
        r1, t1 = pearsonr(Atm.iloc[:, j], TELIS.iloc[:, i])
        if (abs(r1) >= 0) and (t1 <= 0.05):
            rA[i, j] = r1
            print('Tel : ',Tel_cl[i],' Atm : ',Atm_cl[j], r1)
pd.DataFrame(rA).to_csv(r'Z:\6_Scientific_Research\3_Teleconnection\0 Start over again\2_File\Atmos_Index_Cor.csv')
# rO = np.zeros((14, 26))
# tB = np.zeros((14, 26))
# for i in range (14):
#     for j in range(26):
#         r2, t2 = pearsonr(Oce.iloc[:, j], TELIS.iloc[:, i])
#         if (abs(r2) >= 0) and (t2 <=0.01):
#             rO[i, j] = r2
#             print('Tel : ',Tel_cl[i],' Oce : ',Oce_cl[j], r2)
#
#             # print(Tel_cl[i], Oce)
