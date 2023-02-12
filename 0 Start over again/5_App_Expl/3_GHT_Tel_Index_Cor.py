import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import scipy



# -------- Read Data -------- #

# --- TEL --- #
TELIS = pd.read_csv(r'Z:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\Index\Tel_Index.csv', usecols=['H_A1A2', 'H_B1B2', 'H_C1C2', 'H_D1D2', 'H_E1E2',                                                                                                                             'H_F1F2', 'H_H1H2', 'W_A1A2', 'W_B1B2', 'W_B3B4', 'W_C1C2', 'W_D1D2','W_E1E2', 'W_F1F2'])
Tel_cl = TELIS.columns

# --- Tel GHT --- #
GHT_Tel = pd.read_csv(r'Z:\6_Scientific_Research\3_Teleconnection\1_Data\Index\HGT_TEL_INDEX_CAL.csv')
GHT_cl = GHT_Tel.columns


# --- Cor --- #

rA = np.zeros((14, 11))
tA = np.zeros((14, 11))
for i in range (14):
    for j in range(10):
        r1, t1 = pearsonr(GHT_Tel.iloc[:, j], TELIS.iloc[:, i])
        if (abs(r1) >= 0) and (t1 <= 0.1):
            rA[i, j] = r1
pd.DataFrame(rA).to_csv(r'Z:\6_Scientific_Research\3_Teleconnection\0 Start over again\2_File\Tel_Index_Cor_01.csv')







