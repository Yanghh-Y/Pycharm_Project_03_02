import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#--读取数据--#
# 读取遥相关指数
# ---------- TEL-Index
TELIS_H = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\H\Tel_H_I.csv',
                      usecols=['A1A2', 'A3A4', 'B1B2', 'C1C2', 'C3C4', 'D1D2', 'E1E2', 'F1F2', 'G1G2', 'H1H2'])
TELIS_W = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\W\Tel_W_I.csv',
                      usecols=['A1A2', 'A3A4', 'B1B2', 'B3B4', 'C1C2', 'D1D2', 'E1E2', 'F1F2'])
TELI = pd.concat([TELIS_H, TELIS_W], axis=1)
col = ['H_A1A2', 'H_A3A4', 'H_B1B2', 'H_C1C2', 'H_C3C4', 'H_D1D2', 'H_E1E2', 'H_F1F2', 'H_G1G2', 'H_H1H2', 'W_A1A2', 'W_A3A4', 'W_B1B2', 'W_B3B4', 'W_C1C2', 'W_D1D2', 'W_E1E2',
       'W_F1F2']
TELI.columns = col
TELI = TELI.drop(columns=['H_A3A4', 'H_C3C4', 'H_G1G2', 'W_A3A4'])
TELI.to_csv(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\0_Data\1_Teleconnection\Index\Tel_Index.csv')
TL_coor = TELI.corr()
fig = plt.figure(figsize=(9,9), dpi=600)
ax = fig.add_subplot(1, 1, 1)# 设置画布大小，分辨率，和底色
plt.rc('font',family='Times New Roman',size=8)
ax = sns.heatmap(TL_coor, annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1
ax.set_title('COR H-TEL & W-TEL', fontsize=14, pad=30)
fig.savefig(r'F:\6_Scientific_Research\3_Teleconnection\0 Start over again\1_Picture\1_Teleconnection\Cor_H-TEL_W-TEL.png')

# plt.show()








DF_Factors = pd.read_csv(r'F:\7_Data\0_Science_Program\0_Graduation_Program\9_Process_Data\Factors.csv', usecols=['North_Atlantic_SST_PWM', 'NINO_C_SST_SSM', 'SoilMoisture_Spr', 'EMI_SSM','PDOI_PYM'])
DF_Factors_coor = DF_Factors.corr()
plt.subplots(figsize=(9,9),dpi=1080,facecolor='w')# 设置画布大小，分辨率，和底色
fig=sns.heatmap(DF_Factors_coor, annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')#annot为热力图上显示数据；fmt='.2g'为数据保留两位有效数字,square呈现正方形，vmax最大值为1
plt.show()