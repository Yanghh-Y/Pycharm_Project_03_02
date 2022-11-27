import xarray as xr
import pandas as pd
import numpy as np

Point_df = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\R_P\Cir_H_Point.csv')
lat = np.arange(90, 22.5, -2.5)
lon = np.arange(0, 360, 2.5)
Point_DF = pd.DataFrame(columns=['Cor', 'A_lat', 'A_lon', 'B_lat', 'B_lon'])
Point_DF['Cor'] = Point_df['Min_cor']
for i in range(len(Point_df['Min_cor'])):
    Point_DF.loc[i, 'A_lat'] = lat[Point_df.loc[i, 'A_lat']]
    Point_DF.loc[i, 'A_lon'] = lon[Point_df.loc[i, 'A_lon']]
    Point_DF.loc[i, 'B_lat'] = lat[int(Point_df.loc[i, 'B_lat'])]
    Point_DF.loc[i, 'B_lon'] = lon[int(Point_df.loc[i, 'B_lon'])]

# B1B2
Data = [[-0.76560, 72.5, 292.5, 65, 260]]
B1B2 = pd.DataFrame(Data, columns=['Cor', 'A_lat', 'A_lon', 'B_lat', 'B_lon'])

# B1B2
Data = [[-0.75364, 77.5, 50, 77.5, 85]]
C1C2 = pd.DataFrame(Data, columns=['Cor', 'A_lat', 'A_lon', 'B_lat', 'B_lon'])

#D1D2
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 65, 80, 45, 75, 0, 35, 180, 250
D1D2 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
D1D2.reset_index(drop=True, inplace=True)

#E1E2
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 40, 60, 30, 50, 115, 140, 60, 100
E1E2 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
E1E2.reset_index(drop=True, inplace=True)

#F1F2
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 40, 60, 40, 60, 315, 340, 0, 20
F1F2 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
F1F2.reset_index(drop=True, inplace=True)



H_Teleconnection = pd.DataFrame(columns = ['Name', 'Cor', 'Point1_lon', 'Point1_lat', 'Point2_lon', 'Point2_lat'])
Name = ['B1B2', 'C1C2', 'D1D2', 'E1E2', 'F1F2']
Tel = [B1B2, C1C2, D1D2, E1E2, F1F2]
for i in range(5):
    H_Teleconnection.loc[i,'Name'] = Name[i]
    df = Tel[i]
    H_Teleconnection.loc[i,'Cor'] = df.loc[0,'Cor']
    H_Teleconnection.loc[i,'Point1_lon'] = df.loc[0,'A_lon']
    H_Teleconnection.loc[i,'Point1_lat'] = df.loc[0,'A_lat']
    H_Teleconnection.loc[i,'Point2_lat'] = df.loc[0,'B_lat']
    H_Teleconnection.loc[i,'Point2_lon'] = df.loc[0,'B_lon']

H_Teleconnection.to_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\H_Teleconnection.csv', )




