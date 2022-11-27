import xarray as xr
import pandas as pd
import numpy as np

Point_df = pd.read_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\R_P\Cir_W_Point.csv')
lat = np.arange(90, 22.5, -2.5)
lon = np.arange(0, 360, 2.5)
Point_DF = pd.DataFrame(columns=['Cor', 'A_lat', 'A_lon', 'B_lat', 'B_lon'])
Point_DF['Cor'] = Point_df['Min_cor']
for i in range(len(Point_df['Min_cor'])):
    Point_DF.loc[i, 'A_lat'] = lat[Point_df.loc[i, 'A_lat']]
    Point_DF.loc[i, 'A_lon'] = lon[Point_df.loc[i, 'A_lon']]
    Point_DF.loc[i, 'B_lat'] = lat[int(Point_df.loc[i, 'B_lat'])]
    Point_DF.loc[i, 'B_lon'] = lon[int(Point_df.loc[i, 'B_lon'])]

#A1A2
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 80, 90, 27.5, 55, 130, 155, 25, 80
A1A2 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
A1A2.reset_index(drop=True, inplace=True)

#A1A2
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 80, 90, 30, 40, 130, 155, 260, 280
A1A3 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
A1A3.reset_index(drop=True, inplace=True)

#B1B2
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 80, 90, 25, 35, 150, 170, 110, 140
B1B2 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
B1B2.reset_index(drop=True, inplace=True)


#B1B3
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 80, 90, 40, 45, 150, 170, 175, 200
B1B3 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
B1B3.reset_index(drop=True, inplace=True)

#C1C2
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 40, 57.5, 57.5, 72.5, 120, 165, 125, 155
C1C2 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
C1C2.reset_index(drop=True, inplace=True)
#C1C3
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 40, 57.5, 62.5, 75, 120, 165, 155, 210
C1C3 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
C1C3.reset_index(drop=True, inplace=True)

#E1E2
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 40, 55, 65, 80, 325, 355, 310, 340
E1E2 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
E1E2.reset_index(drop=True, inplace=True)

#E1E3
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 40, 55, 60, 80, 325, 355, 340, 360
E1E3_1 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
Alatmin, Alatmax, Blatmin, Blatmax, Alongmin, Alongmax, Blongmin, Blongmax = 40, 55, 60, 80, 325, 355, 0, 20
E1E3_2 = Point_DF[((((Point_DF['A_lat'] >= Alatmin) & (Point_DF['A_lat'] <= Alatmax)) & ((Point_DF['B_lat'] >= Blatmin) & (Point_DF['B_lat'] <= Blatmax)))& \
                (((Point_DF['A_lon'] >= Alongmin) & (Point_DF['A_lon'] <= Alongmax)) & ((Point_DF['B_lon'] >= Blongmin) & (Point_DF['B_lon'] <= Blongmax))))| \
                ((((Point_DF['B_lat'] >= Alatmin) & (Point_DF['B_lat'] <= Alatmax)) & ((Point_DF['A_lat'] >= Blatmin) & (Point_DF['A_lat'] <= Blatmax)))& \
                (((Point_DF['B_lon'] >= Alongmin) & (Point_DF['B_lon'] <= Alongmax)) & ((Point_DF['A_lon'] >= Blongmin) & (Point_DF['A_lon'] <= Blongmax))))]
E1E3_2.reset_index(drop=True, inplace=True)

W_Teleconnection = pd.DataFrame(columns = ['Name', 'Cor', 'Point1_lon', 'Point1_lat', 'Point2_lon', 'Point2_lat'])
Name = ['A1A2', 'A1A3', 'B1B2', 'B1B3', 'C1C2', 'C1C3', 'E1E2', 'E1E3']
Tel = [A1A2, A1A3, B1B2, B1B3, C1C2, C1C3, E1E2, E1E3_2]
for i in range(8):
    W_Teleconnection.loc[i,'Name'] = Name[i]
    df = Tel[i]
    W_Teleconnection.loc[i,'Cor'] = df.loc[0,'Cor']
    W_Teleconnection.loc[i,'Point1_lon'] = df.loc[0,'A_lon']
    W_Teleconnection.loc[i,'Point1_lat'] = df.loc[0,'A_lat']
    W_Teleconnection.loc[i,'Point2_lat'] = df.loc[0,'B_lat']
    W_Teleconnection.loc[i,'Point2_lon'] = df.loc[0,'B_lon']

W_Teleconnection.to_csv(r'F:\6_Scientific_Research\3_Teleconnection\1_Data\Teleconnection\W_Teleconnection.csv', )




