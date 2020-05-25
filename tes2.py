import pandas as pd

Data = pd.read_csv('Apartemen_numerik.csv')
print(Data)

# buat liat mean std min max dll
print(Data.describe())
Data.loc[2, 'Jum_kamar'] = 100

print("\n   ---MIN MAX Normalization---")
print("\n   -------DATA SEBELUM-------")
print(Data)

# metode minmax
Data_minmax = Data.copy()
kol_kamar_minmax = pd.DataFrame(Data_minmax['Jum_kamar'])

for i in range(0, 9):
    kol_kamar_minmax.iloc[i] = (Data_minmax['Jum_kamar'].iloc[i]-1)/(100-1)

Data_minmax['Jum_kamar'] = kol_kamar_minmax
print("\n   -------DATA SESUDAH-------")
print(Data_minmax)

# metode mean
Data_mean = Data.copy()
kol_kamar_mean = pd.DataFrame(Data_mean['Jum_kamar'])

for i in range(0, 9):
    kol_kamar_mean.iloc[i] = (Data_mean['Jum_kamar'].iloc[i]-13.444444)/(100-1)
Data_mean['Jum_kamar'] = kol_kamar_mean

# metode Zscore
Data_Zscore = Data.copy()
kol_kamar_Zscore = pd.DataFrame(Data_Zscore['Jum_kamar'])

for i in range(0, 9):
    kol_kamar_Zscore.iloc[i] = (
        Data_Zscore['Jum_kamar'].iloc[i]-13.444444)/32.465794
Data_Zscore['Jum_kamar'] = kol_kamar_Zscore
print(Data_Zscore['Jum_kamar'])
print(Data_mean['Jum_kamar'])