import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def aeronet():
    laegern08 = pd.read_csv(r'../data_aeronet/20240801_20240831_Laegeren.txt', delimiter=',', skiprows=6)
    laegern09 = pd.read_csv(r'../data_aeronet/20240901_20240930_Laegeren.txt', delimiter=',', skiprows=6)
    laegern10 = pd.read_csv(r'../data_aeronet/20241001_20241031_Laegeren.txt', delimiter=',', skiprows=6)
    data_aeronet = pd.concat([laegern08, laegern09, laegern10], axis=0, ignore_index=True)
    
    data_aeronet['Date(dd:mm:yyyy)'] = pd.to_datetime(data_aeronet['Date(dd:mm:yyyy)'], format='%d:%m:%Y')
    data_aeronet = data_aeronet.replace(-9999, np.nan)
    
    wavelengths_aer = ['AOD_340nm', 'AOD_380nm','AOD_440nm', 'AOD_500nm','AOD_675nm','AOD_500nm']
    for wavelength in wavelengths_aer:
        plt.plot(data_aeronet['Day_of_Year'], data_aeronet[wavelength], label=wavelength)
    
    
    plt.scatter(data_aeronet['Day_of_Year'], np.full((data_aeronet.shape[0],), 0.02), label='Datapoints')
    plt.legend()
    plt.xlabel('Day of year')
    plt.ylabel('AOD')
    plt.title('Aeronet laegern')
    plt.show()
    return data_aeronet


def satelite():
    
    modisAOD = pd.read_csv(r'../data_satelite/g4.areaAvgTimeSeries.MOD08_D3_6_1_Aerosol_Optical_Depth_Land_Ocean_Mean.20240601-20241201.7E_47N_8E_48N.csv', skiprows=7)
    modis_deep_blue = pd.read_csv(r'../data_satelite/g4.areaAvgTimeSeries.MOD08_D3_6_1_Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean.20240601-20241201.7E_47N_8E_48N.csv', skiprows=7)
    AOD_463 = pd.read_csv(r'../data_satelite/g4.areaAvgTimeSeries.OMAEROe_003_AbsorbingAerosolOpticalThicknessMW_463_0.20240601-20241201.7E_47N_8E_48N.csv', skiprows=7)
    
    modisAOD = modisAOD.replace(-9999, np.nan)
    modis_deep_blue = modis_deep_blue.replace(-9999,np.nan)
    AOD_463 = AOD_463.replace(-32767, np.nan)
    data_satelite = pd.concat([modisAOD, modis_deep_blue.drop(columns='time'),AOD_463.drop(columns='time')], axis=1)
    data_satelite['time'] = pd.to_datetime(data_satelite['time'], format='%Y-%m-%d')
    data_satelite.columns = ['time', 'Modis AOD', 'Modis deep blue', 'OMAE']
    
    instruments = ['Modis AOD', 'Modis deep blue', 'OMAE']
    for instrument in instruments:
        plt.plot(data_satelite['time'].dt.dayofyear, data_satelite[instrument], label=instrument )
    
    plt.legend()
    plt.title('Satellite data')
    plt.xlabel('Day of year')
    plt.ylabel('AOD')
    plt.show()
    return data_satelite

def calitoo(filepath):
    #data_calitoo_ours = pd.read_csv(r'../Daten calitoo/0124_20240604_075512_10_ours_adjusted.txt', delimiter=';', skiprows=6)
    data_calitoo_ours = pd.read_csv(filepath, delimiter=';', skiprows=6)

    data_calitoo_ours['Date'] = pd.to_datetime(data_calitoo_ours['Date'], format='%Y-%m-%d')
    
    wavelengths_cal = ['AOT465', 'AOT540', 'AOT619']
    for wavelength in wavelengths_cal:
        plt.plot(data_calitoo_ours['Date'].dt.dayofyear, data_calitoo_ours[wavelength].where(data_calitoo_ours[wavelength]<0.5), label=wavelength)
    plt.scatter(data_calitoo_ours['Date'].dt.dayofyear, np.full((data_calitoo_ours.shape[0],), 0.02), label='Datapoints cal')
    plt.legend()
    plt.xlabel('Day of year')
    plt.ylabel('AOD')
    plt.title('Aeronet laegern')
    plt.ylim([0,1])
    plt.show()
    return data_calitoo_ours

data_satelite = satelite()
data_aeronet = aeronet()
data_calitoo_ours = calitoo(r'../Daten calitoo/0124_20240604_075512_10_ours_adjusted.txt')



wavelengths_aer = ['AOD_440nm', 'AOD_500nm'] # 'AOD_340nm', 'AOD_380nm',,'AOD_675nm','AOD_500nm'
wavelengths_cal = ['AOT465', 'AOT540', 'AOT619']
instruments = ['Modis deep blue'] #, 'OMAE' 'Modis AOD',



for wavelength in wavelengths_aer:
    plt.plot(data_aeronet['Day_of_Year'], data_aeronet[wavelength], label=f'{wavelength} Aeronet')
plt.scatter(data_aeronet['Day_of_Year'], np.full((data_aeronet.shape[0],), 0.02), label='Datapoints earo')

for instrument in instruments:
    plt.plot(data_satelite['time'].dt.dayofyear, data_satelite[instrument], label=instrument )


for wavelength in wavelengths_cal:
    plt.plot(data_calitoo_ours['Date'].dt.dayofyear, data_calitoo_ours[wavelength].where(data_calitoo_ours[wavelength]<0.5),  label=f'{wavelength} Calitoo')
plt.scatter(data_calitoo_ours['Date'].dt.dayofyear, np.full((data_calitoo_ours.shape[0],), 0.02), label='Datapoints cal')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Day of year')
plt.ylabel('AOD')
plt.title('Combined data')
plt.ylim([0,1])
plt.xlim([200,300])
plt.show()

