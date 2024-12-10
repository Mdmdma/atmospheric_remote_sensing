# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:44:37 2024

@author: Nathalie
"""

#%% Import libraries
import pandas as pd
import re
from io import StringIO
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.stats import linregress
from geopy.geocoders import Nominatim
#%%
def plot_daily_aot(df, calitoo):
    # Reset index if needed
    df.reset_index(inplace=True)
    
    # Get unique days
    unique_days = df['Datetime'].dt.date.unique()

    # Calculate grid dimensions
    num_days = len(unique_days)
    ncols = 4
    nrows = (num_days + ncols - 1) // ncols

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25), sharex=False)
    axes = axes.flatten()

    # Plot data for each unique day
    for i, day in enumerate(unique_days):
        # Filter the DataFrame for the current day
        day_data = df[df['Datetime'].dt.date == day]
        day_data = day_data.sort_values(by='Time')
        
        # Plot each signal for the current day
        axes[i].plot(day_data['Time'], day_data['AOT465'], label='RAW 465', color='blue')
        axes[i].plot(day_data['Time'], day_data['AOT540'], label='RAW 540', color='green')
        axes[i].plot(day_data['Time'], day_data['AOT619'], label='RAW 619', color='red')
        
        # Set title and labels
        axes[i].set_title(f'{day}', fontsize=20)
        axes[i].set_xlabel('Time', fontsize=18)
        axes[i].set_ylabel('AOT', fontsize=18)
        axes[i].grid(True)
        
        # Format the x-axis for time
        axes[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[i].tick_params(labelsize=16)

    # Hide empty subplots if there are any
    for i in range(num_days, len(axes)):
        axes[i].axis('off')

    # Adjust layout and show plot
    plt.title(f"Measurements of Calitoo Nr. {calitoo}")
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def extract_variables(lines):
    variables = {}
    pattern = re.compile(r'(\w+_\d+)=([\d.,]+)')
    for line in lines:
        matches = pattern.findall(line)
        for key, value in matches:
            # Convert values to float, replacing commas with dots
            variables[key] = float(value.replace(',', '.'))
    return variables

def read_and_clean_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract header lines and data lines
    header_lines = []
    data_lines = []
    data_start = False
    
    for line in lines:
        if not data_start:
            header_lines.append(line)
        if 'Date' in line and 'Time' in line:
            data_start = True
        if data_start:
            data_lines.append(line)
    
    # Extract variables from header
    variables = extract_variables(header_lines)
    
    # Convert data lines to a single string for reading into a DataFrame
    data_str = ''.join(data_lines)
    
    # Use StringIO to read the data into a pandas DataFrame
    data = StringIO(data_str)
    df = pd.read_csv(data, delimiter=';', decimal=',')
    
    
    return variables, df

def ddm_to_dd(ddm_coord):
    """
    Converts Degrees and Decimal Minutes (DDM) to Decimal Degrees (DD),
    handling cases where longitude has leading zeros.
    
    Args:
        ddm_coord (str): Coordinate in DDM format (e.g., "00830.36148E").
    
    Returns:
        float: Coordinate in Decimal Degrees.
    """
    # Extract hemisphere (last character)
    hemisphere = ddm_coord[-1]
    
    # Remove hemisphere from the coordinate
    ddm_coord = ddm_coord[:-1]
    
    # Determine the number of degrees 
    if ddm_coord[0] == '0':  # For longitude with leading zero
        degrees = int(ddm_coord[:3])  # First 3 characters are degrees
        minutes = float(ddm_coord[3:])  # Remaining are minutes
    else:  # For latitude or longitude without leading zero
        degrees = int(ddm_coord[:2])  # First 2 characters are degrees
        minutes = float(ddm_coord[2:])  # Remaining are minutes
    
    # Convert to Decimal Degrees
    decimal_degrees = degrees + (minutes / 60)
    
    # Adjust for hemisphere
    if hemisphere in ['S', 'W']:
        decimal_degrees *= -1
    
    return decimal_degrees

def plot_langley(data, voltage_column, color, label):
    # Calculate natural logarithm of voltage
    data['ln_' + voltage_column] = np.log(data[voltage_column])

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(data['air_mass'], data['ln_' + voltage_column])

    # Plot the data points
    plt.scatter(data['air_mass'], data['ln_' + voltage_column], color=color)

    # Generate x values from 0 to 2.3 for the regression line
    x_reg = np.linspace(0, 2.3, 100)
    y_reg = intercept + slope * x_reg

    # Plot the fitted line from x=0 to x=2.3
    plt.plot(x_reg, y_reg, color=color, linestyle='--', label=label)

    # Extract the calibration constant
    calibration_constant = np.exp(intercept)
    print(f'Calibration Constant for {label}: {calibration_constant}')

    return calibration_constant

def is_valid_time(time_str):
    try:
        pd.to_datetime(time_str, format='%H:%M:%S')
        return True
    except ValueError:
        return False

def update_data(data):
    data['Alpha'] = pd.to_numeric(data['Alpha'].str.replace(',', '.'), errors='coerce')
    data['RÂ²'] = pd.to_numeric(data['RÂ²'].str.replace(',', '.'), errors='coerce')
    
    # Drop rows where either 'Alpha' or 'RÂ²' is NaN
    data = data.dropna(subset=['Alpha', 'RÂ²'])
    
    data['Elevation'] = pd.to_numeric(data['Elevation'], errors='coerce')
    
    # Drop rows where 'elevation' is NaN
    data = data.dropna(subset=['Elevation'])
    
    data = data[data['Time'].apply(is_valid_time)]
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    
    data['N'] = data['Latitude'].apply(ddm_to_dd)
    data['E'] = data['Longitude'].apply(ddm_to_dd)
    
    #data['Location'] = data.apply(lambda row: get_location(row['N'], row['E']), axis=1)
    
    data['sza_rad'] = np.deg2rad(90-data['Elevation'])
    data['air_mass'] = 1 / np.cos(data['sza_rad'])
    
    return data

def get_location(lat, lon):
    geolocator = Nominatim(user_agent="my_agent")
    location = geolocator.reverse(f"{lat}, {lon}")
    return location.address if location else "Location not found"

def plot_aod(data):
    # Create the figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    # Plot tau_aerosols_465
    ax3.plot(data['Datetime'], data['tau_aerosols_465'], color='blue')
    ax3.set_ylabel('tau_aerosols_465')
    ax3.set_title('Aerosol Optical Depth at 465 nm')
    
    # Plot tau_aerosols_540
    ax1.plot(data['Datetime'], data['tau_aerosols_540'], color='green')
    ax1.set_ylabel('tau_aerosols_540')
    ax1.set_title('Aerosol Optical Depth at 540 nm')

    # Plot tau_aerosols_619
    ax2.plot(data['Datetime'], data['tau_aerosols_619'], color='red')
    ax2.set_ylabel('tau_aerosols_619')
    ax2.set_title('Aerosol Optical Depth at 619 nm')
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()
    
def day_of_year(date):
    return date.timetuple().tm_yday
    
def es_distance_correction(doy):
    d0 = 1
    d = d0 * (1 - 0.01672 * np.cos(2 * np.pi * doy / 365))
    E0 = (d0 / d) ** 2
    return d, E0
#%% Load and Merge Datasets
# Define the file path
calitoo = '427'
if (calitoo=='422'):
    file_paths = [r"C:\Users\Nathalie\Documents\GitHub\atmospheric_remote_sensing\daten calitoo\0124_20240604_075512_10_ours.txt"]
elif (calitoo=='425'):
    file_paths = [r"C:\Users\Nathalie\Documents\GitHub\atmospheric_remote_sensing\daten calitoo\0425_20240604_082717_10.txt"]
elif (calitoo=='427'):
    file_paths =[r"C:\Users\Nathalie\Documents\GitHub\atmospheric_remote_sensing\daten calitoo\0427_20240604_082717_10.txt"]# Replace with your actual file path

preprocessed_data = []
# Extract variables and clean data
for file_path in file_paths:
    variables, dataset = read_and_clean_data(file_path)
    preprocessed_data.append(dataset)
   
data = pd.concat(preprocessed_data, ignore_index=True)

# Print extracted variables
print("Extracted Variables:")
for key, value in variables.items():
    print(f'{key}: {value}')
#%% Preprocessing
data = update_data(data)
#%%
data['day_of_year'] = pd.to_datetime(data['Date']).dt.dayofyear
data['d'], data['E0'] = zip(*data['day_of_year'].apply(es_distance_correction))

data = data.drop('d', axis=1)
#%%
data['RAW465'] = data['RAW465']/data['E0']
data['RAW540'] = data['RAW540']/data['E0']
data['RAW619'] = data['RAW619']/data['E0']
#%%
data.to_csv(rf"C:\Users\Nathalie\Documents\GitHub\atmospheric_remote_sensing\daten calitoo\calibration_interm_results\preprocessed_data_all_no_location_calitoo_{calitoo}.csv")

#%%Outlier Removal
data = data[(data['Alpha']<2) & (data['RÂ²']>0.90)]
#%% Write preprocessed data
data.to_csv(rf"C:\Users\Nathalie\Documents\GitHub\atmospheric_remote_sensing\daten calitoo\calibration_interm_results\preprocessed_data_no_outliers_no_location_calitoo_{calitoo}.csv")

#%% Read preprocessed data
data = pd.read_csv(rf"C:\Users\Nathalie\Documents\GitHub\atmospheric_remote_sensing\daten calitoo\calibration_interm_results\preprocessed_data_all_no_location_calitoo_{calitoo}.csv")
data_calibration= pd.read_csv(rf"C:\Users\Nathalie\Documents\GitHub\atmospheric_remote_sensing\daten calitoo\calibration_interm_results\preprocessed_data_no_outliers_no_location_calitoo_{calitoo}.csv")
#%% Plot Measurements for each day
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
data_calibration['Datetime'] = pd.to_datetime(data_calibration['Datetime'], errors='coerce')
#%%
#data = data[pd.to_datetime(data['Datetime'], errors='coerce').notna()]
plot_daily_aot(data_calibration, calitoo)

#%%
#Here select dates for calibration
if (calitoo=='422'):
    date_list = ['2024-07-05', '2024-06-04', '2024-07-15', '2024-07-11', '2024-07-09']
elif (calitoo=='425'):
    date_list=['2024-07-09', '2024-07-10', '2024-07-11', '2024-07-15', '2024-08-23']
else:
    date_list=['2024-07-09', '2024-07-10', '2024-07-15', '2024-08-06', '2024-08-23']


#%% Langley Plot and Calibration Constant without Outlier Removal
calibration_values_blue= []
calibration_values_green= []
calibration_values_red= []
for day in date_list:
    data_days = data_calibration[data_calibration['Date']==day]
    plt.figure(figsize=(10, 6))
    
    # Plot for RAW465
    calibration_constant_465 = plot_langley(data_days, 'RAW465', 'blue', '465nm')
    calibration_values_blue.append(calibration_constant_465)
    # Plot for RAW540
    calibration_constant_540 = plot_langley(data_days, 'RAW540', 'green', '540nm')
    calibration_values_green.append(calibration_constant_540)
    # Plot for RAW619
    calibration_constant_619= plot_langley(data_days, 'RAW619', 'red', '619nm')
    calibration_values_red.append(calibration_constant_619)
    # Customize and show the plot
    plt.xlabel('Air Mass')
    plt.ylabel('log(V)')
    plt.title(f'Calitoo {calitoo}\nLangley Plot for Blue (465), Green (540) and Red (619)\nDate:{day}')
    legend = plt.legend(title='λ')
    
    plt.xlim([0, 2.3])
    plt.setp(legend.get_title(), fontsize='13', fontweight='bold') 
    plt.show()
    
calibration_constant_465 = np.mean(calibration_values_blue)
calibration_constant_540 = np.mean(calibration_values_green)
calibration_constant_619 = np.mean(calibration_values_red)
#%%
if (calitoo=='422'):
    expected_cc_465 = 3490
    expected_cc_540 = 3551
    expected_cc_619 = 2568
elif (calitoo=='425'):
    expected_cc_465 = 3400
    expected_cc_540 = 3364
    expected_cc_619 = 2570
elif (calitoo=='427'):
    expected_cc_465 = 3413
    expected_cc_540 = 3375
    expected_cc_619 = 2589
else:
    pass

diff_465 = round((abs(expected_cc_465-calibration_constant_465)/expected_cc_465)*100, 1)
diff_540 = round((abs(expected_cc_540-calibration_constant_540)/expected_cc_540)*100, 1)
diff_619 = round((abs(expected_cc_619-calibration_constant_619)/expected_cc_619)*100, 1)

print(f"Calitoo {calitoo}: Deviation between expected calibration value and retrieved calibration value for 465nm: {diff_465}%")
print(f"Calitoo {calitoo}: Deviation between expected calibration value and retrieved calibration value for 540nm: {diff_540}%")
print(f"Calitoo {calitoo}: Deviation between expected calibration value and retrieved calibration value for 619nm: {diff_619}%")
#%%
#calibration_constant, cleaned_data = plot_langley_with_outlier_removal(data_day, 'RAW465', color='blue', label='465nm', threshold=2.0)
#%% Calculate AOD
data['precalibrated_AOT465'] = data['AOT465']
data['precalibrated_AOT540'] = data['AOT540']
data['precalibrated_AOT619'] = data['AOT619']

data['AOT465'] = ((np.log(calibration_constant_465) - np.log(data['RAW465'])) / data['air_mass']) - variables['RAY_465']
data['AOT540'] = ((np.log(calibration_constant_540) - np.log(data['RAW540'])) / data['air_mass']) - variables['RAY_540'] - variables['OZ_540']
data['AOT619'] = ((np.log(calibration_constant_619) - np.log(data['RAW619'])) / data['air_mass']) - variables['RAY_619'] - variables['OZ_619']

data['tau_aerosols_465_calitoo'] = ((np.log(variables['CN0_465']) - np.log(data['RAW465'])) / data['air_mass']) - variables['RAY_465']
data['tau_aerosols_540_calitoo'] = ((np.log(variables['CN0_540']) - np.log(data['RAW540'])) / data['air_mass']) - variables['RAY_540'] - variables['OZ_540']
data['tau_aerosols_619_calitoo'] = ((np.log(variables['CN0_619']) - np.log(data['RAW619'])) / data['air_mass']) - variables['RAY_619'] - variables['OZ_619']

data['tau_aerosols_465_expected'] = ((np.log(expected_cc_465) - np.log(data['RAW465'])) / data['air_mass']) - variables['RAY_465']
data['tau_aerosols_540_expected'] = ((np.log(expected_cc_540) - np.log(data['RAW540'])) / data['air_mass']) - variables['RAY_540'] - variables['OZ_540']
data['tau_aerosols_619_expected'] = ((np.log(expected_cc_619) - np.log(data['RAW619'])) / data['air_mass']) - variables['RAY_619'] - variables['OZ_619']
#%%
data.to_csv(rf"calibrated_data\preprocessed_data_with_AOD_all_calitoo_{calitoo}.csv")
#%% Plot AOD over time
#plot_aod(data)
#%% For comparison with precipitation
#data_zurich = data[data['Location'].str.contains('Zürich')]
#%% Plot AOD over time
#plot_aod(data_zurich)