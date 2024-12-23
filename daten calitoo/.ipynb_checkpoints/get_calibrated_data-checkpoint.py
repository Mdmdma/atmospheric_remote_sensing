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
def plot_daily_aot(df):
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

    # Plot the data points and fitted line
    plt.scatter(data['air_mass'], data['ln_' + voltage_column], color=color)
    plt.plot(data['air_mass'], intercept + slope * data['air_mass'], color=color, linestyle='--', label=label)

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
    data = data[data['Time'].apply(is_valid_time)]
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    
    data['N'] = data['Latitude'].apply(ddm_to_dd)
    data['E'] = data['Longitude'].apply(ddm_to_dd)
    
    data['Location'] = data.apply(lambda row: get_location(row['N'], row['E']), axis=1)
    
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
    
#%%
# Define the file path
file_paths = [r"C:\Users\Nathalie\Documents\GitHub\atmospheric_remote_sensing\daten calitoo\0124_20240604_075512_10_ours.txt"]
file_paths = ["../daten calitoo/0425_20240604_082717_10.txt"]
              #, r"C:\Users\Nathalie\Documents\GitHub\atmospheric_remote_sensing\daten calitoo\0427_20240604_082717_10.txt"# Replace with your actual file path

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
#%%
data = update_data(data)
#%%
#data.to_csv(r"../daten calitoo/preprocessed_data.csv")
#data = pd.read_csv(r"C:\Users\Nathalie\Documents\GitHub\atmospheric_remote_sensing\daten calitoo\preprocessed_data.csv")
#%%
data_zurich = data[data['Location'].str.contains('Zürich')]
data_day = data_zurich[data_zurich['Date']=='2024-11-15']
#%%
plot_daily_aot(data)
#%% Langley Plot without Outlier Removal
plt.figure(figsize=(10, 6))

# Plot for RAW465
calibration_constant_465 = plot_langley(data_day, 'RAW465', 'blue', '465nm')

# Plot for RAW540
calibration_constant_540 = plot_langley(data_day, 'RAW540', 'green', '540nm')

# Plot for RAW619
calibration_constant_619= plot_langley(data_day, 'RAW619', 'red', '619nm')

# Customize and show the plot
plt.xlabel('Air Mass')
plt.ylabel('log(V)')
plt.title('Langley Plot for Blue (465), Green (540) and Red (619)\n On 2024-11-15')
legend = plt.legend(title='λ')
plt.setp(legend.get_title(), fontsize='13', fontweight='bold') 
plt.show()

#%%
#calibration_constant, cleaned_data = plot_langley_with_outlier_removal(data_day, 'RAW465', color='blue', label='465nm', threshold=2.0)
#%%
data_zurich['tau_aerosols_465'] = ((np.log(calibration_constant_465) - np.log(data_zurich['RAW465'])) / data_zurich['air_mass']) - variables['RAY_465']
data_zurich['tau_aerosols_540'] = ((np.log(calibration_constant_540) - np.log(data_zurich['RAW540'])) / data_zurich['air_mass']) - variables['RAY_540'] - variables['OZ_540']
data_zurich['tau_aerosols_619'] = ((np.log(calibration_constant_619) - np.log(data_zurich['RAW619'])) / data_zurich['air_mass']) - variables['RAY_619'] - variables['OZ_619']
#%%
plot_aod(data_zurich)