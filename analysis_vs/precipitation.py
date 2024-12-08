# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:04:04 2024

@author: Valu
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# import data
precipitation = pd.read_csv(r'../data_precipitation/n_zuerich_fluntern.txt', delimiter=',')
calitoo = pd.read_csv(r'../Daten calitoo/0124_20240604_075512_10_ours_adjusted.txt', delimiter=';', skiprows=6)

precipitation['Date'] = pd.to_datetime(precipitation['Date'])
calitoo['Date'] = pd.to_datetime(calitoo['Date'])

calitoo['SZA'] = 90 - calitoo['Elevation']
precipitation['prev_day_p'] = precipitation['p'].shift(1)

calitoo_subset = calitoo[calitoo['Date'].dt.month.isin([9,10,11])]
# basic outlier removal
calitoo_subset = calitoo_subset[calitoo_subset['AOT465'] < 0.5]

plt.figure(figsize=(10, 6))
plt.plot(precipitation['Date'], precipitation['p'], marker='.')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(calitoo_subset['Date'], calitoo_subset['AOT465'], marker='.')
plt.show()

# merge precipitation data with calitoo data
p_cal = pd.merge(precipitation, calitoo_subset, on='Date', how='outer')
#p_cal['prev_day_p'] = p_cal['p'].shift(1)

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

ax1.scatter(p_cal['Date'],p_cal['AOT465'],color='red')
ax2.scatter(p_cal['Date'],p_cal['p'],color='blue')

plt.tight_layout()
plt.show()

## filter data frame
start_date = pd.to_datetime('2024-09-22')
end_date = pd.to_datetime('2024-10-21')
sep_oct = p_cal[(p_cal['Date'] >= start_date) & (p_cal['Date'] <= end_date)]

start_date = pd.to_datetime('2024-11-05')
end_date = pd.to_datetime('2024-11-27')
nov = p_cal[(p_cal['Date'] >= start_date) & (p_cal['Date'] <= end_date)]


fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(10,6))

ax1.scatter(sep_oct['Date'], sep_oct['AOT465'],color='red', marker='x')
ax2.bar(sep_oct['Date'], sep_oct['p'],color='blue')

ax1.set_ylabel('AOT at 465 nm')
ax2.set_ylabel('precipitation [mm]')
ax2.set_xlabel('date')

plt.tight_layout()
ax1.grid(True)
ax2.grid(True)
plt.show()

##
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(10,6))

ax1.scatter(nov['Date'], nov['AOT465'],color='red',marker='x')
ax2.bar(nov['Date'], nov['p'],color='blue')

ax1.set_ylabel('AOT at 465 nm')
ax2.set_ylabel('precipitation [mm]')
ax2.set_xlabel('date')

plt.tight_layout()
ax1.grid(True)
ax2.grid(True)
plt.show()

## correlation
p_cal_clean = p_cal.dropna(subset=['AOT465'])
p_cal_clean = p_cal_clean[p_cal_clean['AOT465'] != 0]

plt.figure(figsize=(10, 6))
scatter = plt.scatter(p_cal_clean['p'],p_cal_clean['AOT465'],c=p_cal_clean['SZA'],cmap='Purples')
# Add color bar with a legend label
colorbar = plt.colorbar(scatter)  # Associate color bar with scatter plot
colorbar.set_label('SZA [°]')  # Label for the color bar
plt.ylabel('AOT465')
plt.xlabel('precipitation [mm]')

plt.grid(True)
plt.show()
# cor coefficient
from scipy.stats import spearmanr

cor_coeff, p_value = spearmanr(p_cal_clean['p'],p_cal_clean['AOT465'])
 # cor_coeff = -0.178, p_value = 0.08
 
## precipitation shift one day
plt.figure(figsize=(10, 6))
scatter = plt.scatter(p_cal_clean['prev_day_p'],p_cal_clean['AOT465'],c=p_cal_clean['SZA'],cmap='Purples')
# Add color bar with a legend label
colorbar = plt.colorbar(scatter)  # Associate color bar with scatter plot
colorbar.set_label('SZA [°]')  # Label for the color bar
plt.ylabel('AOT465')
plt.xlabel('precipitation of the day before [mm]')

plt.grid(True)
plt.show()
# cor coefficient
from scipy.stats import spearmanr

cor_coeff_prev, p_value_prev = spearmanr(p_cal_clean['prev_day_p'],p_cal_clean['AOT465']) 

## plot I vs SZ and color depending on location
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.cm as cm

## function to convert DDM to DD coordinates
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

# convert coordinates in df
calitoo_subset['N'] = calitoo_subset['Latitude'].apply(ddm_to_dd)
calitoo_subset['E'] = calitoo_subset['Longitude'].apply(ddm_to_dd)

# cluster points on location
coordinates = calitoo_subset[['N','E']].to_numpy()
epsilon = 1/6371 # convert 1 km to radians for DBSCAN
db = DBSCAN(eps=epsilon, min_samples=1, metric='haversine').fit(np.radians(coordinates))

calitoo_subset['Cluster'] = db.labels_

##
# Get unique clusters
clusters = calitoo_subset['Cluster'].unique()
cluster_labels = {
    0: 'ETH Zentrum', 
    1: 'Oerlikon', 
    2: 'ETH Höngg', 
    3: 'UZH Irchel', 
    4: 'Seebach', 
    5: 'Leimbach'
}

# Generate a colormap with distinct colors for each cluster
colormap = cm.get_cmap('Dark2', len(clusters))  # Use 'tab10' for up to 10 distinct colors
cluster_colors = {cluster: colormap(i) for i, cluster in enumerate(clusters)}

plt.figure(figsize=(10, 6))

# Plot each cluster separately
for cluster in clusters:
    subset = calitoo_subset[calitoo_subset['Cluster'] == cluster]
    plt.scatter(
        subset['SZA'], 
        subset['AOT465'], 
        label=cluster_labels.get(cluster, f'{cluster}'), 
        color=cluster_colors[cluster]
    )

# Add labels and legend
plt.ylabel('AOT465')
plt.xlabel('SZA [°]')
plt.grid(True)
plt.legend(title='Location')  # Add a legend for clusters
plt.show()