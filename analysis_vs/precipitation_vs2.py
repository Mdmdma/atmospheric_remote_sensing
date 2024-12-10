# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:24:14 2024

@author: Valu
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# import data
precipitation = pd.read_csv(r'../data_precipitation/n_zuerich_fluntern.txt', delimiter=',')
calitoo_422 = pd.read_csv(r'../daten calitoo/calibrated_data/preprocessed_data_with_AOD_all_calitoo_422.csv', delimiter=',')
calitoo_425 = pd.read_csv(r'../daten calitoo/calibrated_data/preprocessed_data_with_AOD_all_calitoo_425.csv', delimiter=',')
calitoo_427 = pd.read_csv(r'../daten calitoo/calibrated_data/preprocessed_data_with_AOD_all_calitoo_427.csv', delimiter=',')

# filter calitoo data with r^2 and alpha
calitoo_422 = calitoo_422[(calitoo_422['Alpha'] < 2) & (calitoo_422.iloc[:,17] > 0.9)]


# plot filtered timeseries of our data
plt.figure(figsize=(10, 6))
plt.scatter(calitoo_422['Date'], calitoo_422['AOT465'])

plt.show()

# ============================================================================= very restrictive outlier removal
# # simple outlier removal
# # Calculate Q1, Q3, and IQR
# Q1 = calitoo_422['AOT465'].quantile(0.25)
# Q3 = calitoo_422['AOT465'].quantile(0.75)
# IQR = Q3 - Q1
# 
# # Define the acceptable range
# lower_bound = Q1 - 3 * IQR
# upper_bound = Q3 + 3 * IQR
# 
# # Remove outliers
# calitoo_422 = calitoo_422[(calitoo_422['AOT465'] >= lower_bound) & (calitoo_422['AOT465'] <= upper_bound)]
# =============================================================================

# ============================================================================= about the same
# # outlier removal using first an 99 percentile
# # Calculate the 1st (1%) and 99th (99%) quantiles
lower_quantile = calitoo_422['AOT465'].quantile(0.01)
upper_quantile = calitoo_422['AOT465'].quantile(0.99)
# 
# # Set values outside the 1st and 99th quantiles to NaN
calitoo_422['AOT465'] =calitoo_422['AOT465'].apply(lambda x: x if lower_quantile <= x <= upper_quantile else np.nan)

# # Calculate the 1st (1%) and 99th (99%) quantiles
lower_quantile = calitoo_422['AOT540'].quantile(0.01)
upper_quantile = calitoo_422['AOT540'].quantile(0.99)
# 
# # Set values outside the 1st and 99th quantiles to NaN
calitoo_422['AOT540'] =calitoo_422['AOT540'].apply(lambda x: x if lower_quantile <= x <= upper_quantile else np.nan)

# # Calculate the 1st (1%) and 99th (99%) quantiles
lower_quantile = calitoo_422['AOT619'].quantile(0.01)
upper_quantile = calitoo_422['AOT619'].quantile(0.99)
# 
# # Set values outside the 1st and 99th quantiles to NaN
calitoo_422['AOT619'] =calitoo_422['AOT619'].apply(lambda x: x if lower_quantile <= x <= upper_quantile else np.nan)
# =============================================================================

# remove values above 1
#calitoo_422['AOT465'] =calitoo_422['AOT465'].apply(lambda x: x if  x <= 1 else np.nan)

# Set global font size for the entire plot
plt.rcParams.update({'font.size': 14})  # Set the font size to 14 for all

# plot filtered timeseries of our data
plt.figure(figsize=(10, 6))
plt.scatter(calitoo_422['day_of_year'], calitoo_422['AOT465'],marker='x', color='blue')
plt.scatter(calitoo_422['day_of_year'], calitoo_422['AOT540'],marker='x', color='green')
plt.scatter(calitoo_422['day_of_year'], calitoo_422['AOT619'],marker='x', color='red')

plt.grid(True)
plt.title('calibrated and outlier-removed data')
plt.xlabel('doy')
plt.ylabel('AOT')
plt.ylim(0,0.35)

plt.tight_layout()
plt.show()

## subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 10))  # 3 rows, 1 column

# Plot for AOT465
axes[0].scatter(calitoo_422['day_of_year'], calitoo_422['AOT465'], marker='x', color='blue')
axes[0].set_title('AOT465')
axes[0].set_xlabel('Day of Year')
axes[0].set_ylabel('AOT')
axes[0].grid(True)
axes[0].set_ylim(0, 0.35)

# Plot for AOT540
axes[1].scatter(calitoo_422['day_of_year'], calitoo_422['AOT540'], marker='x', color='green')
axes[1].set_title('AOT540')
axes[1].set_xlabel('Day of Year')
axes[1].set_ylabel('AOT')
axes[1].grid(True)
axes[1].set_ylim(0, 0.35)

# Plot for AOT619
axes[2].scatter(calitoo_422['day_of_year'], calitoo_422['AOT619'], marker='x', color='red')
axes[2].set_title('AOT619')
axes[2].set_xlabel('Day of Year')
axes[2].set_ylabel('AOT')
axes[2].grid(True)
axes[2].set_ylim(0, 0.35)

# Adjust layout
plt.tight_layout()

plt.savefig("aot_cal_filtered.png", dpi=300, bbox_inches="tight")

# Show the plots
plt.show()

############################## statistic
mean_465 = calitoo_422['AOT465'].mean()
std_465 = calitoo_422['AOT465'].std()

mean_540 = calitoo_422['AOT540'].mean()
std_540 = calitoo_422['AOT540'].std()

mean_619 = calitoo_422['AOT619'].mean()
std_619 = calitoo_422['AOT619'].std()

############################### add all calitoo files in one 

calitoo_all = pd.concat([calitoo_422, calitoo_425, calitoo_427], ignore_index=True)

# plot filtered timeseries of all data
plt.figure(figsize=(10, 6))
plt.scatter(calitoo_all['day_of_year'], calitoo_all['AOT465'])

plt.show()

## filter
calitoo_all = calitoo_all[(calitoo_all['Alpha'] < 2) & (calitoo_all.iloc[:,17] > 0.9)]

# ============================================================================= about the same
# # outlier removal using first an 99 percentile
# # Calculate the 1st (1%) and 99th (99%) quantiles
lower_quantile = calitoo_all['AOT465'].quantile(0.01)
upper_quantile = calitoo_all['AOT465'].quantile(0.99)
# 
# # Set values outside the 1st and 99th quantiles to NaN
calitoo_all['AOT465'] =calitoo_all['AOT465'].apply(lambda x: x if lower_quantile <= x <= upper_quantile else np.nan)

# # Calculate the 1st (1%) and 99th (99%) quantiles
lower_quantile = calitoo_all['AOT540'].quantile(0.01)
upper_quantile = calitoo_all['AOT540'].quantile(0.99)
# 
# # Set values outside the 1st and 99th quantiles to NaN
calitoo_all['AOT540'] =calitoo_all['AOT540'].apply(lambda x: x if lower_quantile <= x <= upper_quantile else np.nan)

# # Calculate the 1st (1%) and 99th (99%) quantiles
lower_quantile = calitoo_all['AOT619'].quantile(0.01)
upper_quantile = calitoo_all['AOT619'].quantile(0.99)
# 
# # Set values outside the 1st and 99th quantiles to NaN
calitoo_all['AOT619'] =calitoo_all['AOT619'].apply(lambda x: x if lower_quantile <= x <= upper_quantile else np.nan)
# =============================================================================
# plot filtered timeseries of all data
plt.figure(figsize=(10, 6))
plt.scatter(calitoo_all['day_of_year'], calitoo_all['AOT465'])

plt.show()

############################################### precipitation
#################################################################################
# use only months 9, 10, 11
calitoo_all['Datetime_2'] = pd.to_datetime(calitoo_all['Date'])
precipitation['Datetime_2'] = pd.to_datetime(precipitation['Date'])
precipitation['day_of_year'] = precipitation['Datetime_2'].dt.dayofyear
precipitation['prev_day_p'] = precipitation['p'].shift(1)
calitoo_all_sub = calitoo_all[calitoo_all['Datetime_2'].dt.month.isin([9,10,11])]

# get all points close to zurich
# cluster points on location
coordinates = calitoo_all_sub[['N','E']].to_numpy()
epsilon = 17/6371 # convert 17 km to radians for DBSCAN
db = DBSCAN(eps=epsilon, min_samples=1, metric='haversine').fit(np.radians(coordinates))

calitoo_all_sub['Cluster'] = db.labels_

# all points in zurich
calitoo_zurich =  calitoo_all_sub[calitoo_all_sub['Cluster'] == 0]

# merge precipitation and calitoo data
p_cal= pd.merge(precipitation, calitoo_zurich, on='day_of_year', how='outer')

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

ax1.scatter(p_cal['day_of_year'],p_cal['AOT465'],color='red',marker='x')
ax2.bar(p_cal['day_of_year'],p_cal['p'],color='blue')

plt.xlim(270,360)

ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.show()



fig, ax1 = plt.subplots(figsize=(8, 4))  # Create a single subplot

# Scatter plot on the first y-axis
ax1.scatter(p_cal['day_of_year'], p_cal['AOT465'], color='red', marker='x', label='AOT465')
ax1.set_ylabel('AOT465', color='red')
ax1.tick_params(axis='y', labelcolor='red')
ax1.grid(True)

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
ax2.bar(p_cal['day_of_year'], p_cal['p'], color='blue', alpha=0.6, label='Pressure')
ax2.set_ylabel('precipitation [mm]', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Set shared x-axis properties
ax1.set_xlabel('Day of Year')
plt.xlim(270, 340)

# Show the plot
plt.tight_layout()
plt.savefig("P_aot_465.png", dpi=300, bbox_inches="tight")
plt.show()

##
## correlation
p_cal_clean = p_cal.dropna(subset=['AOT465'])
p_cal_clean = p_cal_clean[p_cal_clean['AOT465'] != 0]

# cor coefficient
from scipy.stats import spearmanr

cor_coeff, p_value = spearmanr(p_cal_clean['p'],p_cal_clean['AOT465'])

cor_coeff = round(cor_coeff,3)
p_value = f"{p_value:.2e}"

plt.figure(figsize=(8, 7))
scatter = plt.scatter(p_cal_clean['p'],p_cal_clean['AOT465'],marker='x')
# Add color bar with a legend label
plt.ylabel('AOT465')
plt.xlabel('precipitation [mm]')

# Add text to the plot
plt.text(
    0.5, 0.9,  # x and y coordinates (in figure-relative coordinates, from 0 to 1)
    f"spearman correlation: {cor_coeff}\np-value: {p_value}",  # Multiline text with variables
    transform=plt.gca().transAxes,  # Transform to place text relative to axes
    fontsize=14,
    verticalalignment='top'
)

plt.grid(True)
plt.savefig("cor_p.png", dpi=300, bbox_inches="tight")
plt.show()


## precipitation shift one day

cor_coeff_prev, p_value_prev = spearmanr(p_cal_clean['prev_day_p'],p_cal_clean['AOT465'])

cor_coeff_prev = round(cor_coeff_prev,3)
p_value_prev = f"{p_value_prev:.2e}"

plt.figure(figsize=(8, 7))
scatter = plt.scatter(p_cal_clean['prev_day_p'],p_cal_clean['AOT465'],marker='x')

plt.ylabel('AOT465')
plt.xlabel('precipitation of the day before [mm]')

# Add text to the plot
plt.text(
    0.5, 0.9,  # x and y coordinates (in figure-relative coordinates, from 0 to 1)
    f"spearman correlation: {cor_coeff_prev}\np-value: {p_value_prev}",  # Multiline text with variables
    transform=plt.gca().transAxes,  # Transform to place text relative to axes
    fontsize=14,
    verticalalignment='top'
)

plt.grid(True)
plt.savefig("cor_p_prev.png", dpi=300, bbox_inches="tight")
plt.show()

 

################################ locations
###################################################################################
import folium
import geopandas as gpd
from shapely.geometry import Point

geometry = [Point(xy) for xy in zip(calitoo_all_sub['E'], calitoo_all_sub['N'])]
gdf = gpd.GeoDataFrame(calitoo_all_sub, geometry=geometry)

# Define Switzerland's bounding box
# Approximate coordinates for the southwestern and northeastern corners
switzerland_bounds = [[45.81792, 5.95608], [47.80845, 10.49203]]

# Create a Folium map centered in Switzerland
map_switzerland = folium.Map(
    location=[46.8182, 8.2275],  # Switzerland's approximate center
    zoom_start=8,  # Zoom level
    tiles='OpenStreetMap'  # Use OSM tiles
)


# Display the map
#map_switzerland.save("switzerland_map.html")

# Iterate through the GeoDataFrame and add points as markers
for idx, row in gdf.iterrows():
    folium.Marker(
        location=[row['geometry'].y, row['geometry'].x],  # Latitude, Longitude
          ).add_to(map_switzerland)
    
map_switzerland.save("locations.html")
