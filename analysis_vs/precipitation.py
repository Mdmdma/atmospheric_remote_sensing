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


