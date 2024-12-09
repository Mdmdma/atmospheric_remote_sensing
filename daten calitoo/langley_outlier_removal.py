# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:19:50 2024

@author: Nathalie
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pnadas as pd

from scipy.stats import linregress

def plot_langley_with_outlier_removal(data, voltage_column, color, label, threshold=2.0):
    # Step 1: Calculate the initial natural logarithm of the voltage
    data['ln_' + voltage_column] = np.log(data[voltage_column])

    # Perform the initial linear regression (fit)
    slope, intercept, r_value, p_value, std_err = linregress(data['air_mass'], data['ln_' + voltage_column])

    # Step 2: Plot the data points and the initial fitted line
    plt.scatter(data['air_mass'], data['ln_' + voltage_column], color=color)
    plt.plot(data['air_mass'], intercept + slope * data['air_mass'], color=color, linestyle='--', label=label)

    # Step 3: Calculate residuals (difference between the observed and fitted values)
    fitted_values = intercept + slope * data['air_mass']
    residuals = data['ln_' + voltage_column] - fitted_values

    # Step 4: Identify outliers (points with large residuals)
    # Using a simple threshold method (e.g., 2 standard deviations from the mean)
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Filter out points that are outliers
    outlier_condition = np.abs(residuals - mean_residual) > threshold * std_residual
    data_cleaned = data[~outlier_condition]

    # Step 5: Recalculate the Langley plot without outliers
    data_cleaned['ln_' + voltage_column] = np.log(data_cleaned[voltage_column])
    slope_cleaned, intercept_cleaned, r_value_cleaned, p_value_cleaned, std_err_cleaned = linregress(data_cleaned['air_mass'], data_cleaned['ln_' + voltage_column])

    # Step 6: Plot the cleaned data and the new fitted line
    plt.scatter(data_cleaned['air_mass'], data_cleaned['ln_' + voltage_column], color=color, label=f'{label} (cleaned)')
    plt.plot(data_cleaned['air_mass'], intercept_cleaned + slope_cleaned * data_cleaned['air_mass'], color=color, linestyle='-', label=f'{label} (fit without outliers)')

    # Step 7: Extract the calibration constant (intercept from the fit without outliers)
    calibration_constant_cleaned = np.exp(intercept_cleaned)
    print(f'Calibration Constant for {label} (cleaned): {calibration_constant_cleaned}')

    # Display the plot with both fits (initial and cleaned)
    plt.xlabel('Air Mass')
    plt.ylabel('ln(RAW)')
    plt.legend()
    plt.show()

    return calibration_constant_cleaned, data_cleaned