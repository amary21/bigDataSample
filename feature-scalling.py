#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 07:33:03 2020

@author: amary
"""

import pandas as pd
import numpy as np

def minmax_method(data_set):
    fitur_minmax = data_set.copy()
    kol_minmax = pd.DataFrame(fitur_minmax['Jum_kamar'])
    min_value = np.min(kol_minmax)
    max_value = np.max(kol_minmax)
    kol_minmax = (kol_minmax - min_value) / (max_value - min_value)
    fitur_minmax['Jum_kamar'] = kol_minmax
    return fitur_minmax

def mean_method(data_set):
    fitur_mean = data_set.copy()
    kol_mean = pd.DataFrame(fitur_mean['Jum_kamar'])
    min_value = np.min(kol_mean)
    max_value = np.max(kol_mean)
    mean_value = np.mean(kol_mean)
    kol_mean = (kol_mean - mean_value) / (max_value - min_value)
    fitur_mean['Jum_kamar'] = kol_mean
    return fitur_mean

def zscore_method(data_set):
    fitur_zscore = data_set.copy()
    kol_zscore = pd.DataFrame(fitur_zscore['Jum_kamar'])
    mean_value = np.mean(kol_zscore)
    std_value = np.std(kol_zscore)
    kol_zscore = (kol_zscore - mean_value) / (std_value)
    fitur_zscore['Jum_kamar'] = kol_zscore
    return fitur_zscore
    

df = pd.read_csv('Apartemen_numerik.csv')
print(df)

print(df.describe())

df.loc[2, 'Jum_kamar'] = 100
print(df)
print(df.shape)
print(df.Jum_kamar)

print("Min-Max Normalization :\n", minmax_method(df),"\n")
print("Mean Normalization :\n", mean_method(df),"\n")
print("Z-Score Normalization :\n", zscore_method(df),"\n")
