#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data_filename = 'C:\Python\nyc_data.csv'

data = pd.read_csv(data_filename, parse_dates=['pickup_datatmime','dropoff_datetime'])

data.columns