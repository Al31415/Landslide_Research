import pandas as pd
from datetime import timedelta
from meteostat import Point, Daily
import numpy as np

# Assuming Input_Data is a DataFrame with 'event_date', 'Latitude', and 'Longitude' columns
max_1_arr = [None]*len(Input_Data)
max_3_arr = [None]*len(Input_Data)
max_7_arr = [None]*len(Input_Data)
max_14_arr = [None]*len(Input_Data)

avg_30_arr = [None]*len(Input_Data)
avg_60_arr = [None]*len(Input_Data)
avg_90_arr = [None]*len(Input_Data)
avg_365_arr = [None]*len(Input_Data)

for i in range(len(Input_Data)):
    end = Input_Data.iloc[i]['event_date']
    lat = Input_Data.iloc[i]['Latitude']
    lon = Input_Data.iloc[i]['Longitude']

    # Define time intervals
    day_1 = end - timedelta(days=1)
    day_3 = end - timedelta(days=3)
    day_7 = end - timedelta(days=7)
    day_14 = end - timedelta(days=14)
    day_30 = end - timedelta(days=30)
    day_60 = end - timedelta(days=60)
    day_90 = end - timedelta(days=90)
    day_365 = end - timedelta(days=365)

    # Create a Point object
    location = Point(lat, lon)

    # Fetch data for each interval
    # For max precipitation (short intervals)
    data_1 = Daily(location, day_1, end).fetch()
    data_3 = Daily(location, day_3, end).fetch()
    data_7 = Daily(location, day_7, end).fetch()
    data_14 = Daily(location, day_14, end).fetch()

    # For average precipitation (long intervals)
    data_30 = Daily(location, day_30, end).fetch()
    data_60 = Daily(location, day_60, end).fetch()
    data_90 = Daily(location, day_90, end).fetch()
    data_365 = Daily(location, day_365, end).fetch()

    # Compute maximum precipitation for short intervals
    max_1_arr[i] = data_1['prcp'].max() if not data_1.empty else np.nan
    max_3_arr[i] = data_3['prcp'].max() if not data_3.empty else np.nan
    max_7_arr[i] = data_7['prcp'].max() if not data_7.empty else np.nan
    max_14_arr[i] = data_14['prcp'].max() if not data_14.empty else np.nan

    # Compute average precipitation for long intervals
    avg_30_arr[i] = data_30['prcp'].mean() if not data_30.empty else np.nan
    avg_60_arr[i] = data_60['prcp'].mean() if not data_60.empty else np.nan
    avg_90_arr[i] = data_90['prcp'].mean() if not data_90.empty else np.nan
    avg_365_arr[i] = data_365['prcp'].mean() if not data_365.empty else np.nan

# You now have arrays for all the metrics. You can optionally add them as new columns to your Input_Data DataFrame:
Input_Data['max_1_day_prcp'] = max_1_arr
Input_Data['max_3_day_prcp'] = max_3_arr
Input_Data['max_7_day_prcp'] = max_7_arr
Input_Data['max_14_day_prcp'] = max_14_arr

Input_Data['avg_30_day_prcp'] = avg_30_arr
Input_Data['avg_60_day_prcp'] = avg_60_arr
Input_Data['avg_90_day_prcp'] = avg_90_arr
Input_Data['avg_365_day_prcp'] = avg_365_arr


