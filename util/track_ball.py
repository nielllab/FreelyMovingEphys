"""
track_ball.py

tracking mouse movement on ball or running wheel

Dec. 02, 2020
"""

import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime

from util.time import open_time1

# track the movement of the ball for headfixed recordings
def ball_tracking(csv_path, config):
    # get coordinates on screen where optical mouse is centered out of preprocessing config file
    screen_center = config['optical_mouse_screen_center']
    centX = screen_center['x']; centY = screen_center['y']
    # read in one csv file with timestamps, x position, and y position in three columns
    csv_data = pd.read_csv(csv_path)
    # should have date in timestamp
    # output will be in seconds since the midnight before the trial
    time = open_time1(csv_data['Timestamp.TimeOfDay'])
    # get the speed at each sample time
    diff = pd.DataFrame([csv_data['Value.X']-centX,csv_data['Value.Y']-centY]).T
    pixpersamp = [np.sqrt((diff.iloc[i,0])**2 + (diff.iloc[i,1])**2) for i in range(0,len(csv_data))] # pixels/sample
    # convert to cm/sec
    cmpersec = np.array(pixpersamp) * ((1/config['optical_mouse_pix2cm']) * (1/config['optical_mouse_sample_rate_ms']) * 1000) # 1000 converts to s from ms
    # assemble together components
    all_data = pd.DataFrame([time, csv_data['Value.X'], csv_data['Value.Y'], pixpersamp, cmpersec]).T
    all_data.columns = ['timestamps','x_pos','y_pos','pix_per_sample','cm_per_sec']
    # and build into xarray before returning
    xr_out = xr.DataArray(all_data, dims={'frame','move_params'})

    return xr_out
