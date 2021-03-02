"""
track_ball.py

tracking mouse movement on ball or running wheel

Feb. 15, 2021
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
    # from this, we can get the timestamps, as seconds since midnight before the recording
    time = open_time1(csv_data['Timestamp.TimeOfDay'])
    # convert center-subtracted pixels into cm
    x_pos = (csv_data['Value.X']-centX) / config['optical_mouse_pix2cm']
    y_pos = (csv_data['Value.Y']-centY) / config['optical_mouse_pix2cm']
    # cm per second
    xpersec = x_pos[1:] / np.diff(time)
    ypersec = y_pos[1:] / np.diff(time)
    # speed
    speed = np.sqrt(xpersec**2 + ypersec**2)
    # collect all data
    all_data = pd.DataFrame([time, x_pos, y_pos, xpersec, ypersec, speed]).T
    all_data.columns = ['timestamps','cm_x','cm_y','x_persec','y_persec','speed_cmpersec']
    # and build into xarray before returning
    xr_out = xr.DataArray(all_data.T, dims={'frame','move_params'})

    return xr_out
