"""
ball.py
"""
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from utils.time import open_time1

def find_previous_neighbor(sparse_time, arange_time, data, win=.30):
    """
    search timestamps with gaps between data for times in consistant timestamps
    then set up an array of data which matches the timebase of consistant timestamps
    and where zeros are filled in for each timestamp missing data from the sparse timestamps
    INPUTS
        sparse_time: timestamps (in seconds) for samples do not exist when there was no change in data
        arange_time: timestamps (in seconds) with a constant step size, where start and end match sparse_time
        data: array of values that match the timebase of sparse_time
        win: seconds, window in which a previous timestamp in sparse_time must fall, otherwise a zero will be filled in
    OUTPUTS
        data_out: data with constant time step size and with zeros filled in where both data and sparse_time previously had no values
    """
    data_out = np.zeros(len(arange_time))
    for t in sparse_time:
        ind = np.searchsorted(arange_time, t)
        if ind < len(arange_time):
            data_out[ind] = (data[ind] if t >= (arange_time[ind]-win) and t <= arange_time[ind] else 0)
    return data_out

def smooth_vals(y, box_pts=10):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def ball_tracking(csv_path, config):
    """
    track the movement of the ball for headfixed recordings
    INPUTS
        csv_path: optical mouse csv data file
        config: a dictionary of options
    OUTPUTS
        xr_out: xarray of optical mouse tracking data
    """
    # get coordinates on screen where optical mouse is centered out of preprocessing config file
    screen_center = config['parameters']['running_wheel']['optical_mouse_screen_center']
    centX = screen_center['x']; centY = screen_center['y']
    # read in one csv file with timestamps, x position, and y position in three columns
    csv_data = pd.read_csv(csv_path)
    # from this, we can get the timestamps, as seconds since midnight before the recording
    time = open_time1(csv_data['Timestamp.TimeOfDay'])
    # convert center-subtracted pixels into cm
    x_pos = (csv_data['Value.X']-centX) / config['parameters']['running_wheel']['optical_mouse_pix2cm']
    y_pos = (csv_data['Value.Y']-centY) / config['parameters']['running_wheel']['optical_mouse_pix2cm']
    # set up new time base
    t0 = time[0]; samprate = 0.008; t_end = time[-1]
    arange_time = np.arange(t0, t_end, samprate)
    # interpolation of xpos, ypos 
    xinterp = interp1d(time, x_pos, bounds_error=False, kind='nearest')(arange_time)
    yinterp = interp1d(time, y_pos, bounds_error=False, kind='nearest')(arange_time)
    # if no timestamp within 30ms, set interpolated val to 0
    full_x = find_previous_neighbor(time, arange_time, xinterp, win=.030)
    full_y = find_previous_neighbor(time, arange_time, yinterp, win=.030)
    # cm per second
    xpersec = full_x[:-1] / np.diff(arange_time)
    ypersec = full_y[:-1] / np.diff(arange_time)
    # speed
    speed = smooth_vals(np.sqrt(xpersec**2 + ypersec**2), 10)
    # collect all data
    all_data = pd.DataFrame([time, full_x, full_y, xpersec, ypersec, speed]).T
    all_data.columns = ['timestamps','cm_x','cm_y','x_persec','y_persec','speed_cmpersec']
    # and build into xarray before returning
    xr_out = xr.DataArray(all_data.T, dims={'frame','move_params'})
    return xr_out