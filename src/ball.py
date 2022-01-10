"""
FreelyMovingEphys/src/ball.py
"""
import os
import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from src.utils.filter import convfilt
from src.base import BaseInput
from src.utils.path import find

class RunningBall(BaseInput):
    """ Preprocess data from spherical treadmill, with movement recorded by optical mouse.
    """
    def __init__(self, config, recording_name, recording_path):
        BaseInput.__init__(self, config, recording_name, recording_path)
        # output sample rate (data will be set up to match this sample rate,
        # since there is not constant sample rate for optical mouse data
        # from Bonsai)
        self.ball_samprate = 0.008
        # float in seconds, window in which a previous timestamp in
        # sparse_time must fall, otherwise a zero will be filled in
        self.timestamp_seek_window = 0.030

    def sparse_to_constant_timebase(self, sparse_time, arange_time, data):
        """ Adjust optical mouse data to match timestamps with constat time base, filling zeros
        for steps in time with no recorded sample.
        
        Parameters
        --------
        sparse_time : np.array
            Timestamps (in seconds) for samples do not exist when there was no change in data.
        arange_time : np.array
            Timestamps (in seconds) with a constant step size, where start and end match sparse_time.
        data : np.array
            Array of values that match the timebase of sparse_time.
        
        Returns
        --------
        data_out : np.array
            data with constant time step size and with zeros filled in where both data and
            sparse_time previously had no values
        """
        data_out = np.zeros(len(arange_time))
        for t in sparse_time:
            ind = np.searchsorted(arange_time, t)
            if ind < len(arange_time):
                data_out[ind] = (data[ind] if t >= (arange_time[ind]-self.timestamp_seek_window) and t <= arange_time[ind] else 0)
        return data_out

    def save_params(self):
        """ Save preprocessed position and speed data as an .nc file.
        """
        self.data.to_netcdf(os.path.join(self.recording_path, str(self.recording_name + '_speed.nc')))

    def gather_ball_files(self):
        self.running_ball_path = find(self.recording_name+'_BALLMOUSE_BonsaiTS_X_Y.csv', self.recording_path)[0]

    def treadmill_speed(self):
        """ Track the movement of the ball for headfixed recordings.
        """
        # get coordinates on screen where optical mouse is centered out of preprocessing config file
        screen_center = self.config['internals']['optical_mouse_screen_center']
        centX = screen_center['x']; centY = screen_center['y']
        # read in one csv file with timestamps, x position, and y position in three columns
        csv_data = pd.read_csv(self.running_ball_path)
        # from this, we can get the timestamps, as seconds since midnight before the recording
        time = self.read_timestamp_series(csv_data['Timestamp.TimeOfDay'])
        # convert center-subtracted pixels into cm
        x_pos = (csv_data['Value.X']-centX) / self.config['internals']['optical_mouse_pxls_to_cm']
        y_pos = (csv_data['Value.Y']-centY) / self.config['internals']['optical_mouse_pxls_to_cm']
        # set up new time base
        t0 = time[0]; t_end = time[-1]
        arange_time = np.arange(t0, t_end, self.ball_samprate)
        # interpolation of xpos, ypos 
        xinterp = interp1d(time, x_pos, bounds_error=False, kind='nearest')(arange_time)
        yinterp = interp1d(time, y_pos, bounds_error=False, kind='nearest')(arange_time)
        # if no timestamp within 30ms, set interpolated val to 0
        full_x = self.sparse_to_constant_timebase(time, arange_time, xinterp)
        full_y = self.sparse_to_constant_timebase(time, arange_time, yinterp)
        # cm per second
        xpersec = full_x[:-1] / np.diff(arange_time)
        ypersec = full_y[:-1] / np.diff(arange_time)
        # speed
        speed = convfilt(np.sqrt(xpersec**2 + ypersec**2), 10)
        # collect all data
        all_data = pd.DataFrame([time, full_x, full_y, xpersec, ypersec, speed]).T
        all_data.columns = ['timestamps','cm_x','cm_y','x_persec','y_persec','speed_cmpersec']
        # and build into xarray before adding as attribute
        self.data = xr.DataArray(all_data.T, dims={'frame','move_params'})
        
    def process(self):
        self.gather_ball_files()
        self.treadmill_speed()
        self.save_params()