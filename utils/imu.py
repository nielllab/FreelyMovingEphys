"""
imu.py
"""
import xarray as xr
import pandas as pd
import numpy as np
import os, yaml
from time import time

from utils.aux_funcs import *

from utils.base import BaseInput
from utils.imu_orientation import IMU_Orientation

class Imu(BaseInput):
    def __init__(self, config):
        super.__init__(self, config)

def process(self):
    """ Read an 8-channel binary file of variable length
    only channels 0-3, 4-7 will be saved out, channels, 3 and 7 are thrown out
    expected binary channel order: acc first, empty channel, then gyro, then empty channel
    returns a dataarray of constructed timestamps and imu readings from -5V to 5V
    dataarray values are downsampled by value in input dictionary config
    
    Parameters:
    imupath (str): imu binary file
    timepath (str): timestamp csv file to imu data
    config (dict): options
    
    Returns:
    imu_out (xr.DataArray): xarray of IMU data
    """
    # set up datatypes and names for each channel
    dtypes = np.dtype([
        ("acc_x",np.uint16),
        ("acc_y",np.uint16),
        ("acc_z",np.uint16),
        ("none1",np.uint16),
        ("gyro_x",np.uint16),
        ("gyro_y",np.uint16),
        ("gyro_z",np.uint16),
        ("none2",np.uint16)
    ])
    # read in binary file
    binary_in = pd.DataFrame(np.fromfile(self.imu_path, dtypes, -1, ''))
    binary_in = binary_in.drop(columns=['none1','none2'])
    if self.config['parameters']['imu']['flip_gx_gy']:
        temp = binary_in.iloc[:,3].copy()
        binary_in.iloc[:,3] = binary_in.iloc[:,4].copy()
        binary_in.iloc[:,4] = temp
        # binary_in.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    # convert to -5V to 5V
    data = 10 * (binary_in.astype(float)/(2**16) - 0.5)
    # downsample
    data = data.iloc[::self.config['parameters']['imu']['imu_downsample']]
    data = data.reindex(sorted(data.columns), axis=1) # alphabetize columns
    samp_freq = self.config['parameters']['imu']['imu_sample_rate'] / self.config['parameters']['imu']['imu_downsample']
    # read in timestamps
    csv_data = pd.read_csv(self.imu_timestamps_path).squeeze()
    time = pd.DataFrame(self.read_timestamp_series(csv_data))
    # get first/last timepoint, num_samples
    t0 = time.iloc[0,0]; num_samp = np.size(data,0)
    # samples start at t0, and are acquired at rate of 'ephys_sample_rate'/ 'imu_downsample'
    newtime = list(np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq))
    IMU = IMU_Orientation()
    # convert accelerometer to g
    zero_reading = 2.9; sensitivity = 1.6
    acc = pd.DataFrame.to_numpy((data[['acc_x', 'acc_y', 'acc_z']]-zero_reading)*sensitivity)
    # convert gyro to deg/sec
    gyro = pd.DataFrame.to_numpy((data[['gyro_x', 'gyro_y', 'gyro_z']]-pd.DataFrame.mean(data[['gyro_x', 'gyro_y', 'gyro_z']]))*400)
    # collect roll & pitch
    roll_pitch = np.zeros([len(acc),2])
    for x in range(len(acc)):
        roll_pitch[x,:] = IMU.process((acc[x],gyro[x])) # update by row
    roll_pitch = pd.DataFrame(roll_pitch, columns=['roll','pitch'])
    # collect the data together to return
    all_data = pd.concat([data.reset_index(), pd.DataFrame(acc).reset_index(), pd.DataFrame(gyro).reset_index(), roll_pitch], axis=1).drop(labels='index',axis=1)
    all_data.columns = ['acc_x_raw', 'acc_y_raw', 'acc_z_raw', 'gyro_x_raw', 'gyro_y_raw', 'gyro_z_raw','acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'roll', 'pitch']
    output_data = xr.DataArray(all_data, dims=['sample','channel'])
    self.data = output_data.assign_coords({'sample':newtime})
    
    def save(self):
        self.data.to_netcdf(os.path.join(self.recording_path, str(self.recording_name + '_imu.nc')))