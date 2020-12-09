"""
track_imu.py

read imu from binary

Dec. 07, 2020
"""
# package imports
import xarray as xr
import pandas as pd
import numpy as np
from scipy.signal import medfilt
# module imports
from util.time import open_time1

# read an 8-channel binary file of variable length
# only channels 0-3, 4-7 will be saved out, channels, 3 and 7 are thrown out
# expected binary channel order: acc first, empty channel, then gyro, then empty channel
# returns a dataarray of constructed timestamps and imu readings from -5V to 5V
# dataarray values are downsampled by value in input dictionary config
def read_8ch_imu(imupath, timepath, config):
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
    binary_in = pd.DataFrame(np.fromfile(imupath, dtypes, -1, ''))
    binary_in = binary_in.drop(columns=['none1','none2'])
    # convert to -5V to 5V
    data = 10 * (binary_in.astype(float)/(2**16) - 0.5)
    # downsample
    data = data.iloc[0:-1:config['imu_downsample'],:]
    samp_freq = config['imu_sample_rate'] / config['imu_downsample']
    # read in timestamps
    time = pd.DataFrame(open_time1(pd.read_csv(timepath).iloc[:,0]))
    # get first/last timepoint, num_samples
    t0 = time.iloc[0]; t_end = time.iloc[-1]; num_samp = np.size(data,0)
    # make timestamps for all subsequent samples
    newtime = pd.DataFrame(np.linspace(t0, t_end, num=num_samp))
    # collect the data together to return
    all_data = data.copy()
    all_data.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    imu_out = xr.DataArray(all_data, dims={'sample','channel'})
    imu_out = imu_out.assign_coords(timestamps=('channel',list(newtime.iloc[:,0])))
    
    return imu_out

# convert acc and gyro to g and deg/sec
def convert_acc_gyro(imu_out, timepath, config):
    # read in timestamps
    time = pd.DataFrame(open_time1(pd.read_csv(timepath).iloc[:,0]))
    # get first/last timepoint, num_samples
    t0 = time.iloc[0]; t_end = time.iloc[-1]; num_samp = np.size(data,0)
    # make timestamps for all subsequent samples
    newtime = pd.DataFrame(np.linspace(t0, t_end, num=num_samp))
    # convert from V to deg/sec for gyro
    samp_freq = config['imu_sample_rate'] / config['imu_downsample']
    gyro = imu_out.isel(sample=range(3,6)) * (400 / samp_freq)
    # median filter and conversion for acc
    filt_win = 5
    filt_acc = (medfilt(imu_out.isel(sample=range(0,3)), filt_win)-2.5)*1.6
    filt_acc[filt_acc>1] = 1; filt_acc[filt_acc<-1] = -1
    acc = np.rad2deg(np.arcsin(filt_acc))
    # change format of acc back to xarray since this was lost during conversion
    acc = pd.DataFrame(acc)
    acc.columns = ['acc_x', 'acc_y', 'acc_z']
    acc_xr = xr.DataArray(acc, dims=['sample', 'channel'])
    acc_xr = acc_xr.assign_coords(timestamps=('channel',list(newtime.iloc[:,0])))

    return acc_xr, gyro