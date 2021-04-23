"""
track_imu.py

read imu from binary
"""
import xarray as xr
import pandas as pd
import numpy as np
from scipy.signal import medfilt

from util.time import open_time1

# from autopilot.transform.geometry import IMU_Orientation
# IMU = IMU_Orientation()

def read_8ch_imu(imupath, timepath, config):
    """
    read an 8-channel binary file of variable length
    INPUTS:
        imupath -- imu binary file
        timepath -- timestamp csv file to imu data
        config -- options dict
    OUTPUTS:
        imu_out -- xarray of IMU data
    only channels 0-3, 4-7 will be saved out, channels, 3 and 7 are thrown out
    expected binary channel order: acc first, empty channel, then gyro, then empty channel
    returns a dataarray of constructed timestamps and imu readings from -5V to 5V
    dataarray values are downsampled by value in input dictionary config
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
    t0 = time.iloc[0,0]; num_samp = np.size(data,0)
    # samples start at t0, and are acquired at rate of 'ephys_sample_rate'/ 'imu_downsample'
    newtime = pd.DataFrame(np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq))
    # collect the data together to return
    all_data = data.copy()
    all_data.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    imu_out = xr.DataArray(all_data, dims={'channel','sample'})
    try:
        imu_out = imu_out.assign_coords(timestamps=('sample',list(newtime.iloc[:,0])))
    except ValueError:
        imu_out = imu_out.assign_coords(timestamps=('channel',list(newtime.iloc[:,0])))       
    
    return imu_out