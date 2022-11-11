import os, yaml
from time import time
import pandas as pd
import numpy as np

import fmEphys

def read_IMU_binary(path):

    # Set up the data types
    dtypes = np.dtype([
        ("acc_x",np.uint16), # accelerometer
        ("acc_y",np.uint16),
        ("acc_z",np.uint16),
        ("ttl1",np.uint16), # TTL
        ("gyro_x",np.uint16), # gyro
        ("gyro_y",np.uint16),
        ("gyro_z",np.uint16),
        ("ttl2",np.uint16) # TTL
    ])

    data = pd.DataFrame(np.fromfile(path, dtypes, -1, ''))

    return data

def preprocess_TTL(cfg, bin_path=None, csv_path=None):

    # Read in the binary
    ttl_data = read_IMU_binary(bin_path)

    # only keep the TTL channels
    ttl_data = ttl_data.loc(columns=['ttl1','ttl2'])

    # downsample
    ds = cfg['imu_ds']
    ttl_data = ttl_data.iloc[::ds]
    ttl_data = ttl_data.reindex(sorted(ttl_data.columns), axis=1) # alphabetize columns
    samp_freq = cfg['imu_samprate'] / ds

    # read in timestamps
    time = fmEphys.read_time(csv_path)

    # samples start at t0, and are acquired at rate of 'ephys_sample_rate'/ 'imu_downsample'
    t0 = time[0]
    nSamp = np.size(ttl_data, 0)
    imuT = list(np.array(t0 + np.linspace(0, nSamp-1, nSamp) / samp_freq))

    savedata = {
        'ttl1': ttl_data['ttl1'],
        'ttl2': ttl_data['ttl2'],
        'imuT': imuT
    }

    savepath = os.path.join(cfg['rpath'], '{}_ttl.h5'.format(cfg['rfname']))
    fmEphys.write_h5(savepath, savedata)

    return savedata
    
def preprocess_IMU(cfg, bin_path=None, csv_path=None):

    # Read in the binary
    imu_data = read_IMU_binary(bin_path)

    # Drop channels 3 and 7, which are either empty or contain TTL signals
    imu_data = imu_data.drop(columns=['ttl1','ttl2'])
        
    # convert to -5V to 5V
    imu_data = 10 * (imu_data.astype(float)/(2**16) - 0.5)

    # downsample
    ds = cfg['imu_ds']
    imu_data = imu_data.iloc[::ds]
    imu_data = imu_data.reindex(sorted(imu_data.columns), axis=1) # alphabetize columns
    samp_freq = cfg['imu_samprate'] / ds

    # read in timestamps
    time = fmEphys.read_time(csv_path)

    # samples start at t0, and are acquired at rate of 'ephys_sample_rate'/ 'imu_downsample'
    t0 = time[0]
    nSamp = np.size(imu_data, 0)
    imuT = list(np.array(t0 + np.linspace(0, nSamp-1, nSamp) / samp_freq))

    # convert accelerometer to g
    zero_reading = 2.9; sensitivity = 1.6
    acc = pd.DataFrame.to_numpy((imu_data[['acc_x', 'acc_y', 'acc_z']] - zero_reading) * sensitivity)
    
    # convert gyro to deg/sec
    gyro = pd.DataFrame.to_numpy((imu_data[['gyro_x', 'gyro_y', 'gyro_z']] -
                pd.DataFrame.mean(imu_data[['gyro_x', 'gyro_y', 'gyro_z']])) * 400)

    # roll & pitch
    IMU = fmEphys.ImuOrientation()
    roll_pitch = np.zeros([len(acc), 2])

    for x in range(len(acc)):
        roll_pitch[x,:] = IMU.process((acc[x], gyro[x])) # update by row
    roll_pitch = pd.DataFrame(roll_pitch, columns=['roll','pitch'])

    # organize the data before saving it out
    savedata = {
        'acc_x_raw': imu_data['acc_x'].to_numpy(),
        'acc_y_raw': imu_data['acc_y'].to_numpy(),
        'acc_z_raw': imu_data['acc_z'].to_numpy(),
        'gyro_x_raw': imu_data['gyro_x'].to_numpy(),
        'gyro_y_raw': imu_data['gyro_y'].to_numpy(),
        'gyro_z_raw': imu_data['gyro_z'].to_numpy(),
        'acc_x': acc[0],
        'acc_y': acc[1],
        'acc_z': acc[2],
        'gyro_x': gyro[0],
        'gyro_y': gyro[1],
        'gyro_z': gyro[2],
        'roll': roll_pitch['roll'].to_numpy(),
        'pitch': roll_pitch['pitch'].to_numpy(),
        'time': imuT
    }
    savepath = os.path.join(cfg['rpath'], '{}_imu_preprocessing.h5'.format(cfg['rfname']))
    fmEphys.write_h5(savepath, savedata)
