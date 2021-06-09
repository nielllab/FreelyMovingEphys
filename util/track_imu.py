"""
track_imu.py

read imu from binary
"""
import xarray as xr
import pandas as pd
import numpy as np
import os, yaml
from time import time
from tqdm import trange, tqdm
from scipy.signal import medfilt

import sys
sys.path.insert(0, os.path.join('C:\\Users\\Niell Lab\\Documents\\GitHub\\FreelyMovingEphys\\'))

from util.paths import find, check_path, list_subdirs
from util.config import open_config
from util.time import open_time1
from util.Kalman import Kalman

class IMU_Orientation():
    """
    Compute absolute orientation (roll, pitch) from accelerometer and gyroscope measurements
    (eg from :class:`.hardware.i2c.I2C_9DOF` )

    Uses a :class:`.timeseries.Kalman` filter, and implements :cite:`patonisFusionMethodCombining2018a` to fuse
    the sensors

    Can be used with accelerometer data only, or with combined accelerometer/gyroscope data for
    greater accuracy

    Arguments:
        invert_gyro (bool): if the gyroscope's orientation is inverted from accelerometer measurement, multiply
            gyro readings by -1 before using
        use_kalman (bool): Whether to use kalman filtering (True, default), or return raw trigonometric
            transformation of accelerometer readings (if provided, gyroscope readings will be ignored)

    Attributes:
        kalman (:class:`.transform.timeseries.Kalman`): If ``use_kalman == True`` , the Kalman Filter.

    References:
        :cite:`patonisFusionMethodCombining2018a`
        :cite:`abyarjooImplementingSensorFusion2015`
    """

    def __init__(self, use_kalman:bool = True, invert_gyro:bool=False, *args, **kwargs):

        self.invert_gyro = invert_gyro # type: bool
        self._last_update = None # type: typing.Optional[float]
        self._dt = 0 # type: float
        # preallocate orientation array for filtered values
        self.orientation = np.zeros((2), dtype=float) # type: np.ndarray
        # and for unfiltered values so they aren't ambiguous
        self._orientation = np.zeros((2), dtype=float)  # type: np.ndarray

        self.kalman = None # type: typing.Optional[Kalman]
        if use_kalman:
            self.kalman = Kalman(dim_state=2, dim_measurement=2, dim_control=2)  # type: typing.Optional[Kalman]

    def process(self, accelgyro):
        """

        Args:
            accelgyro (tuple, :class:`numpy.ndarray`): tuple of (accelerometer[x,y,z], gyro[x,y,z]) readings as arrays, or
                an array of just accelerometer[x,y,z]

        Returns:
            :class:`numpy.ndarray`: filtered [roll, pitch] calculations in degrees
        """
        # check what we were given...
        if isinstance(accelgyro, (tuple, list)) and len(accelgyro) == 2:
            # combined accelerometer and gyroscope readings
            accel, gyro = accelgyro
        elif isinstance(accelgyro, np.ndarray) and np.squeeze(accelgyro).shape[0] == 3:
            # just accelerometer readings
            accel = accelgyro
            gyro = None
        else:
            # idk lol
            # self.logger.exception(f'Need input to be a tuple of accelerometer and gyroscope readings, or an array of accelerometer readings. got {accelgyro}')
            print('Error')
            return

        # convert accelerometer readings to roll and pitch
        pitch = 180*np.arctan2(accel[0], np.sqrt(accel[1]**2 + accel[2]**2))/np.pi
        roll = 180*np.arctan2(accel[1], np.sqrt(accel[0]**2 + accel[2]**2))/np.pi


        if self.kalman is None:
            # store orientations in external attribute if not using kalman filter
            self.orientation[:] = (roll, pitch)
            return self.orientation.copy()
        else:
            # if using kalman filter, use private array to store raw orientation
            self._orientation[:] = (roll, pitch)



        # TODO: Don't assume that we're fed samples instantatneously -- ie. once data representations are stable, need to accept a timestamp here rather than making one
        if self._last_update is None or gyro is None:
            # first time through don't have dt to scale gyro by
            self.orientation[:] = np.squeeze(self.kalman.process(self._orientation))
            self._last_update = time()

        else:
            if self.invert_gyro:
                gyro *= -1

            # get dt for time since last update
            update_time = time()
            self._dt = update_time-self._last_update
            self._last_update = update_time

            if self._dt>1:
                # if it's been really long, the gyro read is pretty much useless and will give ridiculous reads
                self.orientation[:] = np.squeeze(self.kalman.process(self._orientation))
            else:
                # run predict and update stages separately to incorporate gyro
                self.kalman.predict(u=gyro[0:2]*self._dt)
                self.orientation[:] = np.squeeze(self.kalman.update(self._orientation))

        return self.orientation.copy()


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
    data = data.iloc[0:-1:config['parameters']['imu']['imu_downsample'],:]
    samp_freq = config['parameters']['imu']['imu_sample_rate'] / config['parameters']['imu']['imu_downsample']
    # read in timestamps
    time = pd.DataFrame(open_time1(pd.read_csv(timepath).iloc[:,0]))
    # get first/last timepoint, num_samples
    t0 = time.iloc[0,0]; num_samp = np.size(data,0)
    # samples start at t0, and are acquired at rate of 'ephys_sample_rate'/ 'imu_downsample'
    newtime = pd.DataFrame(np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq))
    # collect the data together to return
    all_data = data.copy()
    all_data.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    IMU = IMU_Orientation()
    acc = pd.DataFrame.to_numpy((data[['acc_x', 'acc_y', 'acc_z']]-2.5)*1.6)
    gyro = pd.DataFrame.to_numpy((data[['gyro_x', 'gyro_y', 'gyro_z']]-pd.DataFrame.mean(data[['gyro_x', 'gyro_y', 'gyro_z']]))*400)
    # collect roll & pitch
    roll_pitch = np.zeros([len(acc),2])
    for x in trange(len(acc)):
        roll_pitch[x,:] = IMU.process((acc[x],gyro[x])) ### update by row
    roll_pitch = pd.DataFrame(roll_pitch, columns=['roll','pitch'], index=data.index)
    # read in timestamps
    time = pd.DataFrame(open_time1(pd.read_csv(timepath).iloc[:,0]))
    # get first/last timepoint, num_samples
    t0 = time.iloc[0,0]; num_samp = np.size(data,0)
    # samples start at t0, and are acquired at rate of 'ephys_sample_rate'/ 'imu_downsample'
    newtime = pd.DataFrame(np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq))
    # collect the data together to return
    all_data = pd.concat([data.copy(), roll_pitch], axis = 1)
    all_data.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z','roll','pitch']
    imu_out = xr.DataArray(all_data, dims={'channel','sample'})
    imu_out = imu_out.assign_coords(timestamps=('sample',list(newtime.iloc[:,0])))
    
    return imu_out
  
if __name__ == '__main__':
    config_path = 'T:/freely_moving_ephys/ephys_recordings/041521/EE11P13LTRN/config.yaml'
    if type(config_path) == dict:
        # if config options were provided instead of the expected path to a file
        config = config_path
    else:
        with open(config_path, 'r') as infile:
            config = yaml.load(infile, Loader=yaml.FullLoader)
    recording_names = [i for i in list_subdirs(config['data_path']) if 'hf' in i or 'fm' in i]
    recording_paths = [os.path.join(config['data_path'], recording_name) for recording_name in recording_names]
    recordings_dict = dict(zip(recording_names, recording_paths))

    config['recording_path'] = recordings_dict['fm1']
    recording_name = '_'.join(os.path.splitext(os.path.split([i for i in find('*.avi', config['recording_path']) if all(bad not in i for bad in ['plot','IR','rep11','betafpv','side_gaze'])][0])[1])[0].split('_')[:-1])
    trial_imu_csv = os.path.join(config['recording_path'],recording_name+'_Ephys_BonsaiBoardTS.csv') # use ephys timestamps
    trial_imu_bin = find(('*IMU.bin'), config['recording_path'])
    imu_data = read_8ch_imu(trial_imu_bin[0], trial_imu_csv, config); imu_data.name = 'IMU_data'
    imu_data.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_imu.nc')))
