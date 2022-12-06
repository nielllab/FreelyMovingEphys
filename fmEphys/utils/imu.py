"""
FreelyMovingEphys/src/imu.py
"""
import os
import yaml

from time import time

import numpy as np
import pandas as pd
import xarray as xr

import fmEphys

class Imu(fmEphys.BaseInput):
    def __init__(self, config, recording_name, recording_path):
        fmEphys.BaseInput.__init__(self, config, recording_name, recording_path)

    def gather_imu_files(self):
        csv_paths = [x for x in fmEphys.find(('*BonsaiBoardTS*.csv'), self.recording_path) if x != []]
        self.imu_timestamps_path = next(i for i in csv_paths if 'Ephys_' in i)
        self.imu_path = fmEphys.find(self.recording_name+'_IMU.bin', self.recording_path)[0]

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
        self.gather_imu_files()
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
        if self.config['internals']['flip_gyro_xy']:
            temp = binary_in.iloc[:,3].copy()
            binary_in.iloc[:,3] = binary_in.iloc[:,4].copy()
            binary_in.iloc[:,4] = temp
            # binary_in.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        # convert to -5V to 5V
        data = 10 * (binary_in.astype(float)/(2**16) - 0.5)
        # downsample
        data = data.iloc[::self.config['internals']['imu_dwnsmpl']]
        data = data.reindex(sorted(data.columns), axis=1) # alphabetize columns
        samp_freq = self.config['internals']['imu_samprate'] / self.config['internals']['imu_dwnsmpl']
        # read in timestamps
        csv_data = pd.read_csv(self.imu_timestamps_path).squeeze()
        pdtime = pd.DataFrame(self.read_timestamp_series(csv_data))
        # get first/last timepoint, num_samples
        t0 = pdtime.iloc[0,0]; num_samp = np.size(data,0)
        # samples start at t0, and are acquired at rate of 'ephys_sample_rate'/ 'imu_downsample'
        newtime = list(np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq))
        IMU = ImuOrientation()
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
        self.data.to_netcdf(os.path.join(self.recording_path, str(self.recording_name + '_imu.nc')))

class Kalman():
    """
    https://github.com/wehr-lab/autopilot/tree/parallax
    """
    def __init__(self, dim_state: int, dim_measurement: int = None, dim_control: int=0,
                 *args, **kwargs):

        self.dim_state = dim_state # type: int
        if dim_measurement is None:
            self.dim_measurement = self.dim_state # type: int
        else:
            self.dim_measurement = dim_measurement # type: int
        self.dim_control = dim_control # type: int

        self._init_arrays()

    def _init_arrays(self, state=None):
        """
        Initialize the arrays
        """
        # State arrays
        if state is not None:
            # TODO: check it's the right shape
            self.x_state = state
        else:
            self.x_state = np.zeros((self.dim_state, 1))

        # initialize kalman arrays
        self.P_cov               = np.eye(self.dim_state)                           # uncertainty covariance
        self.Q_proc_var          = np.eye(self.dim_state)                           # process uncertainty
        self.B_control           = np.eye(self.dim_control)                         # control transition matrix
        self.F_state_trans       = np.eye(self.dim_state)                           # x_state transition matrix
        if self.dim_state == self.dim_measurement:
            self.H_measure = np.eye(self.dim_measurement)
        else:
            self.H_measure           = np.zeros((self.dim_measurement, self.dim_state)) # measurement function
        self.R_measure_var       = np.eye(self.dim_measurement)                     # measurement uncertainty
        self._alpha_sq           = 1.                                               # fading memory control
        self.M_proc_measure_xcor = np.zeros((self.dim_state, self.dim_measurement)) # process-measurement cross correlation
        self.z_measure           = np.array([[None] * self.dim_measurement]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((self.dim_state, self.dim_measurement)) # kalman gain
        self.y = np.zeros((self.dim_measurement, 1))
        self.S = np.zeros((self.dim_measurement, self.dim_measurement)) # system uncertainty
        self.SI = np.zeros((self.dim_measurement, self.dim_measurement)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(self.dim_state)

        # these will always be a copy of x_state,P_cov after predict() is called
        self.x_prior = self.x_state.copy()
        self.P_prior = self.P_cov.copy()

        # these will always be a copy of x_state,P_cov after update() is called
        self.x_post = self.x_state.copy()
        self.P_post = self.P_cov.copy()

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next x_state (prior) using the Kalman filter x_state propagation
        equations.

        Parameters
        ----------

        u : np.array, default 0
            Optional control vector.

        B : np.array(dim_state, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B_control`.

        F : np.array(dim_state, dim_state), or None
            Optional x_state transition matrix; a value of None
            will cause the filter to use `self.F_state_trans`.

        Q : np.array(dim_state, dim_state), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q_proc_var`.
        """
        if B is None:
            B = self.B_control
        if F is None:
            F = self.F_state_trans
        if Q is None:
            Q = self.Q_proc_var
        elif np.isscalar(Q):
            Q = np.eye(self.dim_state) * Q

        # x_state = Fx + Bu
        if B is not None and u is not None:
            # make sure control vector is column
            u = np.atleast_2d(u)
            if u.shape[1] > u.shape[0]:
                u = u.T
            self.x_state = np.dot(F, self.x_state) + np.dot(B, u)
        else:
            self.x_state = np.dot(F, self.x_state)

        # P_cov = FPF' + Q_proc_var
        self.P_cov = self._alpha_sq * np.dot(np.dot(F, self.P_cov), F.T) + Q

        # save prior
        np.copyto(self.x_prior, self.x_state)
        np.copyto(self.P_prior, self.P_cov)

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z_measure) to the Kalman filter.

        If z_measure is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z_measure is set to None.

        Parameters
        ----------
        z : (dim_measurement, 1): array_like
            measurement for this update. z_measure can be a scalar if dim_measurement is 1,
            otherwise it must be convertible to a column vector.

            If you pass in a value of H_measure, z_measure must be a column vector the
            of the correct size.

        R : np.array, scalar, or None
            Optionally provide R_measure_var to override the measurement noise for this
            one call, otherwise  self.R_measure_var will be used.

        H : np.array, or None
            Optionally provide H_measure to override the measurement function for this
            one call, otherwise self.H_measure will be used.
        """
        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z_measure = np.array([[None] * self.dim_measurement]).T
            np.copyto(self.x_post, self.x_state)
            np.copyto(self.P_post, self.P_cov)
            self.y = np.zeros((self.dim_measurement, 1))
            return

        if R is None:
            R = self.R_measure_var
        elif np.isscalar(R):
            R = np.eye(self.dim_measurement) * R

        if H is None:
            z = self._reshape_z(z, self.dim_measurement, self.x_state.ndim)
            H = self.H_measure

        # y = z_measure - Hx
        # error (residual) between measurement and prediction
        self.y = z - np.dot(H, self.x_state)

        # common subexpression for speed
        PHT = np.dot(self.P_cov, H.T)

        # S = HPH' + R_measure_var
        # project system uncertainty into measurement space
        self.S = np.dot(H, PHT) + R
        self.SI = np.linalg.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = np.dot(PHT, self.SI)

        # x_state = x_state + Ky
        # predict new x_state with residual scaled by the kalman gain
        self.x_state = self.x_state + np.dot(self.K, self.y)

        # P_cov = (I-KH)P_cov(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P_cov = (I-KH)P_cov usually seen in the literature.

        I_KH = self._I - np.dot(self.K, H)
        self.P_cov = np.dot(np.dot(I_KH, self.P_cov), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        # save measurement and posterior x_state
        np.copyto(self.z_measure, z)
        np.copyto(self.x_post, self.x_state)
        np.copyto(self.P_post, self.P_cov)
        return self.x_state

    def _reshape_z(self, z, dim_z, ndim):
        """ ensure z is a (dim_z, 1) shaped vector"""

        z = np.atleast_2d(z)
        if z.shape[1] == dim_z:
            z = z.T

        if z.shape != (dim_z, 1):
            raise ValueError('z (shape {}) must be convertible to shape ({}, 1)'.format(z.shape, dim_z))

        if ndim == 1:
            z = z[:, 0]

        if ndim == 0:
            z = z[0, 0]

        return z

    def process(self, z, **kwargs):
        """
        Call predict and update, passing the relevant kwargs

        Args:
            z ():
            **kwargs ():

        Returns:
            np.ndarray: self.x_state
        """

        # prepare args for predict and call
        predict_kwargs = {k:kwargs.get(k, None) for k in ("u", "B", "F", "Q")}
        self.predict(**predict_kwargs)

        # same thing for update
        update_kwargs = {k: kwargs.get(k, None) for k in ('R', 'H')}
        return self.update(z, **update_kwargs)

    def residual_of(self, z):
        """
        Returns the residual for the given measurement (z_measure). Does not alter
        the x_state of the filter.
        """
        return z - np.dot(self.H_measure, self.x_prior)

    def measurement_of_state(self, x):
        """
        Helper function that converts a x_state into a measurement.

        Parameters
        ----------

        x : np.array
            kalman x_state vector

        Returns
        -------

        z_measure : (dim_measurement, 1): array_like
            measurement for this update. z_measure can be a scalar if dim_measurement is 1,
            otherwise it must be convertible to a column vector.
        """

        return np.dot(self.H_measure, x)

    @property
    def alpha(self):
        """
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
        """
        return self._alpha_sq**.5

    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value) or value < 1:
            raise ValueError('alpha must be a float greater than 1')

        self._alpha_sq = value**2

class ImuOrientation():
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
    
    https://github.com/wehr-lab/autopilot/tree/parallax
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