"""
fmEphys/utils/base.py

Base classes for preprocessing data.

Classes
-------
BaseInput
    Base preprocessing input data.


Written by DMM, 2021
"""


import numpy as np
import pandas as pd
from datetime import datetime

import fmEphys as fme


class BaseInput:
    """ Base preprocessing input data.

    Parameters
    ----------
    cfg : dict
        Dictionary of parameters.
    recording_name : str
        Name of recording. e.g. 070921_J553RT_Rig2_control_fm1
    recording_path : str
        Path to recording.

    Methods
    -------
    read_timestamp_series
        Read timestamps as a pd.Series and format time.
    interp_timestamps
        Interpolate timestamps for double the number of frames.
    read_timestamp_file
        Read timestamps from a .csv file.
        
    """


    def __init__(self, cfg, recording_name, recording_path):

        self.cfg = cfg
        self.recording_name = recording_name
        self.recording_path = recording_path


    def read_timestamp_series(self, s):
        """ Read timestamps as a pd.Series and format time.

        Parameters
        ----------
        s : pd.Series
            Timestamps as a Series. Expected to be formated as
            hours:minutes:seconds.microsecond

        Returns
        -------
        output_time : np.array
            Returned as the number of seconds that have passed since the
            previous midnight, with microescond precision, e.g. 700.000000

        """

        # Expected string format for timestamps.
        fmt = '%H:%M:%S.%f'

        output_time = []

        if s.dtype != np.float64:

            for current_time in s:

                str_time = str(current_time).strip()

                try:
                    t = datetime.strptime(str_time, fmt)

                except ValueError as v:
                    # If the string had unexpected characters (too much precision) for
                    # one timepoint, drop the extra characters.

                    ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                    
                    if ulr:
                        str_time = str_time[:-ulr]
                
                try:
                    output_time.append(
                            (datetime.strptime(str_time, '%H:%M:%S.%f')
                                - datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')
                                ).total_seconds())

                except ValueError:
                    output_time.append(np.nan)

            output_time = np.array(output_time)

        else:
            output_time = s.values

        return output_time


    def interp_timestamps(self, camT, use_medstep=False):
        """ Interpolate timestamps for double the number of
        frames. Compensates for video deinterlacing.
        
        Parameters
        ----------
        camT : np.array
            Camera timestamps aquired at 30Hz
        use_medstep : bool
            When True, the median diff(camT) will be used as the timestep
            in interpolation. If False, the timestep between each frame
            will be used instead.

        Returns
        -------
        camT_out : np.array
            Timestamps of camera interpolated so that there are twice the
            number of timestamps in the array. Each timestamp in camT will
            be replaced by two, set equal distances from the original.

        """

        camT_out = np.zeros(np.size(camT, 0)*2)
        medstep = np.nanmedian(np.diff(camT, axis=0))

        if use_medstep:
            
            # Shift each deinterlaced frame by 0.5 frame periods
            # forward/backwards assuming a constant framerate

            camT_out[::2] = camT - 0.25 * medstep
            camT_out[1::2] = camT + 0.25 * medstep
        
        elif not use_medstep:

            # Shift each deinterlaced frame by the actual time between
            # frames. If a camera frame was dropped, this approach will
            # be more accurate than `medstep` above.
            
            steps = np.diff(camT, axis=0, append=camT[-1]+medstep)
            camT_out[::2] = camT
            camT_out[1::2] = camT + 0.5 * steps

        return camT_out


    def read_timestamp_file(self, position_data_length=None,
                            force_timestamp_shift=False):
        """ Read timestamps from a .csv file.

        Parameters
        ----------
        position_data_length : None or int
            Number of timesteps in data from deeplabcut. This is used to
            determine whether or not the number of timestamps is too short
            for the number of video frames.
            Eyecam and Worldcam will have half the number of timestamps as
            the number of frames, since they are aquired as an interlaced
            video and deinterlaced in analysis. To fix this, timestamps need
            to be interpolated.
        force_timestamp_shift : bool
            When True, the timestamps will be interpolated regardless of
            whether or not the number of timestamps is too short for the
            number of frames. Default is False.

        Returns
        -------
        camT : np.array
            Timestamps of camera interpolated so that there are twice the
            number of timestamps in the array than there were in the provided
            csv file.

        """

        # Read data and set up format
        s = pd.read_csv(self.timestamp_path, encoding='utf-8',
                        engine='c', header=None).squeeze()
        
        # If the csv file has a header name for the column, (which is
        # is the int 0 for some early recordings), remove it.
        if s[0] == 0:
            s = s[1:]
        
        # Read the timestamps as a series and format them
        camT = self.read_timestamp_series(s)
        
        # Auto check if vids were deinterlaced
        if position_data_length is not None:

            if position_data_length > len(camT):

                # If the number of timestamps is too short for the number
                # of frames, interpolate the timestamps.

                camT = self.interp_timestamps(camT, use_medstep=False)
        
        # Force the times to be shifted if the user is sure it should be done
        if force_timestamp_shift is True:

            camT = self.interp_timestamps(camT, use_medstep=False)
        
        return camT

