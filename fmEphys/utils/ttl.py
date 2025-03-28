"""
fmEphys/utils/ttl.py

TTL signal preprocessing.

Classes
-------
TTL
    TTL signal preprocessing.


Written by DMM, 2023
"""


import os
import numpy as np
import pandas as pd

import fmEphys as fme


class TTL():
    """ TTL signal preprocessing.

    TTL signals are written into the IMU binary file for head-fixed
    recordings ("..._IMU.bin"). They use the timestamps of the ephys
    board ("...BonsaiBoardTS.csv"). The binary file will have the
    shape (8, n_samples). Only channels 3 and 7 are used for the TTL,
    and channels 0:3 and 4:7 will be ignored.

    One of the TTL channels (ch 3) records the start of each stimulus
    frame. The other (ch 7) recordings the start of each stimulus repeat.

    Parameters
    ----------
    cfg: dict
        Dictionary of pipeline options.
    recording_name: str
        Name of the recording.
    recording_path: str
        Path to the recording directory.
    """


    def __init__(self, cfg, recording_name, recording_path):
        """ Initialize the TTL object.
        """

        self.cfg = cfg
        self.recording_name = recording_name
        self.recording_path = recording_path
        

    def gather_files(self):
        """ Gather the files of TTL data.
        """

        # Timestamp file
        csv_paths = fme.find(('*BonsaiBoardTS*.csv'), self.recording_path)
        self.imu_timestamps_path = fme.filter_file_search(csv_paths,
                                        keep=['_Ephys'], MR=True)
        
        # IMU binary file
        self.imu_path = fme.find(self.recording_name+'_IMU.bin',
                                   self.recording_path)[0]
        

    def process(self):
        """ Process the TTL data.
        """

        self.gather_files()

        # set up datatypes and names for each channel
        dtypes = np.dtype([
            ("none0",np.uint16),
            ("none1",np.uint16),
            ("none2",np.uint16),
            ("frame_TTL",np.uint16),
            ("none4",np.uint16),
            ("none5",np.uint16),
            ("none6",np.uint16),
            ("stim_TTL",np.uint16)
        ])

        # read in binary file
        binary_in = pd.DataFrame(np.fromfile(self.imu_path, dtypes, -1, ''))
        binary_in = binary_in[['frame_TTL','stim_TTL']]

        data = 10 * (binary_in.astype(float)/(2**16) - 0.5)

        # downsample
        data = data.iloc[::self.cfg['imu_dwnsmpl']]
        data = data.reindex(sorted(data.columns), axis=1) # alphabetize columns
        samp_freq = self.cfg['imu_samprate'] / self.cfg['imu_dwnsmpl']

        # read in timestamps
        csv_data = pd.read_csv(self.imu_timestamps_path).squeeze()
        pdtime = pd.DataFrame(fme.fmt_time(csv_data))

        # get first/last timepoint, num_samples
        t0 = pdtime.iloc[0,0]
        num_samp = np.size(data,0)

        # samples start at t0, and are acquired at rate of 'ephys_sample_rate'/ 'imu_downsample'
        newtime = list(np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq))
        
        self.data = {
            'frame_TTL': data['frame_TTL'].to_list(),
            'stim_TTL': data['stim_TTL'].to_list(),
            'time': newtime
        }
        
        savename = '{}_preprocessed_TTL_data.h5'.format(self.recording_name)
        savepath  = os.path.join(self.recording_path, savename)
        fme.write_h5(savepath, self.data)
        print('Saved {}'.format(savepath))
        
        return self.data

