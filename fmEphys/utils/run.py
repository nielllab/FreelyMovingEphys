"""
fmEphys/utils/run.py

This module contains the Session class, which is used to preprocess
and analyze a single session.

Classes
-------
Session
    Class for preprocessing and analyzing a single session.


Written by DMM, 2021
"""


import os
import yaml

import fmEphys as fme


class Session:
    """ Class for preprocessing and analyzing a single session.

    Methods
    -------
    get_session_recordings
        Collect recording names and paths in a dictionary.
    clear_dlc
        Delete all DeepLabCut hdf and pickle files in the session directory.
    preprocessing
        Preprocess all recordings in the session.
    ephys_analysis
        Analyze the ephys and stimulus data for all recordings.
    run_main
        Run the preprocessing pipeline.

    """


    def __init__(self, cfg_path):

        # Read config file
        if type(cfg_path)==str:

            with open(cfg_path, 'r') as infile:
                _tmp_cfg = yaml.load(infile, Loader=yaml.FullLoader)

        # Unless it's already a dictionary
        elif type(cfg_path)==dict:
            _tmp_cfg = cfg_path.copy()

        # Read the internals file, which is in the /utils/ directory of
        # the repository. This file contains the default values for all
        # parameters that can be specified in the config file.
        internals_path = os.path.join(os.path.split(__file__)[0], 'internals.yml')
        
        with open(internals_path, 'r') as infile:
            internals_dict = yaml.load(infile, Loader=yaml.FullLoader)

        # Anything specificied in the config file will override the default
        # value from internals.yml
        # e.g., for z = {**x, **y} and a key in both x and y, the value for
        # that key in y will be the one represented in z, and the one in x
        # will be overwritten.
        self.cfg = {**internals_dict, **_tmp_cfg}


    def get_session_recordings(self):
        """ Collect recording names and paths in a dictionary.

        The dictionary is stored as self.recordings_dict, and has the format
        {recording_name: recording_path}, with keys sorted to put freely moving
        recordings before head-fixed recordings.
        """

        # Get a list of recordings by checking the names of all subdirectories
        # in the session directory. If the cfg flag 'strict_dir' is True, then
        # only subdirectories with 'hf' or 'fm' in the name will be included. If
        # the flag is False, then all subdirectories will be included.
        if self.cfg['strict_dir']:
            recording_names = [i for i
                               in fme.list_subdirs(self.cfg['animal_directory'])
                               if 'hf' in i or 'fm' in i]
            
        elif not self.cfg['strict_dir']:
            recording_names = fme.list_subdirs(self.cfg['animal_directory'])

        # If the cfg file listed specific recordings that should be analyzed, then
        # ignore any of the subdirectories that are not in that list from the cfg file.
        if self.cfg['recording_list'] != []:
            recording_names = [i for i in recording_names
                               if i in self.cfg['recording_list']]

        # Eliminate recording names that are never used
        recording_names = [i for i in recording_names
                           if 'transfer' not in i and 'test' not in i]
        
        # Append the directory for the full session to the name of each
        # recording so that it is the complete file path for that recording.
        recording_paths = [os.path.join(self.cfg['animal_directory'], recording_name)
                           for recording_name in recording_names]
        
        # Zip it into a dictionary with the format {name: path}
        recordings_dict = dict(zip(recording_names, recording_paths))

        # Sort so that freely moving recordings are always handled before
        # head-fixed recordings.
        sorted_keys = sorted(recordings_dict, key=lambda x:('fm' not in x, x))
        self.recordings_dict = dict(zip(sorted_keys, [recordings_dict[k] for k in sorted_keys]))


    def clear_dlc(self):
        """ Delete all DeepLabCut hdf and pickle files in the session directory.

        This is only useful if you re-run DLC with a different model, and
        duplicate DLC files exist. The pipeline should pick the hdf file that
        was written to disk most recently, but this function can be run as an
        extra precaution.

        """
        # Get the list of hdf files in the session directory
        h5_list = fme.find('*DLC*.h5',self.cfg['animal_directory'])
        # And the pickle files, which are written by DLC but
        # not used by the pipeline.
        pickle_list = fme.find('*DLC*.pickle',self.cfg['animal_directory'])
        
        # Merge the lists
        file_list = h5_list + pickle_list
        
        # Then, iterate through the list and delete each file.
        for item in file_list:

            os.remove(item)


    def preprocessing(self):
        """ Preprocess all recordings in the session.
        """
        if self.cfg['delete_DLC_h5s']:
            self.clear_dlc()

        # get list of recordings from cfg file, or search subdirectories if none listed
        self.get_session_recordings()

        # iterate through recordings in the session
        for _, recording_path in self.recordings_dict.items():

            recording_name = fme.auto_recording_name(recording_path)

            print('preprocessing {} (path= {})'.format(recording_name, recording_path))

            # Skip this recording if it was acquired while the animal was transfered
            # between ball and arena
            if 'transfer' in recording_name or 'test' in recording_name:
                continue

            # get a list of cameras in the current recording
            recording_cams = []
            for p in ['REYE','LEYE','SIDE','TOP1','TOP2','TOP3','WORLD']:

                date_str = recording_name.split('_')[0]
                animal_str = recording_name.split('_')[1]
                rec_str = recording_name.split('_')[4:]

                if type(rec_str) == list:
                    rec_str = '_'.join(rec_str)

                if fme.find(recording_name+'_'+p+'.avi', recording_path) != []:
                    recording_cams.append(p)

                elif self.cfg['eye_crnrs_1st'] and (fme.find('{}_{}_*_{}_{}.avi'.format(date_str,
                                                  animal_str, rec_str, p), recording_path) != []):
                    recording_cams.append(p)

            for camname in recording_cams:

                if camname.lower() in ['reye','leye']:
                    
                    print(recording_name + ' for input: ' + camname)
                    ec = fme.Eyecam(self.cfg, recording_name, recording_path, camname)
                    ec.safe_process(show=True)
                
                elif camname.lower() in ['world']:
                    
                    print(recording_name + ' for input: ' + camname)
                    wc = fme.Worldcam(self.cfg, recording_name, recording_path, camname)
                    wc.safe_process(show=True)
                
                elif camname.lower() in ['top1','top2','top3'] and 'dark' not in recording_name:
                    
                    print(recording_name + ' for input: ' + camname)
                    tc = fme.Topcam(self.cfg, recording_name, recording_path, camname)
                    tc.safe_process(show=True)
                
                elif camname.lower() in ['side']:
                    
                    sc = fme.Sidecam(self.cfg, recording_name, recording_path, camname)
                    sc.safe_process(show=True)

            if self.cfg['run']['parameters']:

                if fme.find(recording_name+'_IMU.bin', recording_path) != []:

                    # For a freely moving recording, this will be the gyro/acc data
                    if 'fm' in recording_name:
                        print(recording_name + ' for input: IMU')
                        fme.Imu(self.cfg, recording_name, recording_path).process()

                    # For head fixed, it is the TTL signal from stim computer
                    elif 'hf' in recording_name:

                        print(recording_name + ' for input: TTL')
                        fme.TTL(self.cfg, recording_name, recording_path).process()

                if fme.find(recording_name+'_BALLMOUSE_BonsaiTS_X_Y.csv', recording_path) != []:

                    print(recording_name + ' for input: head-fixed running ball')
                    fme.RunningBall(self.cfg, recording_name, recording_path).process()


    def ephys_analysis(self):
        """ Analyze the ephys and stimulus data for all recordings.
        """
        
        self.get_session_recordings()
        
        for _, recording_path in self.recordings_dict.items():

            recording_name = fme.auto_recording_name(recording_path)
            
            if ('fm' in recording_name and 'light' in recording_name) or ('fm' in recording_name and 'light' not in recording_name and 'dark' not in recording_name):
                ephys = fme.FreelyMovingLight(self.cfg, recording_name, recording_path)
                ephys.analyze()
            
            elif 'fm' in recording_name and 'dark' in recording_name:
                ephys = fme.FreelyMovingDark(self.cfg, recording_name, recording_path)
                ephys.analyze()
           
            elif 'wn' in recording_name:
                ephys = fme.HeadFixedWhiteNoise(self.cfg, recording_name, recording_path)
                ephys.analyze()
            
            elif 'grat' in recording_name:
                # only drifting gratings
                ephys = fme.HeadFixedGratings(self.cfg, recording_name, recording_path)
                ephys.analyze()
            
            elif 'sp' in recording_name and ('noise' in recording_name or 'flash'):
                # this works for sparse noise of constant or random dT
                # also work w/ or w/out ISI
                ephys = fme.HeadFixedSparseNoise(self.cfg, recording_name, recording_path)
                ephys.analyze()
            
            elif 'revchecker' in recording_name:
                # does not work for recordings with an ISI
                ephys = fme.HeadFixedReversingCheckboard(self.cfg, recording_name, recording_path)
                ephys.analyze()


    def run_main(self):
        """ Run the preprocessing pipeline.
        """

        if self.cfg['run']['deinterlace'] or self.cfg['run']['undistort'] or self.cfg['run']['pose_estimation'] or self.cfg['run']['parameters']:
            self.preprocessing()

        if self.cfg['run']['stim_analysis']:
            self.ephys_analysis()

