"""
FreelyMovingEphys/src/run.py
"""
import os
import yaml

import fmEphys
class Session:
    """ Preprocessing and analysis of an individual session.
    """
    def __init__(self, cfg_path):
        """

        """
        # read config file
        with open(cfg_path, 'r') as infile:
            _tmp_cfg = yaml.load(infile, Loader=yaml.FullLoader)

        internals_path = os.path.join(os.path.split(__file__)[0], 'internals.yml')
        
        with open(internals_path, 'r') as infile:
            internals_dict = yaml.load(infile, Loader=yaml.FullLoader)

        # merge, with a preference for anything specified in cfg
        # i.e., a key that exists in both dictionaries will be set to the value it has in cfg.yml
        
        # if a key exists in x and y, the value for that key in y will be the one represented in z
        # z = {**x, **y}

        self.cfg = {**internals_dict, **_tmp_cfg}

    def get_session_recordings(self):
        if self.cfg['strict_dir']:
            recording_names = [i for i in fmEphys.list_subdirs(self.cfg['animal_directory']) if 'hf' in i or 'fm' in i]
        elif not self.cfg['strict_dir']:
            recording_names = fmEphys.list_subdirs(self.cfg['animal_directory'])
        if self.cfg['recording_list'] != []:
            recording_names = [i for i in recording_names if i in self.cfg['recording_list']]
        recording_names = [i for i in recording_names if 'transfer' not in i and 'test' not in i]
        recording_paths = [os.path.join(self.cfg['animal_directory'], recording_name) for recording_name in recording_names]
        recordings_dict = dict(zip(recording_names, recording_paths))
        # sort dictionary of {name: path} so freely-moving recordings are always handled first
        sorted_keys = sorted(recordings_dict, key=lambda x:('fm' not in x, x))
        self.recordings_dict = dict(zip(sorted_keys, [recordings_dict[k] for k in sorted_keys]))

    def clear_dlc(self):
        h5_list = fmEphys.find('*DLC*.h5',self.cfg['animal_directory'])
        pickle_list = fmEphys.find('*DLC*.pickle',self.cfg['animal_directory'])
        file_list = h5_list + pickle_list
        for item in file_list:
            os.remove(item)

    def preprocessing(self):
        if self.cfg['delete_DLC_h5s']:
            self.clear_dlc()

        # get list of recordings from cfg file, or search subdirectories if none listed
        self.get_session_recordings()

        # iterate through recordings in the session
        for _, recording_path in self.recordings_dict.items():

            recording_name = fmEphys.auto_recording_name(recording_path)

            print('preprocessing {} (path= {})'.format(recording_name, recording_path))

            # skip this recording if it was acquired while the animal was transfered between ball and arena
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
                if fmEphys.find(recording_name+'_'+p+'.avi', recording_path) != []:
                    recording_cams.append(p)
                elif self.cfg['eye_crnrs_1st'] and (fmEphys.find('{}_{}_*_{}_{}.avi'.format(date_str, animal_str, rec_str, p), recording_path) != []):
                    recording_cams.append(p)

            for camname in recording_cams:
                if camname.lower() in ['reye','leye']:
                    print(recording_name + ' for input: ' + camname)
                    ec = fmEphys.Eyecam(self.cfg, recording_name, recording_path, camname)
                    ec.safe_process(show=True)
                elif camname.lower() in ['world']:
                    print(recording_name + ' for input: ' + camname)
                    wc = fmEphys.Worldcam(self.cfg, recording_name, recording_path, camname)
                    wc.safe_process(show=True)
                elif camname.lower() in ['top1','top2','top3'] and 'dark' not in recording_name:
                    print(recording_name + ' for input: ' + camname)
                    tc = fmEphys.Topcam(self.cfg, recording_name, recording_path, camname)
                    tc.safe_process(show=True)
                elif camname.lower() in ['side']:
                    sc = fmEphys.Sidecam(self.cfg, recording_name, recording_path, camname)
                    sc.safe_process(show=True)
            if self.cfg['run']['parameters']:
                if fmEphys.find(recording_name+'_IMU.bin', recording_path) != []:
                    print(recording_name + ' for input: IMU')
                    imu = fmEphys.Imu(self.cfg, recording_name, recording_path)
                    imu.process()
                if fmEphys.find(recording_name+'_BALLMOUSE_BonsaiTS_X_Y.csv', recording_path) != []:
                    print(recording_name + ' for input: head-fixed running ball')
                    rb = fmEphys.RunningBall(self.cfg, recording_name, recording_path)
                    rb.process()

    def ephys_analysis(self):
        self.get_session_recordings()
        for _, recording_path in self.recordings_dict.items():
            recording_name = fmEphys.auto_recording_name(recording_path)
            if ('fm' in recording_name and 'light' in recording_name) or ('fm' in recording_name and 'light' not in recording_name and 'dark' not in recording_name):
                ephys = fmEphys.FreelyMovingLight(self.cfg, recording_name, recording_path)
                ephys.analyze()
            elif 'fm' in recording_name and 'dark' in recording_name:
                ephys = fmEphys.FreelyMovingDark(self.cfg, recording_name, recording_path)
                ephys.analyze()
            elif 'wn' in recording_name:
                ephys = fmEphys.HeadFixedWhiteNoise(self.cfg, recording_name, recording_path)
                ephys.analyze()
            elif 'grat' in recording_name:
                ephys = fmEphys.HeadFixedGratings(self.cfg, recording_name, recording_path)
                ephys.analyze()
            elif 'sp' in recording_name and 'noise' in recording_name:
                ephys = fmEphys.HeadFixedSparseNoise(self.cfg, recording_name, recording_path)
                ephys.analyze()
            elif 'revchecker' in recording_name:
                ephys = fmEphys.HeadFixedReversingCheckboard(self.cfg, recording_name, recording_path)
                ephys.analyze()

    def run_main(self):
        if self.cfg['run']['deinterlace'] or self.cfg['run']['undistort'] or self.cfg['run']['pose_estimation'] or self.cfg['run']['parameters']:
            self.preprocessing()
        if self.cfg['run']['stim_analysis']:
            self.ephys_analysis()

# class Batch(Session):
#     """ Preprocessing and analysis of a batch of sessions.
#     """
#     def __init__(self, cfg_path, metadata_path):
#         super().__init__()
#         # read metadata csv
#         self.metadata = pd.read_csv(metadata_path)
#         # batch options
#         self.force_probe_name = None
#         self.force_flip_gx_gy = None
#         self.force_drop_slow_frames = None

#     def cfg_overwrite(self):
#         if self.force_probe_name is not None:
#             self.cfg['ephys_analysis']['probe_type'] = self.force_probe_name
#         if self.force_flip_gx_gy is not None:
#             self.cfg['parameters']['imu']['flip_gx_gy'] = self.force_flip_gx_gy
#         if self.force_drop_slow_frames is not None:
#             self.cfg['parameters']['drop_slow_frames'] = self.drop_slow_frames
#     def process_batch(self)