"""
FreelyMovingEphys/src/run.py
"""
import yaml
import os
import pandas as pd

from fmEphys.utils.path import find, list_subdirs, auto_recording_name
from fmEphys.utils.ball import RunningBall
from fmEphys.utils.imu import Imu
from fmEphys.utils.eyecam import Eyecam
from fmEphys.utils.topcam import Topcam
from fmEphys.utils.worldcam import Worldcam
from fmEphys.utils.sidecam import Sidecam
from fmEphys.utils.freelymoving import FreelyMovingLight, FreelyMovingDark
from fmEphys.utils.headfixed import HeadFixedGratings, HeadFixedWhiteNoise, HeadFixedReversingCheckboard, HeadFixedSparseNoise

class Session:
    """ Preprocessing and analysis of an individual session.
    """
    def __init__(self, config_path, crop_for_dlc=False, filter_dlc_predictions=False, multianimal_top_project=False, likelihood_threshold=0.99,
                ephys_samprate=30000, eye_ellipticity_thresh=0.85, eye_dist_thresh_cm=4.1, eye_pxl_per_cm=24, ellipse_pts_needed_for_calibration=8,
                ellipse_pts_needed_for_eye=7, pts_needed_for_reflection=5, max_pupil_radius_pxls=50, imu_dwnsmpl=100, imu_samprate=30000, video_dwnsmpl=0.25,
                video_frames_to_save=3600, optical_mouse_pxls_to_cm=2840, optical_mouse_samprate_ms=200, optical_mouse_screen_center={'x': 960, 'y': 540},
                strict_likelihood_threshold=0.9999, rotate_eyecam=True, rotate_worldcam=True, flip_gyro_xy=False, do_rough_glm_fit=False, do_glm_model_preprocessing=True):
        """
        Parameters:
        config_path (str): path to config.yaml
        crop_for_dlc (bool): crop videos prior to pose estimation
        filter_dlc_predictions (bool): apply median filter to dlc outputs
        multianimal_top_project (bool): is the topdown camera's dlc network a multianimal project?
        likelihood_threshold (float): dlc points with confidence below this value will be set to NaN
        ephys_samprate (int): sample rate of ephys data
        eye_ellipticity_thresh (float): maximum ratio of ellipse shortaxis to longaxis during ellipse fit of pupil
        eye_dist_thresh_cm (float): maximum acceptable distance in cm that each frame's point can be from the mean position of that eye point across the recording
        eye_pxl_per_cm (int): scale factor for camera from pixels to cm on eye
        ellipse_pts_needed_for_calibration (int): only use eyecam frames for camera calibration which have this number of good points around the eye
        ellipse_pts_needed_for_eye (int): only use eyecam frames which have this number of good points around the eye
        pts_needed_for_reflection (int): only use eyecam frames which have this number of good points around the reflection of the ir light
        max_pupil_radius_pxls (int): maximum acceptable number of pixels for radius of the pupil
        imu_dwnsmpl (int): factor by which to downsample imu data
        imu_samprate (int): sample rate of imu data
        video_dwnsmpl (float): factor by which to downsample videos before packing into .nc
        video_frames_to_save (int): number of frames to write to dianostic .avi
        optical_mouse_pxls_to_cm (int): pixel to cm conversion factor for optical mouse of running ball
        optical_mouse_samprate_ms (int): optical mouse sample rate in ms
        optical_mouse_screen_center (dict): coordinates that the optical mouse resets to in x and y
        strict_likelihood_threshold (float): a value higher than the given likelihood_threshold for tracking the IR spot
        rotate_eyecam (bool): if eyecam is being deinterlaced, rotate by 180deg
        rotate_worldcam (bool): if worldcam is being deinteralced, rotate by 180deg
        """
        # read config file
        with open(config_path, 'r') as infile:
            self.config = yaml.load(infile, Loader=yaml.FullLoader)

        if 'internals' not in self.config.keys():
            self.config['internals'] = {}

        self.config['internals'] = dict(self.config['internals'], **{
            'crop_for_dlc': crop_for_dlc,
            'filter_dlc_predictions': filter_dlc_predictions,
            'multianimal_top_project': multianimal_top_project,
            'likelihood_threshold': likelihood_threshold,
            'ephys_samprate': ephys_samprate,
            'eye_ellipticity_thresh': eye_ellipticity_thresh,
            'eye_dist_thresh_cm': eye_dist_thresh_cm,
            'eye_pxl_per_cm': eye_pxl_per_cm,
            'ellipse_pts_needed_for_calibration': ellipse_pts_needed_for_calibration,
            'ellipse_pts_needed_for_eye': ellipse_pts_needed_for_eye,
            'pts_needed_for_reflection': pts_needed_for_reflection,
            'max_pupil_radius_pxls': max_pupil_radius_pxls,
            'imu_dwnsmpl': imu_dwnsmpl,
            'imu_samprate': imu_samprate,
            'video_dwnsmpl': video_dwnsmpl,
            'video_frames_to_save': video_frames_to_save,
            'optical_mouse_pxls_to_cm': optical_mouse_pxls_to_cm,
            'optical_mouse_samprate_ms': optical_mouse_samprate_ms,
            'optical_mouse_screen_center': optical_mouse_screen_center,
            'strict_likelihood_threshold': strict_likelihood_threshold,
            'rotate_eyecam': rotate_eyecam,
            'rotate_worldcam': rotate_worldcam,
            'flip_gyro_xy': flip_gyro_xy,
            'do_rough_glm_fit': do_rough_glm_fit,
            'do_glm_model_preprocessing': do_glm_model_preprocessing
        })

    def get_session_recordings(self):
        if self.config['internals']['follow_strict_directory_naming']:
            recording_names = [i for i in list_subdirs(self.config['animal_directory']) if 'hf' in i or 'fm' in i]
        elif not self.config['internals']['follow_strict_directory_naming']:
            recording_names = list_subdirs(self.config['animal_directory'])
        if self.config['options']['recording_list'] != []:
            recording_names = [i for i in recording_names if i in self.config['options']['recording_list']]
        recording_names = [i for i in recording_names if 'transfer' not in i and 'test' not in i]
        recording_paths = [os.path.join(self.config['animal_directory'], recording_name) for recording_name in recording_names]
        recordings_dict = dict(zip(recording_names, recording_paths))
        # sort dictionary of {name: path} so freely-moving recordings are always handled first
        sorted_keys = sorted(recordings_dict, key=lambda x:('fm' not in x, x))
        self.recordings_dict = dict(zip(sorted_keys, [recordings_dict[k] for k in sorted_keys]))

    def clear_dlc(self):
        h5_list = find('*DLC*.h5',self.config['animal_directory'])
        pickle_list = find('*DLC*.pickle',self.config['animal_directory'])
        file_list = h5_list + pickle_list
        for item in file_list:
            os.remove(item)

    def preprocessing(self):
        if self.config['internals']['clear_dlc']:
            self.clear_dlc()

        # get list of recordings from config file, or search subdirectories if none listed
        self.get_session_recordings()

        # iterate through recordings in the session
        for _, recording_path in self.recordings_dict.items():

            recording_name = auto_recording_name(recording_path)

            print('preprocessing {} (path= {})'.format(recording_name, recording_path))

            # skip this recording if it was acquired while the animal was transfered between ball and arena
            if 'transfer' in recording_name or 'test' in recording_name:
                continue

            # get a list of cameras in the current recording
            recording_cams = []
            for p in ['REYE','LEYE','Reye','Leye','Side','SIDE','TOP1','TOP2','TOP3','WORLD','World']:
                date_str = recording_name.split('_')[0]
                animal_str = recording_name.split('_')[1]
                rec_str = recording_name.split('_')[3]
                if find(recording_name+'_'+p+'.avi', recording_path) != []:
                    recording_cams.append(p)
                elif self.config['internals']['eye_corners_first'] and (find('{}_{}_*_{}_{}.avi'.format(date_str, animal_str, rec_str, p), recording_path) != []):
                    recording_cams.append(p)

            for camname in recording_cams:
                if camname.lower() in ['reye','leye']:
                    print(recording_name + ' for input: ' + camname)
                    ec = Eyecam(self.config, recording_name, recording_path, camname)
                    ec.safe_process(show=True)
                elif camname.lower() in ['world']:
                    print(recording_name + ' for input: ' + camname)
                    wc = Worldcam(self.config, recording_name, recording_path, camname)
                    wc.safe_process(show=True)
                elif camname.lower() in ['top1','top2','top3'] and 'dark' not in recording_name:
                    print(recording_name + ' for input: ' + camname)
                    tc = Topcam(self.config, recording_name, recording_path, camname)
                    tc.safe_process(show=True)
                elif camname.lower() in ['side']:
                    sc = Sidecam(self.config, recording_name, recording_path, camname)
                    sc.safe_process(show=True)
            if find(recording_name+'_IMU.bin', recording_path) != []:
                print(recording_name + ' for input: IMU')
                imu = Imu(self.config, recording_name, recording_path)
                imu.process()
            if find(recording_name+'_BALLMOUSE_BonsaiTS_X_Y.csv', recording_path) != []:
                print(recording_name + ' for input: head-fixed running ball')
                rb = RunningBall(self.config, recording_name, recording_path)
                rb.process()

    def ephys_analysis(self):
        self.get_session_recordings()
        for _, recording_path in self.recordings_dict.items():
            recording_name = auto_recording_name(recording_path)
            if ('fm' in recording_name and 'light' in recording_name) or ('fm' in recording_name and 'light' not in recording_name and 'dark' not in recording_name):
                ephys = FreelyMovingLight(self.config, recording_name, recording_path)
                ephys.analyze()
            elif 'fm' in recording_name and 'dark' in recording_name:
                ephys = FreelyMovingDark(self.config, recording_name, recording_path)
                ephys.analyze()
            elif 'wn' in recording_name:
                ephys = HeadFixedWhiteNoise(self.config, recording_name, recording_path)
                ephys.analyze()
            elif 'grat' in recording_name:
                ephys = HeadFixedGratings(self.config, recording_name, recording_path)
                ephys.analyze()
            elif 'sp' in recording_name and 'noise' in recording_name:
                ephys = HeadFixedSparseNoise(self.config, recording_name, recording_path)
                ephys.analyze()
            elif 'revchecker' in recording_name:
                ephys = HeadFixedReversingCheckboard(self.config, recording_name, recording_path)
                ephys.analyze()

    def run_main(self):
        if self.config['main']['deinterlace'] or self.config['main']['undistort'] or self.config['main']['pose_estimation'] or self.config['main']['parameters']:
            self.preprocessing()
        if self.config['main']['ephys']:
            self.ephys_analysis()

# class Batch(Session):
#     """ Preprocessing and analysis of a batch of sessions.
#     """
#     def __init__(self, config_path, metadata_path):
#         super().__init__()
#         # read metadata csv
#         self.metadata = pd.read_csv(metadata_path)
#         # batch options
#         self.force_probe_name = None
#         self.force_flip_gx_gy = None
#         self.force_drop_slow_frames = None

#     def config_overwrite(self):
#         if self.force_probe_name is not None:
#             self.config['ephys_analysis']['probe_type'] = self.force_probe_name
#         if self.force_flip_gx_gy is not None:
#             self.config['parameters']['imu']['flip_gx_gy'] = self.force_flip_gx_gy
#         if self.force_drop_slow_frames is not None:
#             self.config['parameters']['drop_slow_frames'] = self.drop_slow_frames
#     def process_batch(self)