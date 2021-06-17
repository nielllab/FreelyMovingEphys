"""
config.py

config and user input utilities
"""
import os, json, yaml
import platform

def set_preprocessing_config_defaults(novel_config):
    """
    set default values for config file, if any are missing
    defaults are read in from /FreelyMovingEphys/example_configs/preprocessing_config.json
    changing the default values, or adding new config options should be done in that json in the /example_configs/ directory
    INPUTS:
        novel_config: dictionary with config options, structured largely like the default config .json in /example_configs/
    OUTPUTS:
        novel_config: same config as input, with any missing values filled in with the defaults in /example_configs/
    """
    if platform.system() == 'Linux':
        # on linux, the file path needs to be found differently
        default_json_path = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/example_configs/preprocessing_config.json'
    else:
        # get the path of the default json config file in this repository, relative to util/config.py
        # this assumes windows filepaths
        default_json_path = '/'.join(os.path.abspath(__file__).split('\\')[:-2]) + '/example_configs/preprocessing_config.json'
    with open(default_json_path, 'r') as fp:
        default_config = json.load(fp)
    # iterate through keys in the dictionary loaded in from json
    for default_key in default_config:
        # if a key does not exist, add the value in the default config file
        if default_key not in novel_config:
            novel_config[default_key] = default_config[default_key]
            print('filling default value for config option '+default_key +' -- value will be '+str(default_config[default_key]))
    
    return novel_config

def str_to_bool(value):
    """
    parse strings to read argparse flag entries in as True/False
    INPUTS:
        value: input value
    OUTPUTS:
        either True, False, or raises error
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def open_config(input_path):
    """
    open config file, either a .json or a .yaml
    formats .json and .yaml into identical dictionaries
    INPUTS
        input_path: path to config file
    OUTPUTS
        config: options and paths dictionary to be used in preprocessing (and ephys analysis, if the starting file is a .yaml)
    """
    if os.path.splitext(input_path)[1] == '.json':
        with open(input_path, 'r') as fp:
            config = json.load(fp)
        # fill defaults
        config = set_preprocessing_config_defaults(config)
    elif os.path.splitext(input_path)[1] == '.yaml':
        with open(input_path, 'r') as infile:
            y = yaml.load(infile, Loader=yaml.FullLoader)
        # always try to find worldcam if it's there
        y['pose_estimation']['projects']['WORLD'] = ''
        # build a working config
        config = {
            'data_path': y['animal_dir'],
            'steps_to_run':{
                'deinter': y['deinterlace']['run_deinter'],
                'img_correction': y['img_correction']['run_img_correction'],
                'get_cam_calibration_params': y['calibration']['run_cam_calibration'],
                'undistort_recording': y['calibration']['undistort_recordings'],
                'dlc': y['pose_estimation']['run_dlc'],
                'params': y['parameters']['run_params'],
                'addtl_params': y['ir_spot_in_space']['run_is_spot_in_space'],
                'ephys': y['ephys_analysis']['run_ephys_analysis']
            },
            'cams': y['pose_estimation']['projects'],
            'calibration':{
                'eye_LED_config': y['ir_spot_in_space']['LED_eye_view_config'],
                'world_LED_config': y['ir_spot_in_space']['LED_world_view_config'],
                'world_checker_vid': y['calibration']['world_checker_vid'],
                'world_checker_npz': y['calibration']['world_checker_npz'],
                'top_checker_vid': y['calibration']['top_checker_vid'],
                'top_checker_npz': y['calibration']['top_checker_npz']
            },
            'LED_dir_name': y['ir_spot_in_space']['ir_spot_in_space_dir_name'],
            'flip_eye_during_deinter': y['deinterlace']['flip_eye_during_deinter'],
            'flip_world_during_deinter': y['deinterlace']['flip_world_during_deinter'],
            'apply_auto_gamma': y['img_correction']['apply_auto_gamma'],
            'crop_for_dlc': y['pose_estimation']['crop_for_dlc'],
            'multianimal_TOP': y['pose_estimation']['multianimal_top_project'],
            'filter_dlc_predictions': y['pose_estimation']['filter_dlc_predictions'],
            'lik_thresh': y['parameters']['lik_thresh'],
            'lik_thresh_strict': y['ir_spot_in_space']['lik_thresh_strict'],
            'has_ephys': False,
            'has_cricket_labeled': y['pose_estimation']['has_cricket_labeled'],
            'has_tear_labeled': y['pose_estimation']['has_tear_labeled'],
            'has_ir_spot_labeled': y['pose_estimation']['has_ir_spot_labeled'],
            'spot_subtract': y['parameters']['eyes']['spot_subtract'],
            'pxl_thresh': y['parameters']['eyes']['pxl_thresh'],
            'ell_thresh': y['parameters']['eyes']['ell_thresh'],
            'eye_dist_thresh_cm': y['parameters']['eyes']['eyecam_pxl_per_cm'],
            'eyecam_pxl_per_cm': y['parameters']['eyes']['eye_dist_thresh_cm'],
            'save_avi_vids': y['parameters']['outputs_and_visualization']['save_avi_vids'],
            'num_save_frames': y['parameters']['outputs_and_visualization']['num_save_frames'],
            'save_nc_vids': y['parameters']['outputs_and_visualization']['save_nc_vids'],
            'save_figs': y['parameters']['outputs_and_visualization']['save_figs'],
            'use_BonsaiTS': True,
            'range_radius': 10,
            'world_interp_method': 'linear',
            'calib_ellipse_pts_needed': y['parameters']['eyes']['calib_ellipse_pts_needed'],
            'num_ellipse_pts_needed': y['parameters']['eyes']['num_ellipse_pts_needed'],
            'num_ir_spot_pts_needed': y['parameters']['eyes']['num_ir_spot_pts_needed'],
            'eye_fig_pts_dwnspl': y['parameters']['outputs_and_visualization']['eye_fig_pts_dwnspl'],
            'dwnsmpl': y['parameters']['outputs_and_visualization']['dwnsmpl'],
            'ephys_sample_rate': y['parameters']['ephys']['ephys_sample_rate'],
            'run_pupil_rotation': y['parameters']['eyes']['get_eye_omega'],
            'run_top_angles': y['parameters']['topdown']['get_top_thetas'],
            'run_with_form_time': y['parameters']['follow_strict_naming'],
            'optical_mouse_screen_center': y['parameters']['running_wheel']['optical_mouse_screen_center'],
            'optical_mouse_pix2cm': y['parameters']['running_wheel']['optical_mouse_pix2cm'],
            'optical_mouse_sample_rate_ms': y['parameters']['running_wheel']['optical_mouse_sample_rate_ms'],
            'imu_sample_rate': y['parameters']['imu']['imu_sample_rate'],
            'imu_downsample': y['parameters']['imu']['imu_downsample'],
            'unit2highlight': y['ephys_analysis']['unit_to_highlight'],
            'probe': y['ephys_analysis']['probe_type'],
            'write_ephys_vids': y['ephys_analysis']['write_videos'],
            'specific_ephys_recs': y['ephys_analysis']['recording_list']
        }

    return config