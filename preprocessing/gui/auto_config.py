"""
auto_config.py

automaticlly build a config file from inputs to the GUI

Jan. 15, 2021
"""

import json, os

def write_config(user_inputs):

    # if no DLC information was provided, skip DLC
    if len([key for (key,val) in user_inputs['cam_inputs'].items() if key != 'None']) > 0:
        run_dlc = True
    else:
        run_dlc = False

    if user_inputs['undistort_vids_use_paths_from_current_session'] is True:
        world_npz_read_dir = os.path.join(user_inputs['npz_save_path'], user_inputs['world_npz_save_name'])
        top_npz_read_dir = os.path.join(user_inputs['npz_save_path'], user_inputs['top_npz_save_name'])
    else:
        world_npz_read_dir = user_inputs['world_npz_read_dir']
        top_npz_read_dir = user_inputs['top_npz_read_dir']

    # if no cameras were provided for DLC, but they were provided in params
    # this assumes that params will always match DLC, but that DLC need not always match params?
    try:
        del user_inputs['cam_inputs']['None']
    except KeyError:
        pass
    if user_inputs['cam_inputs'] == {}:
        user_inputs['cam_inputs'] = dict(zip(key,'') for key in user_inputs['param_cams'] if key is not None)

    internal_config = {
        'data_path': user_inputs['data_path'],
        'steps_to_run': {
            'deinter': user_inputs['deinterlace'],
            'get_cam_calibration_params': user_inputs['get_checker_calib'],
            'undistort_recording': user_inputs['undistort_vids'],
            'dlc': run_dlc,
            'params': user_inputs['run_params'],
            'addtl_params': user_inputs['run_addtl_params']
        },
        'cams': user_inputs['cam_inputs'],
        'calibration': {
            'eye_LED_config': user_inputs['ledE'],
            'world_LED_config': user_inputs['ledW'],
            'world_checker_vid': user_inputs['world_checker_avi_path'],
            'world_checker_npz': world_npz_read_dir,
            'top_checker_vid': user_inputs['top_checker_avi_path'],
            'top_checker_npz': top_npz_read_dir
        },
        'LED_dir_name': user_inputs['LED_dir_name'],
        'flip_eye_during_deinter': user_inputs['flip_eyecam'],
        'flip_world_during_deinter': user_inputs['flip_worldcam'],
        'crop_for_dlc': user_inputs['crop_vids'],
        'multianimal_TOP': user_inputs['multiTOP'],
        'lik_thresh': [float(user_inputs['lik_thresh']) if user_inputs['lik_thresh'] is not None else user_inputs['lik_thresh']][0],
        'lik_thresh_strict': [float(user_inputs['strict_lik_thresh']) if user_inputs['strict_lik_thresh'] is not None else user_inputs['strict_lik_thresh']][0],
        'has_ephys': user_inputs['has_ephys'],
        'cricket': user_inputs['cricket'],
        'tear': user_inputs['tear'],
        'pxl_thresh': [float(user_inputs['pxl_thresh']) if user_inputs['pxl_thresh'] is not None else user_inputs['pxl_thresh']][0],
        'ell_thresh': [float(user_inputs['ell_thresh']) if user_inputs['ell_thresh'] is not None else user_inputs['ell_thresh']][0],
        'eye_dist_thresh_cm': [float(user_inputs['eye_dist_thresh_cm']) if user_inputs['eye_dist_thresh_cm'] is not None else user_inputs['eye_dist_thresh_cm']][0],
        'eyecam_pxl_per_cm': [float(user_inputs['eyecam_pxl_per_cm']) if user_inputs['eyecam_pxl_per_cm'] is not None else user_inputs['eyecam_pxl_per_cm']][0],
        'save_avi_vids': user_inputs['save_avi_vids'],
        'num_save_frames': [int(user_inputs['num_save_frames']) if user_inputs['num_save_frames'] is not None else user_inputs['num_save_frames']][0],
        'save_figs': user_inputs['save_figs'],
        'save_nc_vids': user_inputs['save_nc_vids'],
        'use_BonsaiTS': user_inputs['use_BonsaiTS'],
        'range_radius': [float(user_inputs['range_radius']) if user_inputs['range_radius'] is not None else user_inputs['range_radius']][0],
        'world_interp_method': user_inputs['world_interp_method'],
        'num_ellipse_pts_needed': [int(user_inputs['num_ellipse_pts_needed']) if user_inputs['num_ellipse_pts_needed'] is not None else user_inputs['num_ellipse_pts_needed']][0],
        'dwnsmpl': [float(user_inputs['dwnsmpl']) if user_inputs['dwnsmpl'] is not None else user_inputs['dwnsmpl']][0],
        'ephys_sample_rate': [int(user_inputs['ephys_sample_rate']) if user_inputs['ephys_sample_rate'] is not None else user_inputs['ephys_sample_rate']][0],
        'run_pupil_rotation': user_inputs['run_pupil_rotation'],
        'run_top_angles': user_inputs['run_top_angles'],
        'run_with_form_time': user_inputs['run_with_form_time'],
        'optical_mouse_screen_center': {
            'x': [int(user_inputs['hf_ball_x']) if user_inputs['hf_ball_x'] is not None else user_inputs['hf_ball_x']][0],
            'y': [int(user_inputs['hf_ball_y']) if user_inputs['hf_ball_y'] is not None else user_inputs['hf_ball_y']][0]
        },
        'optical_mouse_pix2cm': [int(user_inputs['optical_mouse_pix2cm']) if user_inputs['optical_mouse_pix2cm'] is not None else user_inputs['optical_mouse_pix2cm']][0],
        'optical_mouse_sample_rate_ms': [int(user_inputs['optical_mouse_sample_rate_ms']) if user_inputs['optical_mouse_sample_rate_ms'] is not None else user_inputs['optical_mouse_sample_rate_ms']][0],
        'imu_sample_rate': [int(user_inputs['imu_sample_rate']) if user_inputs['imu_sample_rate'] is not None else user_inputs['imu_sample_rate']][0],
        'imu_downsample': [int(user_inputs['imu_downsample']) if user_inputs['imu_downsample'] is not None else user_inputs['imu_downsample']][0]
    }

    print('writing config file to '+ str(user_inputs['data_path']))

    print(internal_config)

    with open(os.path.join(user_inputs['data_path'], 'preprocessing_config.json'),'w') as fp:
        json.dump(internal_config, fp)

    return os.path.join(user_inputs['data_path'], 'preprocessing_config.json')