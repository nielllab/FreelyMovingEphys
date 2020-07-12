#####################################################################################
"""
eye_calibration.py

Safety check plots of eye ellipse parameters.
This is NOT currently used in by load_from_DLC.py or associated functions, and isn't yet finished being written.

Adapted from /niell-lab-analysis/freely moving/EyeCameraCalc1.m

last modified: June 24, 2020
"""
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np

def plot_check_eye_calibration(all_eye_ellipses, all_eye_dlc_pts, trial_id_list, side, savepath_input):
    '''
    Get how well the eye tracking worked, and create visualizations of calibration.
    '''
    for trial_num in range(0, len(trial_id_list)):
        current_trial_name = trial_id_list[trial_num]
        if all_eye_ellipses.sel(trial=current_trial_name) is not None:
            ellipse_data = all_eye_ellipses.sel(trial=current_trial_name)

            timestart = ellipse_data['time_start'].values
            timeend = ellipse_data['time_end'].values

            ellipse_data_crop = ellipse_data.sel(time=slice(timestart, timeend))

            thetas = ellipse_data_crop.sel(ellipse_params='theta').values
            phis = ellipse_data_crop.sel(ellipse_params='phi').values
            longaxes = ellipse_data_crop.sel(ellipse_params='longaxis_all').values
            shortaxes = ellipse_data_crop.sel(ellipse_params='shortaxis_all').values
            camcenter = (ellipse_data_crop['cam_center_x'].values, ellipse_data_crop['cam_center_y'].values)

            ellipticity = shortaxes / longaxes
            pix2deg_scale = np.nansum(np.sqrt(1 - (ellipticity) ** 2 * np.linalg.norm(np.concatenate([np.array(thetas), np.array(phis)]).T - camcenter, 2, 1).T) / np.nansum(1 - (ellipticity) ** 2))

            thetaphi_diff = np.concatenate(thetas, phis).T - camcenter
            xvals = np.linalg.norm(thetaphi_diff, ord=None)
            yvals = pix2deg_scale * np.sqrt(1-(shortaxes) / longaxes)**2
            plt.figure(figsize=(10,10))
            plt.plot(xvals, yvals, 'k.')
            plt.savefig(savepath_input + '/' + current_trial_name + '/' + str(side) + '_side_ellipse_calibration.png', dpi=300)
            plt.close()
