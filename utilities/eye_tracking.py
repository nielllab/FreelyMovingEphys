#####################################################################################
"""
eye_tracking.py of FreelyMovingEphys

Function eye_angles() takes in xarray DataArray of left and right eye videos. Will run
with one or both eyes fine. Data are thresholded by likelihood, turned into
pandas arrays, and passed to calc_ellipse() which gets out the least squares estimate
for a 2D ellipse. The parameters of the ellipse estimated by calc_ellipse() is
cleaned up by preen_then_get_eye_angles(), and through that, passed to get_eye_angles()
to extract from the ellipse parameters the angle of the mouse's eye for each frame.

Adapted from code by Elliott Abe, DLCEyeVids.py, especially for the functions
get_eye_angles(), preen_then_get_eye_angles(), and calc_ellipse().

last modified: June 18, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################

import pandas as pd
import numpy as np
from skimage import measure
import xarray as xr
import matplotlib.pyplot as plt

from utilities.data_cleaning import split_xyl

####################################################
def get_eye_angles(ellipseparams):
    R = np.linspace(0,2*np.pi,100)
    longaxis_all = np.maximum(ellipseparams[:,2],ellipseparams[:,3])
    shortaxis_all = np.minimum(ellipseparams[:,2],ellipseparams[:,3])
    Ellipticity = shortaxis_all/longaxis_all
    lis, = np.where(Ellipticity<.9)
    A = np.vstack([np.cos(ellipseparams[lis,4]),np.sin(ellipseparams[lis,4])])
    b = np.expand_dims(np.diag(A.T@np.squeeze(ellipseparams[lis,0:2].T)),axis=1)
    CamCent = np.linalg.inv(A@A.T)@A@b
    longaxis = np.squeeze(np.maximum(ellipseparams[lis,2],ellipseparams[lis,3]))
    shortaxis = np.squeeze(np.minimum(ellipseparams[lis,2],ellipseparams[lis,3]))
    Ellipticity = shortaxis/longaxis
    scale = np.sum(np.sqrt(1-(Ellipticity)**2)*(np.linalg.norm(ellipseparams[lis,0:2]-CamCent.T,axis=1)))/np.sum(1-(Ellipticity)**2);
    temp = (ellipseparams[:,0]-CamCent[0])/scale
    theta = np.arcsin(temp)
    phi = np.arcsin((ellipseparams[:,1]-CamCent[1])/np.cos(theta)/scale)
    return theta, phi, longaxis_all, shortaxis_all, CamCent

####################################################
def preen_then_get_eye_angles(ellipseparams, pxl_thresh):
    bdfit2, temp = np.where(ellipseparams[:, 2:4] > pxl_thresh)
    eparams = pd.DataFrame(ellipseparams)
    eparams.iloc[bdfit2, :] = np.nan
    eparams = eparams.interpolate(method='linear', limit_direction='both', axis=0)
    ellipseparams[bdfit2, :] = eparams.iloc[bdfit2, :]
    # run get_eye_angles on the cleaned data
    theta, phi, longaxis_all, shortaxis_all, CamCent = get_eye_angles(ellipseparams)
    return theta, phi, longaxis_all, shortaxis_all, CamCent

####################################################
def calc_ellipse(num_frames, x_vals, y_vals, pxl_thresh):
    emod = measure.EllipseModel()
    # create an empty array to be populated by the five outputs of EllipseModel()
    ellipseparams = np.empty((0, 5))
    # get list of all timestamps
    timestamp_list = x_vals.index.values
    # index through each frame and stack the ellipse parameters
    for timestamp in timestamp_list:
        try:
            # first the ellipse
            x_block = x_vals.loc[timestamp, :]
            y_block = y_vals.loc[timestamp, :]
            xy = np.column_stack((x_block, y_block))
            if emod.estimate(xy) is True:
                params_raw = np.array(emod.params)
                params_expanded = np.expand_dims(params_raw, axis=0)
                ellipseparams = np.append(ellipseparams, params_expanded, axis=0)
        except KeyError:
            # if the timestamp cannot be found, add a filler entry of parameters
            ellipseparams = np.append(ellipseparams, np.empty((0, 5)), axis=0)

    theta, phi, longaxis_all, shortaxis_all, CamCent = preen_then_get_eye_angles(ellipseparams, pxl_thresh)

    return theta, phi, longaxis_all, shortaxis_all, CamCent

####################################################
def eye_angles(eye_data_input, eye_names, trial_id_list, savepath_input, all_trial_time, showfig=False, savefig=False, thresh=0.99, pxl_thresh=50, side='left'):
    # prepares data for use with Elliott's get_eye_angles
    # runs on one eye at a time, but can run on both if needed
    # pxl_thresh is the max number of pixels for radius of pupil
    # thresh is the liklihood threshold
    for trial_num in range(0, len(trial_id_list)):
        current_trial_name = trial_id_list[trial_num]
        if eye_data_input.sel(trial=current_trial_name) is not None:
            eye_data = eye_data_input.sel(trial=current_trial_name)

            x_vals, y_vals, likeli_vals = split_xyl(eye_names, eye_data, thresh)

            # get the number of frames
            num_frames = len(x_vals)

            # make a plot of an example frame, showing the points of the ellipse
            # a way to make sure the data are somewhat elliptical
            if savefig is True:
                timestamp_list = x_vals.index.values
                frame_slice = timestamp_list[3]
                x_to_plot = x_vals.loc[[frame_slice]]
                y_to_plot = y_vals.loc[[frame_slice]]
                plt.figure(figsize=(15,10))
                plt.scatter(x_to_plot, y_to_plot, color='r')
                plt.title('dlc points at time ' + str(frame_slice) + ' of ' + str(side) + ' eye of ' + str(current_trial_name))
                plt.savefig(savepath_input + '/' + current_trial_name + '/' + 'dlc_eye_pts_at_time_' + str(frame_slice) + '.png', dpi=300)
                plt.close()

            # get the ellipse parameters out of the point positional data
            theta, phi, longaxis_all, shortaxis_all, CamCent = calc_ellipse(num_frames, x_vals, y_vals, pxl_thresh)

            if savefig is True:
                plt.subplots(2, 1, figsize=(30,20))
                plt.subplot(211)
                plt.plot(theta * 180 / np.pi)
                plt.xlabel('frame')
                plt.ylabel('angle')
                plt.title('theta for ' + str(side) + ' eye of ' + str(current_trial_name))
                plt.subplot(212)
                plt.plot(phi * 180 / np.pi)
                plt.xlabel('frame')
                plt.ylabel('angle')
                plt.title('phi for ' + str(side) + ' eye of ' + str(current_trial_name))
                plt.savefig(savepath_input + '/' + current_trial_name + '/' + 'theta_phi_traces_over_time_.png', dpi=300)
                plt.close()

            cam_center = [np.squeeze(CamCent[0]).tolist(), np.squeeze(CamCent[1]).tolist()]

            # make a DataFrame of the data that calc_ellipse() outputs
            trial_ellipse_df = pd.DataFrame({'theta':list(theta), 'phi':list(phi), 'longaxis_all':list(longaxis_all),
                                             'shortaxis_all':list(shortaxis_all)})

            # turn DataFrame into an xr DataArray, name the dims, fill in metadata (the trial name, which eye it is, etc.)
            ellipse_params = ['theta', 'phi', 'longaxis_all', 'shortaxis_all']
            len_index = len(x_vals.index.values)
            len_data = len(trial_ellipse_df)
            len_diff = len_index - len_data
            if len_index > len_data:
                time = x_vals.index.values[:-len_diff]
            elif len_index < len_data:
                step = x_vals.index.values[-1] - x_vals.index.values[-2]
                time = x_vals.index.values
                time.append(x_vals.index.values[-1] + step)
            elif len_index == len_data:
                time = x_vals.index.values

            trial_ellipse_data = xr.DataArray(trial_ellipse_df, coords=[('time', time), ('ellipse_params', ellipse_params)])
            trial_ellipse_data['trial'] = current_trial_name
            trial_ellipse_data['eye_side'] = side
            trial_ellipse_data['time_start'] = eye_data['time_start'].values
            trial_ellipse_data['time_end'] = eye_data['time_end'].values
            trial_ellipse_data['cam_center_x'] = cam_center[0]
            trial_ellipse_data['cam_center_y'] = cam_center[1]

            # append ellipse data from the current trial to a main xr DataArray to be saved out
            if trial_num == 0:
                side_ellipse = trial_ellipse_data
            elif trial_num > 0:
                side_ellipse = xr.concat([side_ellipse, trial_ellipse_data], dim='trial', fill_value=np.nan)

    return side_ellipse
