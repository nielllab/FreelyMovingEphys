"""
FreelyMovingEphys eye tracking utilities
track_eye.py

Last modified July 14, 2020
"""

# package imports
import pandas as pd
import numpy as np
from skimage import measure
import xarray as xr
import matplotlib.pyplot as plt
import os

# module imports
from util.read_data import split_xyl

# get out eye angles
def eye_angles(ellipseparams):
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

# clean ellipse estimate created in estimate_ellipse(), run eye_angles()
def clean_ellipse_estimate(ellipseparams, pxl_thresh):
    bdfit2, temp = np.where(ellipseparams[:, 2:4] > pxl_thresh)
    eparams = pd.DataFrame(ellipseparams)
    eparams.iloc[bdfit2, :] = np.nan
    eparams = eparams.interpolate(method='linear', limit_direction='both', axis=0)
    ellipseparams[bdfit2, :] = eparams.iloc[bdfit2, :]
    # run get_eye_angles on the cleaned data
    theta, phi, longaxis_all, shortaxis_all, CamCent = eye_angles(ellipseparams)

    return theta, phi, longaxis_all, shortaxis_all, CamCent

# estimate ellipse, run clean_ellipse_estimate() and eye_angles()
def estimate_ellipse(num_frames, x_vals, y_vals, pxl_thresh):
    emod = measure.EllipseModel()
    # create an empty array to be populated by the five outputs of EllipseModel()
    ellipseparams = np.empty((0, 5))
    timestamp_list = x_vals.index.values
    # index through each frame and stack the ellipse parameters
    for timestamp in timestamp_list:
        try:
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
    theta, phi, longaxis_all, shortaxis_all, CamCent = clean_ellipse_estimate(ellipseparams, pxl_thresh)

    return theta, phi, longaxis_all, shortaxis_all, CamCent

# make visualizations of how well eye tracking has worked
# def check_eye_calibration():

# track eye angle by calling other functions, takes in ONE trial at a time
def eye_tracking(eye_data, eye_pt_names, savepath, trial_name, lik_thresh, pxl_thresh, eye_pt_num, tear):
    # make directory for figure saving, if it does not already exist
    fig_dir = savepath + '/' + trial_name + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    eye_interp = xr.DataArray.interpolate_na(eye_data, dim='frame', use_coordinate='frame', method='linear')

    # break xarray into a pandas structure so it can be used by functions that get out the eye angle
    x_vals, y_vals, likeli_vals = split_xyl(eye_pt_names, eye_interp, lik_thresh)

    # drop tear
    if tear is True:
        x_vals = x_vals.drop([-2, -1], axis=1)
        y_vals = y_vals.drop([-2, -1], axis=1)
        likeli_vals = likeli_vals.drop([-2, -1], axis=1)

    num_frames = len(x_vals)

    theta, phi, longaxis_all, shortaxis_all, CamCent = estimate_ellipse(num_frames, x_vals, y_vals, pxl_thresh)

    # figure: theta and phi values over time in frames
    plt.subplots(2, 1, figsize=(30, 20))
    plt.subplot(211)
    plt.plot(theta * 180 / np.pi)
    plt.xlabel('frame')
    plt.ylabel('angle')
    plt.title(str(trial_name) + ' theta over time')
    plt.subplot(212)
    plt.plot(phi * 180 / np.pi)
    plt.xlabel('frame')
    plt.ylabel('angle')
    plt.title(str(trial_name) + ' phi over time')
    plt.savefig(fig_dir + 'theta_phi_traces.png', dpi=300)
    plt.close()

    cam_center = [np.squeeze(CamCent[0]).tolist(), np.squeeze(CamCent[1]).tolist()]
    ellipse_df = pd.DataFrame({'theta':list(theta), 'phi':list(phi), 'longaxis_all':list(longaxis_all),
                                             'shortaxis_all':list(shortaxis_all)})
    ellipse_params = ['theta', 'phi', 'longaxis_all', 'shortaxis_all']
    ellipse_out = xr.DataArray(ellipse_df, coords=[('frame', range(0, len(ellipse_df))), ('ellipse_params', ellipse_params)])
    ellipse_out['trial'] = trial_name
    ellipse_out['cam_center_x'] = cam_center[0]
    ellipse_out['cam_center_y'] = cam_center[1]

    return ellipse_out
