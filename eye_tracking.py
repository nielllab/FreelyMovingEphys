#####################################################################################
"""
eye_tracking.py of FreelyMovingEphys

Function eye_angles() takes in xarray DataArray of left and right eye videos. Will run
with one or both eyes fine. Data are thresholded by likelihood, turned into
pandas arrays, and passed to calc_ellipse() whih gets out the least squares estimate
for a 2D ellipse. The paramaters of the ellipse estimated by calc_ellipse() is
cleaned up by preen_then_get_eye_angles(), and through that, passed to get_eye_angles()
to extract from the ellipse paramaters the angle of the mouse's eye for each frame.

Adapted from code by Elliott Abe, DLCEyeVids.py, especially for the functions
get_eye_angles(), preen_then_get_eye_angles(), and calc_ellipse().

TO DO:
- get eye angles out, this is not working yet with the xarray DataArray passed into the
function eye_angles()

last modified: June 8, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################

import os
import fnmatch
import cv2
import pandas as pd
import numpy as np
from skimage import draw, measure
import xarray as xr

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

def preen_then_get_eye_angles(ellipseparams, pxl_thresh):
    bdfit2, temp = np.where(ellipseparams[:, 2:4] > pxl_thresh)
    eparams = pd.DataFrame(ellipseparams)
    eparams.iloc[bdfit2, :] = np.nan
    eparams = eparams.interpolate(method='linear', limit_direction='both', axis=0)
    ellipseparams[bdfit2, :] = eparams.iloc[bdfit2, :]
    # run get_eye_angles on the cleaned data
    theta, phi, longaxis_all, shortaxis_all, CamCent = get_eye_angles(ellipseparams)
    return theta, phi, longaxis_all, shortaxis_all, CamCent

def calc_ellipse(num_frames, x_vals, y_vals, pxl_thresh):
    emod = measure.EllipseModel()
    # create an empty array to be populated by the five outputs of EllipseModel()
    ellipseparams = np.empty((0, 5))
    # index through each frame and stack the ellipse parameters
    for frame in range(0, num_frames):
        # first the ellipse
        xy = np.column_stack((x_vals[frame], y_vals[frame]))
        if emod.estimate(xy) is True:
            params_raw = np.array(emod.params)
            params_expanded = np.expand_dims(params_raw, axis=0)
            ellipseparams = np.append(ellipseparams, params_expanded, axis=0)

    theta, phi, longaxis_all, shortaxis_all, CamCent = preen_then_get_eye_angles(ellipseparams, pxl_thresh)

    return theta, phi, longaxis_all, shortaxis_all, CamCent

def eye_angles(eye_data_input, eye_names, trial_id_list, figures=False, thresh=0.99, pxl_thresh=50):
    # prepares data for use with Elliott's get_eye_angles
    # runs on one eye at a time, but can run on both if needed
    # pxl_thresh is the max number of pixels for radius of pupil
    # thresh is the liklihood threshold
    for trial_num in range(0, len(trial_id_list)):
        current_trial_name = trial_id_list[trial_num]

        # run on left eye first
        if eye_data_input.sel(trial=current_trial_name) is not None:
            with eye_data_input.sel(trial=current_trial_name) as eye_data:

                # make list of x and y point_loc coords
                x_locs = []
                y_locs = []
                likeli_locs = []
                for loc_num in range(0, len(eye_names)):
                    loc = eye_names[loc_num]
                    if ' x' in loc:
                        x_locs.append(loc)
                    elif ' y' in loc:
                        y_locs.append(loc)
                    elif ' likeli' in loc:
                        likeli_locs.append(loc)
                    elif loc is None:
                        print('loc is None')

                # OBJECTIVE: try getting all eight eye points into the array passed to calc_ellipse()

                # get the xarray split up into x, y,and likelihood
                for loc_num in range(0, len(x_locs)):
                    pt_loc = x_locs[loc_num]
                    if loc_num == 0:
                        x_pts = eye_data.sel(point_loc=pt_loc)
                    elif loc_num > 0:
                        x_pts = xr.concat([x_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
                for loc_num in range(0, len(y_locs)):
                    pt_loc = y_locs[loc_num]
                    if loc_num == 0:
                        y_pts = eye_data.sel(point_loc=pt_loc)
                    elif loc_num > 0:
                        y_pts = xr.concat([y_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
                for loc_num in range(0, len(likeli_locs)):
                    pt_loc = likeli_locs[loc_num]
                    if loc_num == 0:
                        likeli_pts = eye_data.sel(point_loc=pt_loc)
                    elif loc_num > 0:
                        likeli_pts = xr.concat([likeli_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)

                # threshold from likelihood
                x_pts[likeli_pts < thresh] = np.nan
                y_pts[likeli_pts < thresh] = np.nan

                # get out values as numpy array, dropping NaNs
                x_vals = np.array(x_pts.values[np.isfinite(x_pts.values)])
                y_vals = np.array(y_pts.values[np.isfinite(y_pts.values)])

                # get the number of frames
                num_frames = len(x_vals)

                theta, phi, longaxis_all, shortaxis_all, CamCent = calc_ellipse(num_frames, x_vals, y_vals, pxl_thresh)

                print(theta)

    # return theta, phi, longaxis_all, shortaxis_all, CamCent