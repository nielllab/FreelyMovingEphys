"""
FreelyMovingEphys eye tracking utilities
track_eye.py

Last modified August 14, 2020
"""

# package imports
import pandas as pd
import numpy as np
from skimage import measure
import xarray as xr
import matplotlib.pyplot as plt
import os
import cv2
from skimage import measure
from itertools import product

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
    centX = []; centY = []
    for timestamp in timestamp_list:
        try:
            x_block = x_vals.loc[timestamp, :]
            y_block = y_vals.loc[timestamp, :]
            xy = np.column_stack((x_block, y_block))
            if emod.estimate(xy) is True:
                params_raw = np.array(emod.params)
                params_expanded = np.expand_dims(params_raw, axis=0)
                ellipseparams = np.append(ellipseparams, params_expanded, axis=0)
                centX.append(np.nanmean(x_block, axis=0)); centY.append(np.nanmean(y_block, axis=0))
            else:
                ellipseparams = np.append(ellipseparams, np.empty((0, 5)), axis=0)
                centX.append(np.empty(np.shape(1))); centY.append(np.empty(np.shape(1)))
        except np.linalg.LinAlgError as err:
            # if the timestamp cannot be found, add a filler entry of parameters
            ellipseparams = np.append(ellipseparams, np.empty((0, 5)), axis=0)
    theta, phi, longaxis_all, shortaxis_all, CamCent = clean_ellipse_estimate(ellipseparams, pxl_thresh)

    return theta, phi, longaxis_all, shortaxis_all, CamCent, centX, centY

# make visualizations of how well eye tracking has worked
# def check_eye_calibration():

# track eye angle by calling other functions, takes in ONE trial at a time
def eye_tracking(eye_data, eye_pt_names, savepath, trial_name, lik_thresh, pxl_thresh, eye_pt_num, tear):
    # make directory for figure saving, if it does not already exist
    fig_dir = savepath + '/' + trial_name + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    try:
        eye_interp = xr.DataArray.interpolate_na(eye_data, dim='frame', use_coordinate='frame', method='linear')
    except AttributeError:
        # this catches an error being raised by the jupyter notebook, not needed for terminal interface
        eye_data = xr.Dataset.to_array(eye_data)
        eye_interp = xr.DataArray.interpolate_na(eye_data, dim='frame', use_coordinate='frame', method='linear')

    # break xarray into a pandas structure so it can be used by functions that get out the eye angle
    x_vals, y_vals, likeli_vals = split_xyl(eye_pt_names, eye_interp, lik_thresh)

    # drop tear
    # these points ought to be used, this will be addressed later
    if tear is True:
        x_vals = x_vals.drop([-2, -1], axis=1)
        y_vals = y_vals.drop([-2, -1], axis=1)
        likeli_vals = likeli_vals.drop([-2, -1], axis=1)

    num_frames = len(x_vals)

    theta, phi, longaxis, shortaxis, CamCent, centX, centY = estimate_ellipse(num_frames, x_vals, y_vals, pxl_thresh)

    # figure: theta and phi values over time in frames
    plt.subplots(2, 1)
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

    ellipse_df = pd.DataFrame({'theta':list(theta), 'phi':list(phi), 'longaxis':list(longaxis), 'shortaxis':list(shortaxis), 'centX':list(centX), 'centY':list(centY)})
    ellipse_params = ['theta', 'phi', 'longaxis', 'shortaxis', 'centX', 'centY']
    ellipse_out = xr.DataArray(ellipse_df, coords=[('frame', range(0, len(ellipse_df))), ('ellipse_params', ellipse_params)])
    ellipse_out['trial'] = trial_name
    ellipse_out['cam_center_x'] = cam_center[0]
    ellipse_out['cam_center_y'] = cam_center[1]

    return ellipse_out

# plot points and ellipse on eye video as a saftey check, then save as .avi
def check_eye_tracking(trial_name, vid_path, savepath, dlc_data=None, ell_data=None, vext=None):

    # read topdown video in
    vidread = cv2.VideoCapture(vid_path)
    width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # setup the file to save out of this
    savepath = str(savepath) + '/' + str(trial_name) + '/' + str(trial_name) + '_' + vext + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (width, height))

    # set colors
    plot_color0 = (225, 255, 0)
    plot_color1 = (0, 255, 255)

    while (1):
        # read the frame for this pass through while loop
        ret_le, frame_le = vidread.read()

        if not ret_le:
            break

        if dlc_data is not None and ell_data is not None:
            # get current frame number to be displayed, so that it can be used to slice DLC data
            try:
                frame_time = vidread.get(cv2.CAP_PROP_POS_FRAMES)
                ell_data_thistime = ell_data.sel(frame=frame_time)
                dlc_data_thistime = dlc_data.sel(frame=frame_time)
                # get out ellipse parameters and plot them on the video
                ellipse_axes = (int(ell_data_thistime.sel(ellipse_params='longaxis').values), int(ell_data_thistime.sel(ellipse_params='shortaxis').values))
                ellipse_phi = int(ell_data_thistime.sel(ellipse_params='phi').values)
                ellipse_cent = (int(ell_data_thistime.sel(ellipse_params='centX').values), int(ell_data_thistime.sel(ellipse_params='centY').values))
                frame_le = cv2.ellipse(frame_le, ellipse_cent, ellipse_axes, ellipse_phi, 0, 360, plot_color0, 2)
            except (ValueError, KeyError) as e:
                pass

            # get out the DLC points and plot them on the video
            try:
                leftptsTS = dlc_data.sel(frame=vidread.get(cv2.CAP_PROP_POS_FRAMES))
                for k in range(0, 24, 3):
                    pt_cent = (int(leftptsTS.isel(point_loc=k).values), int(leftptsTS.isel(point_loc=k+1).values))
                    frame_le = cv2.circle(frame_le, pt_cent, 3, plot_color1, -1)
            except (ValueError, KeyError) as e:
                pass

        elif dlc_data is None or ell_data is None:
            pass

        out_vid.write(frame_le)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_vid.release()
    cv2.destroyAllWindows()
