"""
track_eye.py

Eye tracking utilities

Last modified September 12, 2020
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
from tqdm import tqdm

# module imports
from util.read_data import split_xyl

def eye_angles(ellipseparams):
    '''
    get out eye angles
    '''

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

def clean_ellipse_estimate(ellipseparams, pxl_thresh):
    '''
    clean ellipse estimate created in estimate_ellipse(), run eye_angles()
    '''

    bdfit2, temp = np.where(ellipseparams[:, 2:4] > pxl_thresh)
    eparams = pd.DataFrame(ellipseparams)
    eparams.iloc[bdfit2, :] = np.nan
    eparams = eparams.interpolate(method='linear', limit_direction='both', axis=0)
    ellipseparams[bdfit2, :] = eparams.iloc[bdfit2, :]
    # run get_eye_angles on the cleaned data
    theta, phi, longaxis_all, shortaxis_all, CamCent = eye_angles(ellipseparams)

    return theta, phi, longaxis_all, shortaxis_all, CamCent

def estimate_ellipse(num_frames, x_vals, y_vals, pxl_thresh):
    '''
    estimate ellipse, run clean_ellipse_estimate() and eye_angles()
    '''

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

def eye_tracking(eye_data, config, trial_name, eye_letter):
    '''
    calculate ellipse parameters from DLC points and format in xarray
    takes in ONE trial at a time

    inputs
    eye_data: xarray of DLC points formatted by h5_to_xr function
    config: dictionary read in from .json file giving metadata about experiments

    returns
    ellipse_out: xarray of ellipse parameters
    '''

    eye_pt_names = list(eye_data['point_loc'].values)
    try:
        eye_interp = xr.DataArray.interpolate_na(eye_data, dim='frame', use_coordinate='frame', method='linear')
    except AttributeError:
        # this catches an error being raised by the jupyter notebook, not needed for terminal interface
        eye_data = xr.Dataset.to_array(eye_data)
        eye_interp = xr.DataArray.interpolate_na(eye_data, dim='frame', use_coordinate='frame', method='linear')

    # break xarray into a pandas structure so it can be used by functions that get out the eye angle
    x_vals, y_vals, likeli_vals = split_xyl(eye_pt_names, eye_interp, config['lik_thresh'])

    # drop tear
    # these points ought to be used, this will be addressed later
    if config['tear'] is True:
        x_vals = x_vals.iloc[:,:-2]
        y_vals = y_vals.iloc[:,:-2]

    num_frames = len(x_vals)

    theta, phi, longaxis, shortaxis, CamCent, centX, centY = estimate_ellipse(num_frames, x_vals, y_vals, config['pxl_thresh'])

    if config['save_vids'] is True:
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
        plt.savefig(os.path.join(config['save_path'], (trial_name + '_' + eye_letter + 'EYE_theta_phi.png')), dpi=300)
        plt.close()

    cam_center = [np.squeeze(CamCent[0]).tolist(), np.squeeze(CamCent[1]).tolist()]

    ellipse_df = pd.DataFrame({'theta':list(theta), 'phi':list(phi), 'longaxis':list(longaxis), 'shortaxis':list(shortaxis), 'centX':list(centX), 'centY':list(centY)})
    ellipse_params = ['theta', 'phi', 'longaxis', 'shortaxis', 'centX', 'centY']
    ellipse_out = xr.DataArray(ellipse_df, coords=[('frame', range(0, len(ellipse_df))), ('ellipse_params', ellipse_params)])
    ellipse_out['trial'] = trial_name
    ellipse_out['cam_center_x'] = cam_center[0]
    ellipse_out['cam_center_y'] = cam_center[1]

    return ellipse_out

def plot_eye_vid(vid_path, dlc_data, ell_data, config, trial_name, eye_letter):
    '''
    plot DLC points around eye and ellipse from previously calculated parameters on video and save out as an .avi

    inputs
    vid_path: file path to one eye video for a single trial, should be a string, '/path/to/vid.avi'
    dlc_data: xarray of dlc points around the eye as formatted by h5_to_xr() function
    ell_data: xarray of ellipse parameters
    config: dictionary read in from .json file giving metadata about experiments
    trial_name: string, the name of this trial EXCLUDING the type of video this is (i.e. it shoudn't include 'LEYE' or 'REYE')
    eye_letter: string, either 'R' or 'L' to indicate the side of the mouse that the eye video comes from, used to label the saved out .avi file

    returns nothing
    '''

    # read topdown video in
    vidread = cv2.VideoCapture(vid_path)
    width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # setup the file to save out of this
    savepath = os.path.join(config['save_path'], (trial_name + '_' + eye_letter + 'EYE.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))

    # set colors
    plot_color0 = (225, 255, 0)
    plot_color1 = (0, 255, 255)

    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # read the frame for this pass through while loop
        ret_le, frame_le = vidread.read()

        if not ret_le:
            break

        if dlc_data is not None and ell_data is not None:
            # get current frame number to be displayed, so that it can be used to slice DLC data
            try:
                ell_data_thistime = ell_data.sel(frame=vidread.get(cv2.CAP_PROP_POS_FRAMES))
                # get out ellipse parameters and plot them on the video
                ellipse_axes = (int(ell_data_thistime.sel(ellipse_params='longaxis').values), int(ell_data_thistime.sel(ellipse_params='shortaxis').values))
                ellipse_phi = int(np.rad2deg(ell_data_thistime.sel(ellipse_params='phi').values))
                ellipse_cent = (int(ell_data_thistime.sel(ellipse_params='centX').values), int(ell_data_thistime.sel(ellipse_params='centY').values))
                frame_le = cv2.ellipse(frame_le, ellipse_cent, ellipse_axes, ellipse_phi, 0, 360, plot_color0, 2)
            except (ValueError, KeyError) as e:
                pass

            # get out the DLC points and plot them on the video
            try:
                leftptsTS = dlc_data.sel(frame=vidread.get(cv2.CAP_PROP_POS_FRAMES))
                for k in range(0, len(leftptsTS), 3):
                    pt_cent = (int(leftptsTS.isel(point_loc=k).values), int(leftptsTS.isel(point_loc=k+1).values))
                    frame_le = cv2.circle(frame_le, pt_cent, 3, plot_color1, -1)
            except (ValueError, KeyError) as e:
                pass

        elif dlc_data is None or ell_data is None:
            pass

        out_vid.write(frame_le)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    out_vid.release()
    # cv2.destroyAllWindows()
