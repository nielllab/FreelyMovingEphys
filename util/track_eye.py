"""
track_eye.py
"""

from skimage import measure
from itertools import product
from numpy import *
import json
from math import e as e
from numpy.linalg import eig
import math
import matplotlib.backends.backend_pdf
from scipy import stats
import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
import cv2
from scipy import signal
from scipy.optimize import curve_fit
import scipy.stats as st
import time
import subprocess as sp
import multiprocessing
import sys
import warnings
from scipy import ndimage
import time
from tqdm import tqdm
import matplotlib as mpl
from astropy.convolution import convolve

from util.format_data import split_xyl
from util.time import open_time
from util.paths import find
from util.aux_funcs import nanxcorr

def fit_ellipse(x,y):
    """
    finds the best fit to an ellipse for the given set of points in a single frame
    INPUTS
        x: x values of points around pupil as two numpy arrays
        y: y values of points around pupil as two numpy arrays
    RETURNS
        ellipse_dict: dictionary of ellipse parameters for a single frame
    adapted from /niell-lab-analysis/freely moving/fit_ellipse2.m
    """
    orientation_tolerance = 1*np.exp(-3)
    
    # remove bias of the ellipse
    meanX = np.mean(x)
    meanY = np.mean(y)
    x = x - meanX
    y = y - meanY
    
    # estimation of the conic equation
    X = np.array([x**2, x*y, y**2, x, y])
    X = np.stack(X).T
    a = dot(np.sum(X, axis=0), linalg.pinv(np.matmul(X.T,X)))
    
    # extract parameters from the conic equation
    a, b, c, d, e = a[0], a[1], a[2], a[3], a[4]
    
    # eigen decomp
    Q = np.array([[a, b/2],[b/2, c]])
    eig_val, eig_vec = eig(Q)
    
    # get angle to long axis
    if eig_val[0] < eig_val[1]:
      angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
    else:
      angle_to_x = np.arctan2(eig_vec[1,1], eig_vec[0,1])
    angle_from_x = angle_to_x

    orientation_rad = 0.5 * np.arctan2(b, (c-a))
    cos_phi = np.cos(orientation_rad)
    sin_phi = np.sin(orientation_rad)
    a, b, c, d, e = [a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
                     0,
                     a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
                     d*cos_phi - e*sin_phi,
                     d*sin_phi + e*cos_phi]
    meanX, meanY = [cos_phi*meanX - sin_phi*meanY,
                    sin_phi*meanX + cos_phi*meanY]
    
    # check if conc expression represents an ellipse
    test = a*c
    if test > 0:
        # make sure coefficients are positive as required
        if a<0:
            a, c, d, e = [-a, -c, -d, -e]
        
        # final ellipse parameters
        X0 = meanX - d/2/a
        Y0 = meanY - e/2/c
        F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
        a = np.sqrt(F/a)
        b = np.sqrt(F/c)
        long_axis = 2*np.maximum(a,b)
        short_axis = 2*np.minimum(a,b)
        
        # rotate axes backwards to find center point of original tilted ellipse
        R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
        P_in = R @ np.array([[X0],[Y0]])
        X0_in = P_in[0][0]
        Y0_in = P_in[1][0]
        
        # organize parameters in dictionary to return
        # makes some final modifications to values here, maybe those should be done above for cleanliness
        ellipse_dict = {'X0':X0, 'Y0':Y0, 'F':F, 'a':a, 'b':b, 'long_axis':long_axis/2, 'short_axis':short_axis/2,
                        'angle_to_x':angle_to_x, 'angle_from_x':angle_from_x, 'cos_phi':cos_phi, 'sin_phi':sin_phi,
                        'X0_in':X0_in, 'Y0_in':Y0_in, 'phi':orientation_rad}
        
    else:
        # if the conic equation didn't return an ellipse, don't return any real values and fill the dictionary with NaNs
        ellipse_dict = {'X0':np.nan, 'Y0':np.nan, 'F':np.nan, 'a':np.nan, 'b':np.nan, 'long_axis':np.nan, 'short_axis':np.nan,
                        'angle_to_x':np.nan, 'angle_from_x':np.nan, 'cos_phi':np.nan, 'sin_phi':np.nan,
                        'X0_in':np.nan, 'Y0_in':np.nan, 'phi':np.nan}

    return ellipse_dict

def eye_tracking(eye_data, config, trial_name, eye_side):
    """
    get the ellipse parameters from DeepLabCut points and save into an xarray
    INPUTS
        eye_data: xarray of eye point positions and likelihood
        config: dictionary of options
        trial_name: str, e.g. '010121_subject_...'
        eye_side: str, e.g. 'REYE'
    OUTPUTS
        ellipse_out: xarray DataArray of ellipse parameters
    also saves to file a pdf of diagnostic figures
    adapted from /niell-lab-analysis/freely moving/EyeCameraCalc1.m
    """
    # set up the pdf to be saved out with diagnostic figures
    if config['parameters']['outputs_and_visualization']['save_figs'] is True:
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(config['recording_path'], (trial_name + '_' + eye_side + '_tracking_figs.pdf')))

    fig_dwnsmpl = config['parameters']['outputs_and_visualization']['eye_fig_pts_dwnspl']

    # if this is a hf recoridng, read in existing fm camera center, scale, etc.
    # it should run all fm recordings first, so it will be possible to read in fm camera calibration parameters for every hf recording
    if 'hf' in trial_name:
        path_to_existing_props = sorted(find('*fm_eyecameracalc_props.json', config['animal_dir'])) # should always go for fm1 before fm2
        if len(path_to_existing_props) == 0:
            print('found no existing camera calibration properties from freely moving recording')
            path_to_existing_props = None
        elif len(path_to_existing_props) == 1:
            print('found one existing file of camera calirbation properties from freely moving recording')
            path_to_existing_props = path_to_existing_props[0]
        elif len(path_to_existing_props) > 1:
            print('found multiple existing files of camera calibration properties from freely moving recordings -- using first option from sorted list')
            path_to_existing_props = path_to_existing_props[0]
        if path_to_existing_props is not None:
            with open(path_to_existing_props, 'r') as fp:
                existing_camera_calib_props = json.load(fp)
        elif path_to_existing_props is None:
            # if a json of paramters can't be found, though, we'll get these values for the hf recording
            existing_camera_calib_props = None
    elif 'fm' in trial_name:
        existing_camera_calib_props = None

    # names of the different points
    pt_names = list(eye_data['point_loc'].values)

    x_vals, y_vals, likeli_vals = split_xyl(pt_names, eye_data, config['parameters']['lik_thresh'])
    likelihood_in = likeli_vals.values

    # subtract center of IR light reflection from all other pts
    if config['pose_estimation']['has_ir_spot_labeled'] and config['parameters']['eyes']['spot_subtract']:

        spot_xcent = np.mean(x_vals.iloc[:,-5:], 1)
        spot_ycent = np.mean(y_vals.iloc[:,-5:], 1)
        
        spot_likelihood = likelihood_in[:,-5:].copy()
        likelihood = likelihood_in[:,:9]

        x_vals = x_vals.iloc[:,:-5].subtract(spot_xcent, axis=0)
        y_vals = y_vals.iloc[:,:-5].subtract(spot_ycent, axis=0)

    elif config['pose_estimation']['has_ir_spot_labeled'] is True and config['parameters']['eyes']['spot_subtract'] is False:
        spot_xvals = x_vals.iloc[:,-5:]
        spot_yvals = y_vals.iloc[:,-5:]
        spot_likelihood = likelihood_in[:,-5:]

        x_vals = x_vals.iloc[:,:-5]
        y_vals = y_vals.iloc[:,:-5]
        likelihood = likelihood_in[:,:-5]

    # drop tear/outer
    if config['pose_estimation']['has_ir_spot_labeled'] is True:
        x_vals = x_vals.iloc[:,:-2]
        y_vals = y_vals.iloc[:,:-2]
        likelihood = likelihood_in[:,:-2]

    # get bools of when a frame is usable with the right number of points above threshold
    if config['parameters']['eyes']['spot_subtract'] is True:
        # if spot subtraction is being done, we should only include frames where all five pts marked around the ir spot are good (centroid would be off otherwise)
        usegood_req5 = (np.sum(likelihood >= config['parameters']['lik_thresh'], 1) >= config['parameters']['eyes']['num_ellipse_pts_needed'])# & (np.sum(spot_likelihood >= config['parameters']['lik_thresh'], 1) >= config['parameters']['eyes']['num_ir_spot_pts_needed'])
        usegood_req8 = (np.sum(likelihood >= config['parameters']['lik_thresh'], 1) >= config['parameters']['eyes']['calib_ellipse_pts_needed'])# & (np.sum(spot_likelihood >= config['parameters']['lik_thresh'], 1) >= config['parameters']['eyes']['num_ir_spot_pts_needed'])
        spot_usegood = (np.sum(spot_likelihood >= config['parameters']['lik_thresh'], 1) >= config['parameters']['eyes']['num_ir_spot_pts_needed'])
    else:
        usegood_req5 = np.sum(likelihood >= config['parameters']['lik_thresh'], 1) >= config['parameters']['eyes']['num_ellipse_pts_needed']
        usegood_req8 = np.sum(likelihood >= config['parameters']['lik_thresh'], 1) >= config['parameters']['eyes']['calib_ellipse_pts_needed']

    # plot all good timepoints
    if config['parameters']['outputs_and_visualization']['save_figs'] is True:
        if config['parameters']['eyes']['spot_subtract'] is True:
            plt.figure()
            plt.plot(np.sum(spot_likelihood >= config['parameters']['eye']['num_ir_spot_pts_needed'], 1)[0:-1:10])
            plt.title(str(np.round(np.mean(spot_usegood), 3)) + ' good (req5) for IR spot; thresh= ' + str(config['parameters']['lik_thresh']))
            plt.ylabel('num good IR spot points'); plt.xlabel('every 10th frame')
            pdf.savefig()
            plt.close()

        plt.figure()
        plt.plot(np.sum(likelihood >= config['parameters']['lik_thresh'], 1)[0:-1:10])
        plt.title(str(np.round(np.mean(usegood_req5), 3)) + ' good (req5); thresh= ' + str(config['parameters']['lik_thresh']))
        plt.ylabel('num good eye points'); plt.xlabel('every 10th frame')
        pdf.savefig()
        plt.close()
    
        plt.figure()
        plt.hist(np.sum(likelihood >= config['parameters']['lik_thresh'], 1),bins=9, range = (0,9))
        plt.xlabel('num good eye points'); plt.ylabel('n frames')
        pdf.savefig()
        plt.close()

    # threshold out pts more than a given distance away from nanmean of that point
    std_thresh_x = np.empty(np.shape(x_vals))
    for point_loc in range(0,np.size(x_vals, 1)):
        std_thresh_x[:,point_loc] = (np.absolute(np.nanmean(x_vals.iloc[:,point_loc]) - x_vals.iloc[:,point_loc]) / config['parameters']['eyes']['eyecam_pxl_per_cm']) > config['parameters']['eyes']['eye_dist_thresh_cm']
    std_thresh_y = np.empty(np.shape(y_vals))
    for point_loc in range(0,np.size(x_vals, 1)):
        std_thresh_y[:,point_loc] = (np.absolute(np.nanmean(y_vals.iloc[:,point_loc]) - y_vals.iloc[:,point_loc]) / config['parameters']['eyes']['eyecam_pxl_per_cm']) > config['parameters']['eyes']['eye_dist_thresh_cm']
    std_thresh_x = np.nanmean(std_thresh_x, 1)
    std_thresh_y = np.nanmean(std_thresh_y, 1)
    x_vals[std_thresh_x > 0] = np.nan
    y_vals[std_thresh_y > 0] = np.nan

    ellipse_params = np.empty([len(usegood_req5), 14])
    # step through each frame, fit an ellipse to points, and
    # add ellipse parameters to array with data for all frames together
    linalgerror = 0
    for step in tqdm(range(0,len(usegood_req5))):
        if usegood_req5[step] == True:
            try:
                e_t = fit_ellipse(x_vals.iloc[step].values, y_vals.iloc[step].values)
                ellipse_params[step] = [e_t['X0'], e_t['Y0'], e_t['F'], e_t['a'], e_t['b'],
                                        e_t['long_axis'], e_t['short_axis'], e_t['angle_to_x'], e_t['angle_from_x'],
                                        e_t['cos_phi'], e_t['sin_phi'], e_t['X0_in'], e_t['Y0_in'], e_t['phi']]
            except np.linalg.LinAlgError as e:
                linalgerror = linalgerror + 1
                e_t = {'X0':np.nan, 'Y0':np.nan, 'F':np.nan, 'a':np.nan, 'b':np.nan, 'long_axis':np.nan, 'short_axis':np.nan,
                            'angle_to_x':np.nan, 'angle_from_x':np.nan, 'cos_phi':np.nan, 'sin_phi':np.nan,
                            'X0_in':np.nan, 'Y0_in':np.nan, 'phi':np.nan}
                ellipse_params[step] = [e_t['X0'], e_t['Y0'], e_t['F'], e_t['a'], e_t['b'],
                                    e_t['long_axis'] ,e_t['short_axis'], e_t['angle_to_x'], e_t['angle_from_x'],
                                    e_t['cos_phi'], e_t['sin_phi'], e_t['X0_in'], e_t['Y0_in'], e_t['phi']]
        elif usegood_req5[step] == False:
            e_t = {'X0':np.nan, 'Y0':np.nan, 'F':np.nan, 'a':np.nan, 'b':np.nan, 'long_axis':np.nan, 'short_axis':np.nan,
                            'angle_to_x':np.nan, 'angle_from_x':np.nan, 'cos_phi':np.nan, 'sin_phi':np.nan,
                            'X0_in':np.nan, 'Y0_in':np.nan, 'phi':np.nan}
            ellipse_params[step] = [e_t['X0'], e_t['Y0'], e_t['F'], e_t['a'], e_t['b'],
                                    e_t['long_axis'] ,e_t['short_axis'], e_t['angle_to_x'], e_t['angle_from_x'],
                                    e_t['cos_phi'], e_t['sin_phi'], e_t['X0_in'], e_t['Y0_in'], e_t['phi']]
    print('lin alg error count = ' + str(linalgerror))

    # list of all places where the ellipse meets threshold
    R = np.linspace(0,2*np.pi, 100)
    usegood_ellipcalb = np.where((usegood_req8 == True) & ((ellipse_params[:,6] / ellipse_params[:,5]) < config['parameters']['eyes']['ell_thresh'])) # short axis / long axis

    # this limits the number of frames used for the calibration
    if np.size(usegood_ellipcalb,1) > 50000:
        shortlist = sorted(np.random.choice(usegood_ellipcalb[0],size=50000, replace=False))
    else:
        shortlist = usegood_ellipcalb

    # find camera center
    A = np.vstack([np.cos(ellipse_params[shortlist,7]),np.sin(ellipse_params[shortlist,7])])
    b = np.expand_dims(np.diag(A.T@np.squeeze(ellipse_params[shortlist,11:13].T)),axis=1)
    if existing_camera_calib_props is None:
        cam_cent = np.linalg.inv(A@A.T)@A@b
    elif existing_camera_calib_props is not None:
        cam_cent = np.array([[float(existing_camera_calib_props['cam_cent_x'])],[float(existing_camera_calib_props['cam_cent_y'])]])

    # ellipticity and scale
    ellipticity = (ellipse_params[shortlist,6] / ellipse_params[shortlist,5]).T
    if existing_camera_calib_props is None:
        try:
            scale = np.nansum(np.sqrt(1-(ellipticity)**2)*(np.linalg.norm(ellipse_params[shortlist,11:13]-cam_cent.T,axis=0)))/np.sum(1-(ellipticity)**2)
        except ValueError:
            scale = np.nansum(np.sqrt(1-(ellipticity)**2)*(np.linalg.norm(ellipse_params[shortlist,11:13]-cam_cent.T,axis=1)))/np.sum(1-(ellipticity)**2)
    elif existing_camera_calib_props is not None:
        scale = float(existing_camera_calib_props['scale'])

    # angles
    theta = np.arcsin((ellipse_params[:,11]-cam_cent[0])/scale)
    phi = np.arcsin((ellipse_params[:,12]-cam_cent[1])/np.cos(theta)/scale)

    # if config['parameters']['outputs_and_visualization']['save_figs'] is True:
    #     plt.figure()
    #     plt.scatter(ellipse_params[:,7], phi)
    #     plt.title('angle_to_x vs phi')
    #     pdf.savefig()
    #     plt.close()

    if config['parameters']['outputs_and_visualization']['save_figs'] is True:
        try:
            plt.figure()
            plt.plot(np.rad2deg(phi)[0:-1:10])
            plt.title('phi')
            plt.ylabel('deg'); plt.xlabel('every 10th frame')
            pdf.savefig()
            plt.close()

            plt.figure()
            plt.plot(np.rad2deg(theta)[0:-1:10])
            plt.title('theta')
            plt.ylabel('deg'); plt.xlabel('every 10th frame')
            pdf.savefig()
            plt.close()
        except:
            print('figure error')

    # organize data to return as an xarray of most essential parameters
    ellipse_df = pd.DataFrame({'theta':list(theta), 'phi':list(phi), 'longaxis':list(ellipse_params[:,5]), 'shortaxis':list(ellipse_params[:,6]),
                               'X0':list(ellipse_params[:,11]), 'Y0':list(ellipse_params[:,12]), 'ellipse_phi':list(ellipse_params[:,7])})
    ellipse_param_names = ['theta', 'phi', 'longaxis', 'shortaxis', 'X0', 'Y0', 'ellipse_phi']
    ellipse_out = xr.DataArray(ellipse_df, coords=[('frame', range(0, len(ellipse_df))), ('ellipse_params', ellipse_param_names)], dims=['frame', 'ellipse_params'])
    ellipse_out.attrs['cam_center_x'] = cam_cent[0,0]
    ellipse_out.attrs['cam_center_y'] = cam_cent[1,0]

    # ellipticity histogram
    if config['parameters']['outputs_and_visualization']['save_figs'] is True:
        try:
            plt.figure()
            plt.hist(ellipticity)
            plt.title('ellipticity; thresh= ' + str(config['parameters']['eyes']['ell_thresh']))
            plt.ylabel('num good eye points'); plt.xlabel('frame')
            pdf.savefig()
            plt.close()

            w = ellipse_params[:,7]
            
            # eye axes relative to center
            plt.figure()
            for i in range(0,len(usegood_ellipcalb)):
                plt.plot((ellipse_params[usegood_ellipcalb[i::fig_dwnsmpl],11] + [-5*np.cos(w[usegood_ellipcalb[i::fig_dwnsmpl]]), 5*np.cos(w[usegood_ellipcalb[i::fig_dwnsmpl]])]), (ellipse_params[usegood_ellipcalb[i::fig_dwnsmpl],12] + [-5*np.sin(w[usegood_ellipcalb[i::fig_dwnsmpl]]), 5*np.sin(w[usegood_ellipcalb[i::fig_dwnsmpl]])]))
            plt.plot(cam_cent[0],cam_cent[1],'r*')
            plt.title('eye axes relative to center')
            pdf.savefig()
            plt.close()
        except:
            print('figure error')

        # check calibration
        try:
            xvals = np.linalg.norm(ellipse_params[usegood_req8, 11:13].T - cam_cent, axis=0)
            yvals = scale * np.sqrt(1-(ellipse_params[usegood_req8,6]/ellipse_params[usegood_req8,5])**2)
            calib_mask = ~np.isnan(xvals) & ~np.isnan(yvals)
            slope, intercept, r_value, p_value, std_err = stats.linregress(xvals[calib_mask], yvals[calib_mask].T)
        except ValueError:
            print('no good frames that meet criteria... check DLC tracking!')

        # save out camera center and scale as np array (but only if this is a freely moving recording)
        if 'fm' in trial_name:
            calib_props_dict = {'cam_cent_x':float(cam_cent[0]), 'cam_cent_y':float(cam_cent[1]), 'scale':float(scale), 'regression_r':float(r_value), 'regression_m':float(slope)}
            calib_props_dict_savepath = os.path.join(config['recording_path'], str(trial_name+eye_side+'_fm_eyecameracalc_props.json'))
            with open(calib_props_dict_savepath, 'w') as f:
                json.dump(calib_props_dict, f)

        try:
            plt.figure()
            plt.plot(xvals[::fig_dwnsmpl], yvals[::fig_dwnsmpl], '.', markersize=1)
            plt.plot(np.linspace(0,50),np.linspace(0,50),'r')
            plt.title('scale=' + str(np.round(scale, 1)) + ' r=' + str(np.round(r_value, 1)) + ' m=' + str(np.round(slope, 1)))
            plt.xlabel('pupil camera dist'); plt.ylabel('scale * ellipticity')
            pdf.savefig()
            plt.close()

            # calibration of camera center
            delta = (cam_cent - ellipse_params[:,11:13].T)
            short_usegood_req8 = usegood_req8[::fig_dwnsmpl]
            sq_ellipcalib = np.squeeze(usegood_ellipcalb)
            short_list3 = sq_ellipcalib[::fig_dwnsmpl]

            plt.figure()
            plt.plot(np.linalg.norm(delta[:,short_usegood_req8],2,axis=0), ((delta[0,short_usegood_req8].T * np.cos(ellipse_params[short_usegood_req8,7])) + (delta[1,short_usegood_req8].T * np.sin(ellipse_params[short_usegood_req8, 7]))) / np.linalg.norm(delta[:, short_usegood_req8],2,axis=0).T, 'y.', markersize=1)
            plt.plot(np.linalg.norm(delta[:,short_list3],2,axis=0), ((delta[0,short_list3].T * np.cos(ellipse_params[short_list3,7])) + (delta[1,short_list3].T * np.sin(ellipse_params[short_list3, 7]))) / np.linalg.norm(delta[:, short_list3],2,axis=0).T, 'r.', markersize=1)
            plt.title('camera center calibration')
            plt.ylabel('abs([PC-EC]).[cosw;sinw]')
            plt.xlabel('abs(PC-EC)')
            plt.legend('all points','list points')
            pdf.savefig()
            plt.close()
        except:
            print('figure error')
    
        pdf.close()

    return ellipse_out

def plot_eye_vid(vid_path, dlc_data, ell_data, config, trial_name, eye_letter):
    """
    plot the ellipse and dlc points on the video frames
    then, save the video out as an .avi file
    INPUTS
        vid_path: file path of existing .avi video (should be deinterlaced and calibrated, i.e. what ran through DeepLabCut)
        dlc_data: xarray of dlc data (this should be before pts and ell params are merged into one)
        ell_data: ellipse fit xarray
        config: dict of options
        trial_name: str, e.g. '010121_subject_...'
        eye_letter: str, e.g. 'R'
    OUTPUTS
        None
    """
    # read in video
    # setup the file to save out
    vidread = cv2.VideoCapture(vid_path)
    width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))
    savepath = os.path.join(config['recording_path'], (trial_name + '_' + eye_letter + '_plot.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))

    # only do the first number of frames (limit of frames to use should be set in config dict)
    if config['parameters']['outputs_and_visualization']['num_save_frames'] > int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)):
        num_save_frames = int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        num_save_frames = config['parameters']['outputs_and_visualization']['num_save_frames']

    # iterate through frames
    for frame_num in tqdm(range(0,num_save_frames)):
        # read frame and make sure it's read in correctoy
        ret, frame = vidread.read()
        if not ret:
            break
        # plot on the frame if there is data to be used
        if dlc_data is not None and ell_data is not None:
            try:
                # open the ellipse data
                ell_data_thistime = ell_data.sel(frame=frame_num)
                # get out ellipse long/short axes and put into tuple
                ellipse_axes = (int(ell_data_thistime.sel(ellipse_params='longaxis').values), int(ell_data_thistime.sel(ellipse_params='shortaxis').values))
                # get out ellipse phi and round to int -- note: this is ellipse_phi not phi
                ellipse_phi = int(np.rad2deg(ell_data_thistime.sel(ellipse_params='ellipse_phi').values))
                # get ellipse center out, round to int, and put into tuple
                ellipse_cent = (int(ell_data_thistime.sel(ellipse_params='X0').values), int(ell_data_thistime.sel(ellipse_params='Y0').values))
                # update this frame with an ellipse
                frame = cv2.ellipse(frame, ellipse_cent, ellipse_axes, ellipse_phi, 0, 360, (255,0,0), 2) # ellipse in blue
            # skip if the ell data from this frame are bad
            except (ValueError, KeyError):
                pass
            try:
                # open the points that come directly from DLC
                pts = dlc_data.sel(frame=frame_num)
                # iterate through each point in the list
                for k in range(0, len(pts), 3):
                    # get the point center of each point num, k
                    pt_cent = (int(pts.isel(point_loc=k).values), int(pts.isel(point_loc=k+1).values))
                    # compare to threshold set in config and plot
                    if pts.isel(point_loc=k+2).values < config['parameters']['lik_thresh']: # bad points in red
                        frame = cv2.circle(frame, pt_cent, 3, (0,0,255), -1)
                    elif pts.isel(point_loc=k+2).values >= config['parameters']['lik_thresh']: # good points in green
                        frame = cv2.circle(frame, pt_cent, 3, (0,255,0), -1)
            except (ValueError, KeyError):
                pass
        elif dlc_data is None or ell_data is None:
            pass
        out_vid.write(frame)
    out_vid.release()

def curve_func(xval, a, b, c):
    """
    sigmoid function for pupil rotation
    """
    return a+(b-a)/(1+10**((c-xval)*2))

def sigm_fit_mp(d):
    """
    multiprocessing-ready fit to sigmoid function
    """
    try:
        popt, pcov = curve_fit(curve_func, xdata=range(1,len(d)+1), ydata=d,
                                p0=[100.0,200.0,len(d)/2], method='lm', xtol=10**-3, ftol=10**-3)
        ci = np.sqrt(np.diagonal(pcov))
    except RuntimeError:
        popt = np.nan*np.zeros(4)
        ci = np.nan*np.zeros(4)
    return (popt, ci)

def find_pupil_rotation(eye_ell_params, config, trial_name, side_letter='REYE'):
    """
    find the cyclotorsion (pupil rotation) using eye videos
    INPUTS
        eye_ell_params: eye theta, phi, etc. xarray
        config: options dict
        trial_name: str, e.g. '010121_subject_...'
        side_letter: str, side name, default = 'REYE'
    OUTPUTS
        rfit_xr: pupil radius as xarray (at each of 360deg)
        rfit_conv_xr: convolved radius of pupil as xarray (at each of 360deg)
        shift: shift in pxls needed to get best correlation with template 
    """

    eyevidpath = find((trial_name + '*' + side_letter + 'deinter.avi'), config['recording_path'])[0]
    eyetimepath = find((trial_name + '*' + side_letter + '_BonsaiTSformatted.csv'), config['recording_path'])[0]
    save_path = config['recording_path']; world_interp_method = 'linear'; ranger = 10

    print('found ' + str(multiprocessing.cpu_count()) + ' as cpu count for multiprocessing')

    if config['parameters']['outputs_and_visualization']['save_figs'] is True:
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(config['recording_path'], (trial_name + '_' + side_letter + '_pupil_rotation_figs.pdf')))

    # set up range of degrees in radians
    rad_range = np.deg2rad(np.arange(360))

    # open time files
    eyeTS = open_time(eyetimepath, np.size(eye_ell_params, axis=0))
    # worldTS = open_time(worldtimepath, np.size(eye_ell_params, axis=0))

    # interpolate ellipse parameters to worldcam timestamps
    # eye_ell_interp_params = eye_ell_params.interp_like(xr.DataArray(worldTS), method=world_interp_method)
    eye_ell_interp_params = eye_ell_params.copy()

    # get the ellipse parameters for this trial from the time-interpolated xarray
    eye_theta = eye_ell_interp_params.sel(ellipse_params='theta')
    eye_phi = eye_ell_interp_params.sel(ellipse_params='phi')
    eye_longaxis= eye_ell_interp_params.sel(ellipse_params='longaxis')
    eye_shortaxis = eye_ell_interp_params.sel(ellipse_params='shortaxis')
    eye_centX = eye_ell_interp_params.sel(ellipse_params='X0')
    eye_centY = eye_ell_interp_params.sel(ellipse_params='Y0')

    # set up for the read-in video
    eyevid = cv2.VideoCapture(eyevidpath)
    totalF = int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)) # this can be changed to a small number of frames for testing
    set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # set up for the multiprocessing that'll be used during sigmoid fit function
    n_proc = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_proc)
    n=0

    key_error_count = 0

    print('getting cross-section of pupil at each angle and fitting to sigmoid (slow)')
    for step in tqdm(np.arange(totalF)):
        try:
            # frame reading and black and white conversion
            eye_ret, eye_frame = eyevid.read()

            if not eye_ret:
                break

            eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

            # get ellipse parameters for this time
            current_theta = eye_theta.sel(frame=step).values
            current_phi = eye_phi.sel(frame=step).values
            current_longaxis = eye_longaxis.sel(frame=step).values
            current_shortaxis = eye_shortaxis.sel(frame=step).values
            current_centX = eye_centX.sel(frame=step).values
            current_centY = eye_centY.sel(frame=step).values

            # some configuration
            meanr = 0.5 * (current_longaxis + current_shortaxis) # mean radius
            r = range(int(meanr - ranger), int(meanr + ranger)) # range of values over mean radius (meanr)
            pupil_edge = np.zeros([360, len(r)]) # empty array that the calculated edge of the pupil will be put into

            rad_range = np.deg2rad(np.arange(360))
            # get cross-section of pupil at each angle 1-360 and fit to sigmoid
            for i in range(0, len(r)):
                pupil_edge[:,i] = eye_frame[((current_centY + r[i]*(np.sin(rad_range))).astype(int),(current_centX + r[i]*(np.cos(rad_range))).astype(int))]
            d = pupil_edge[:,:]

            # apply sigmoid fit with multiprocessing
            param_mp = [pool.apply_async(sigm_fit_mp, args=(d[n,:],)) for n in range(360)]
            params_output = [result.get() for result in param_mp]

            # apply sigmoid fit without multiprocessing
            # params_output = []
            # for n in range(360):
            #     params_output.append(sigm_fit_mp(d[n,:]))

            # unpack outputs of sigmoid fit
            params = []; ci = []
            for vals in params_output:
                params.append(vals[0])
                ci.append(vals[1])
            params = np.stack(params); ci = np.stack(ci)

            # extract radius variable from parameters
            rfit = params[:,2] - 1

            # if confidence interval in estimate is > fit_thresh pix, set to to NaN
            ci_temp = (ci[:,0] > 5) | (ci[:,1] > 5)  | (ci[:,2]>0.75)
            rfit[ci_temp] = np.nan

            # remove if luminance goes the wrong way (e.g. from reflectance)
            params_temp1 = (params[:,1] - params[:,0]) < 10
            params_temp2 = params[:,1] > 250
            rfit[params_temp1] = np.nan
            rfit[params_temp2] = np.nan

            try:
                # median filter
                rfit_filt = signal.medfilt(rfit,5)

                # subtract baseline because our points aren't perfectly centered on ellipse
                filtsize = 31
                # rfit_conv = rfit - np.convolve(rfit_interp, np.ones(filtsize)/filtsize, mode='same')
                rfit_conv = rfit_filt - convolve(rfit_filt, np.ones(filtsize)/filtsize, boundary='wrap')

            except ValueError as e: # in case every value in rfit is NaN
                rift = np.nan*np.zeros(360)
                rfit_conv = np.nan*np.zeros(360)
        except (KeyError, ValueError) as e:
            key_error_count = key_error_count + 1
            rift = np.nan*np.zeros(360)
            rfit_conv = np.nan*np.zeros(360)

        # get rid of outlier points
        rfit_conv[np.abs(rfit_conv)>1.5] = np.nan

        # save out pupil edge data into one xarray for all frames
        if step == 0:
            rfit_conv_xr = xr.DataArray(rfit_conv)
            rfit_conv_xr['frame'] = step
            rfit_conv_xr = xr.DataArray.rename(rfit_conv_xr, {'dim_0':'deg'})

            rfit_xr = xr.DataArray(rfit)
            rfit_xr['frame'] = step
            rfit_xr = xr.DataArray.rename(rfit_xr, {'dim_0':'deg'})
        if step > 0:
            rfit_conv_temp = xr.DataArray(rfit_conv)
            rfit_conv_temp['frame'] = step
            rfit_conv_temp = xr.DataArray.rename(rfit_conv_temp, {'dim_0':'deg'})
            rfit_conv_xr = xr.concat([rfit_conv_xr, rfit_conv_temp], dim='frame', fill_value=np.nan)

            rfit_temp = xr.DataArray(rfit)
            rfit_temp['frame'] = step
            rfit_temp = xr.DataArray.rename(rfit_temp, {'dim_0':'deg'})
            rfit_xr = xr.concat([rfit_xr, rfit_temp], dim='frame', fill_value=np.nan)

    # threshold out any frames with large or small rfit_conv distributions
    for frame in range(0,np.size(rfit_conv_xr,0)):
        if np.min(rfit_conv_xr[frame,:]) < -10 or np.max(rfit_conv_xr[frame,:]) > 10:
            rfit_conv_xr[frame,:] = np.nan

    # correlation across first minute of recording
    timepoint_corr_rfit = pd.DataFrame(rfit_conv_xr.isel(frame=range(0,3600)).values).T.corr()

    # plot the correlation matrix of rfit over all timepoints
    if config['parameters']['outputs_and_visualization']['save_figs'] is True:
        plt.figure()
        fig, ax = plt.subplots()
        im = ax.imshow(timepoint_corr_rfit)
        ax.set_title('correlation of radius fit during first min. of recording')
        fig.colorbar(im, ax=ax)
        pdf.savefig()
        plt.close()

    n = np.size(rfit_conv_xr.values, 0)
    pupil_update = rfit_conv_xr.values.copy()
    total_shift = np.zeros(n); peak = np.zeros(n)
    c = total_shift.copy()
    template = np.nanmean(rfit_conv_xr.values, 0)

    # calculate mean as template
    try:
        template_rfitconv_cc, template_rfit_cc_lags = nanxcorr(rfit_conv_xr[7].values, template, 30)
        template_nanxcorr = True
    except ZeroDivisionError:
        template_nanxcorr = False

    if config['parameters']['outputs_and_visualization']['save_figs'] is True:
        plt.figure()
        plt.plot(template)
        plt.title('mean as template')
        pdf.savefig()
        plt.close()

        if template_nanxcorr is True:
            plt.plot(template_rfitconv_cc)
            plt.title('rfit_conv template cross correlation')
            pdf.savefig()
            plt.close()

    # xcorr of two random timepoints
    if config['parameters']['outputs_and_visualization']['save_figs'] is True:
        try:
            t0 = np.random.random_integers(0,totalF-1); t1 = np.random.random_integers(0,totalF-1)
            rfit2times_cc, rfit2times_lags = nanxcorr(rfit_conv_xr.isel(frame=t0).values, rfit_conv_xr.isel(frame=t1).values, 10)
            rand_frames = True
        except ZeroDivisionError:
            rand_frames = False
        if rand_frames is True:
            plt.figure()
            plt.plot(rfit2times_cc, 'b-')
            plt.title('nanxcorr of frames ' + str(t0) + ' and ' + str(t1))
            pdf.savefig()
            plt.close()

    num_rfit_samples_to_plot = 100
    ind2plot_rfit = sorted(np.random.randint(0,totalF-1,num_rfit_samples_to_plot))

    # iterative fit to alignment
    # start with mean as template
    # on each iteration, shift individual frames to max xcorr with template
    # then recalculate mean template
    print('doing iterative fit for alignment of each frame')
    for rep in tqdm(range(0,12)): # twelve iterations
        # for each frame, get correlation, and shift
        for frame_num in range(0,n): # do all frames
            try:
                xc, lags = nanxcorr(template, pupil_update[frame_num,:], 20)
                c[frame_num] = np.amax(xc) # value of max
                peaklag = np.argmax(xc) # position of max
                peak[frame_num] = lags[peaklag]
                total_shift[frame_num] = total_shift[frame_num] + peak[frame_num]
                pupil_update[frame_num,:] = np.roll(pupil_update[frame_num,:], int(peak[frame_num]))
            except ZeroDivisionError:
                total_shift[frame_num] = np.nan
                pupil_update[frame_num,:] = np.nan

        template = np.nanmean(pupil_update, axis=0) # update template

        if config['parameters']['outputs_and_visualization']['save_figs'] is True:
            # plot template with pupil_update for each iteration of fit
            plt.figure()
            plt.title('pupil_update of rep='+str(rep)+' in iterative fit')
            plt.plot(pupil_update[ind2plot_rfit,:].T, alpha=0.2)
            plt.plot(template, 'k--', alpha=0.8)
            pdf.savefig()
            plt.close()

            # histogram of correlations
            plt.figure()
            plt.title('correlations of rep='+str(rep)+' in iterative fit')
            plt.hist(c[c>0], bins=300) # gets rid of NaNs in plot
            pdf.savefig()
            plt.close()

    win = 5
    shift_nan = -total_shift
    shift_nan[c < 0.35] = np.nan
    shift_nan = shift_nan - np.nanmedian(shift_nan)
    shift_nan[shift_nan >= 20] = np.nan; shift_nan[shift_nan <= -20] = np.nan # get rid of very large shifts
    shift_smooth = signal.medfilt(shift_nan,3)  # median filt to get rid of outliers
    shift_smooth = convolve(shift_nan, np.ones(win)/win)  # convolve to smooth and fill in nans
    shift_smooth = shift_smooth - np.nanmedian(shift_smooth)

    if config['parameters']['outputs_and_visualization']['save_figs'] is True:
        plt.figure()
        plt.plot(shift_nan)
        plt.title('shift nan')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(shift_smooth)
        plt.title('shift smooth')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(shift_smooth[:3600])
        plt.title('shift smooth for first 1min of recording')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(shift_smooth, linewidth = 4, label = 'shift_smooth')
        plt.plot(-total_shift,linewidth=1, alpha = 0.5, label = 'raw total_shift')
        plt.legend()
        plt.title('shift_smooth and raw total shift')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(rfit_xr.isel(frame=ind2plot_rfit).T, alpha=0.2)
        plt.plot(np.nanmean(rfit_xr.T,1), 'b--', alpha=0.8)
        plt.title('rfit for 100 random frames')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(rfit_conv_xr.isel(frame=ind2plot_rfit).T, alpha=0.2)
        plt.plot(np.nanmean(rfit_conv_xr.T,1), 'b--', alpha=0.8)
        plt.title('rfit_conv for 100 random frames')
        pdf.savefig()
        plt.close()

        # plot of 5 random frames' rfit_conv
        plt.figure()
        fig, axs = plt.subplots(5,1)
        axs = axs.ravel()
        for i in range(0,5):
            rand_num = np.random.randint(0,totalF-1)
            axs[i].plot(rfit_conv_xr.isel(frame=rand_num))
            axs[i].set_title(('rfit conv; frame ' + str(rand_num)))
        pdf.savefig()
        plt.close()

    shift_smooth1 = xr.DataArray(shift_smooth, dims=['frame'])

    if config['parameters']['outputs_and_visualization']['save_avi_vids'] is True:
        eyevid = cv2.VideoCapture(eyevidpath)
        vidsavepath = os.path.join(config['recording_path'], str(trial_name + '_pupil_rotation_rep' + str(rep) + '_' + side_letter + '.avi'))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vidout = cv2.VideoWriter(vidsavepath, fourcc, 60.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        if config['parameters']['outputs_and_visualization']['num_save_frames'] > int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)):
            num_save_frames = int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            num_save_frames = config['parameters']['outputs_and_visualization']['num_save_frames']

        print('plotting pupil rotation on eye video')
        for step in tqdm(range(0,num_save_frames)):
            eye_ret, eye_frame = eyevid.read()
            eye_frame0 = eye_frame.copy()

            if not eye_ret:
                break

            # get ellipse parameters for this time
            current_theta = eye_theta.sel(frame=step).values
            current_phi = eye_phi.sel(frame=step).values
            current_longaxis = eye_longaxis.sel(frame=step).values
            current_shortaxis = eye_shortaxis.sel(frame=step).values
            current_centX = eye_centX.sel(frame=step).values
            current_centY = eye_centY.sel(frame=step).values

            # plot the ellipse edge
            rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
            for deg_th in range(0,360):
                rad_th = rad_range[deg_th]
                edge_x = np.round(current_centX+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.cos(rad_th))
                edge_y = np.round(current_centY+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.sin(rad_th))
                if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                    eye_frame1 = cv2.circle(eye_frame, (int(edge_x),int(edge_y)), 1, (235,52,155), thickness=-1)

            # plot the rotation of the eye as a vertical line made up of many circles
            for d in np.linspace(-0.5,0.5,100):
                rot_x = np.round(current_centX + d*(np.rad2deg(np.cos(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                rot_y = np.round(current_centY + d*(np.rad2deg(np.sin(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                    eye_frame1 = cv2.circle(eye_frame1, (int(rot_x),int(rot_y)),1,(255,255,255),thickness=-1)

            # plot the center of the eye on the frame as a larger dot than the others
            if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:
                eye_frame1 = cv2.circle(eye_frame1, (int(current_centX),int(current_centY)),3,(0,255,0),thickness=-1)

            frame_out = np.concatenate([eye_frame0, eye_frame1], axis=1)

            vidout.write(frame_out)

        vidout.release()

    # temporary: save pupil rotation values to csv in case of error during xarray formatting
    shift_smooth_pd = pd.DataFrame(shift_smooth)
    # shift_smooth_pd.to_csv(os.path.join(config['recording_path'], str(trial_name + '_shift_smooth.csv')), index=False)
    shift = xr.DataArray(shift_smooth_pd, dims=['frame','shift'])
    print('key/value error count during sigmoid fit: ' + str(key_error_count))

    # plotting omega on some random frames to be saved into the pdf
    eyevid = cv2.VideoCapture(eyevidpath)
    rand_frame_nums = list(np.random.randint(0,int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)), size=20))
    for step in rand_frame_nums:
        eyevid.set(cv2.CAP_PROP_POS_FRAMES, step)
        eye_ret, eye_frame = eyevid.read()
        if not eye_ret:
            break
        plt.subplots(2,2)
        plt.subplot(221)
        plt.imshow(eye_frame)
        # get ellipse parameters for this time
        current_theta = eye_theta.sel(frame=step).values
        current_phi = eye_phi.sel(frame=step).values
        current_longaxis = eye_longaxis.sel(frame=step).values
        current_shortaxis = eye_shortaxis.sel(frame=step).values
        current_centX = eye_centX.sel(frame=step).values
        current_centY = eye_centY.sel(frame=step).values
        # plot the ellipse edge
        rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
        plt.subplot(222)
        plt.imshow(eye_frame)
        for deg_th in range(0,360):
            rad_th = rad_range[deg_th]
            edge_x = np.round(current_centX+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.cos(rad_th))
            edge_y = np.round(current_centY+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.sin(rad_th))
            if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                plt.plot(edge_x, edge_y, color='orange', marker='.',markersize=1,alpha=0.1)
        # plot the rotation of the eye as a vertical line made up of many circles
        plt.subplot(223)
        plt.imshow(eye_frame)
        for d in np.linspace(-0.5,0.5,100):
            rot_x = np.round(current_centX + d*(np.rad2deg(np.cos(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
            rot_y = np.round(current_centY + d*(np.rad2deg(np.sin(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
            if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                plt.plot(rot_x, rot_y, color='white',marker='.',markersize=1,alpha=0.1)

        plt.subplot(223)
        plt.imshow(eye_frame)
        # plot the center of the eye on the frame as a larger dot than the others
        if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:
            plt.plot(int(current_centX),int(current_centY), color='blue', marker='o')

        plt.subplot(224)
        plt.imshow(eye_frame)
        for deg_th in range(0,360):
            rad_th = rad_range[deg_th]
            edge_x = np.round(current_centX+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.cos(rad_th))
            edge_y = np.round(current_centY+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.sin(rad_th))
            if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                plt.plot(edge_x, edge_y, color='orange', marker='.',markersize=1,alpha=0.1)
        for d in np.linspace(-0.5,0.5,100):
            rot_x = np.round(current_centX + d*(np.rad2deg(np.cos(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
            rot_y = np.round(current_centY + d*(np.rad2deg(np.sin(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
            if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                plt.plot(rot_x, rot_y, color='white',marker='.',markersize=1,alpha=0.1)
        # plot the center of the eye on the frame as a larger dot than the others
        if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:
            plt.plot(int(current_centX),int(current_centY), color='blue', marker='o')

        pdf.savefig()
        plt.close()

    pdf.close()

    return rfit_xr, rfit_conv_xr, shift
