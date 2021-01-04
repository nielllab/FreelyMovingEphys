"""
track_world.py

tracking world camera and finding pupil rotation

Dec. 28, 2020
"""

# package imports
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

# module imports
from util.time import open_time
from util.paths import find
from util.aux_funcs import nanxcorr
from util.dlc import run_DLC_on_LED

def track_LED(config):
    # are they already deinterlaced--should be, but I should check!
    # DLC tracking
    calib = config['calibration']
    dlc_config_eye = calib['eye_LED_config']
    dlc_config_world = calib['world_LED_config']
    led_dir = os.path.join(config['data_path'], 'hf1_IRspot')
    led_dir_avi = find('*IR*.avi', led_dir)
    led_dir_csv = find('*IR*BonsaiTSformatted.csv', led_dir)
    led_dir_h5 = find('*IR*.h5', led_dir)
    t_name = os.path.split('_'.join(led_dir_avi[0].split('_')[:-1]))[1] # get the trail name
    run_DLC_on_LED(dlc_config_world,'WORLD',led_dir_avi)
    run_DLC_on_LED(dlc_config_eye,'REYE',led_dir_avi)
    # extract params for eye view
    eye_h5 = [i for i in led_dir_h5 if 'REYE' in i and 'deinter' in i][0]
    eye_csv = [i for i in led_dir_csv if 'REYE' in i and 'formatted' in i][0]
    eye_avi = [i for i in led_dir_avi if 'REYE' in i and 'deinter' not in i][0]
    eyexr = h5_to_xr(eye_h5, eye_csv, 'REYE', config=config) # format in xarray
    # next, the world view
    world_h5 = [i for i in led_dir_h5 if 'WORLD' in i and 'calib' in i][0]
    world_csv = [i for i in led_dir_csv if 'WORLD' in i and 'formatted' in i][0]
    world_avi = [i for i in led_dir_avi if 'WORLD' in i and 'calib' not in i][0]
    worldxr = h5_to_xr(world_h5, world_csv, 'WORLD', config=config) # format in xarray
    # save out the paramters in nc files
    out = xr.concat([eyexr, worldxr],dim='frame')
    out.to_netcdf(os.path.join(config['trial_path'], str('led_positions.nc')))
    # then make some plots in a pdf
    # if config['save_figs'] is True:
    #     pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(config['trial_path'], (trial_name + 'LED_tracking.pdf')))
    #     # plot of x/y of eye reflection position and world position over time
    #     plt.figure()
    #     plt.plot()
    #     pdf.close()

# basic world shifting without pupil rotation
def adjust_world(data_path, file_name, eyeext, topext, worldext, eye_ds, savepath):
    # get eye data out of dataset
    eye_pts = eye_data.raw_pt_values
    eye_ell_params = eye_data.ellipse_param_values

    # find the needed files from path and trial key
    top1vidpath = os.path.join(data_path, file_name) + '_' + topext + '.avi'
    eyevidpath = os.path.join(data_path, file_name) + '_' + eyeext + '.avi'
    worldvidpath = os.path.join(data_path, file_name) + '_' + worldext + '.avi'
    top1timepath = os.path.join(data_path, file_name) + '_' + topext +'_BonsaiTSformatted.csv'
    eyetimepath = os.path.join(data_path, file_name) + '_' + eyeext +'_BonsaiTSformatted.csv'
    worldtimepath = os.path.join(data_path, file_name) + '_' + worldext +'_BonsaiTSformatted.csv'

    # create save directory if it does not already exist
    fig_dir = savepath + '/' + file_name + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # open time files
    eyeTS = open_time(eyetimepath)
    worldTS = open_time(worldtimepath)
    topTS = open_time(top1timepath)

    # interpolate ellipse parameters to worldcam timestamps
    eye_ell_interp_params = eye_ell_params.interp_like(xr.DataArray(worldTS), method=interp_method)

    # the very first timestamp
    start_time = min(eyeTS[0], worldTS[0], topTS[0])

    eye_theta = eye_ell_interp_params.sel(ellipse_params='theta')
    eye_phi = eye_ell_interp_params.sel(ellipse_params='phi')
    eye_longaxis= eye_ell_interp_params.sel(ellipse_params='longaxis')
    eye_shortaxis = eye_ell_interp_params.sel(ellipse_params='shortaxis')

    eye_raw_theta = eye_ell_params.sel(ellipse_params='theta')
    eye_raw_phi = eye_ell_params.sel(ellipse_params='phi')
    eye_raw_longaxis= eye_ell_params.sel(ellipse_params='longaxis')
    eye_raw_shortaxis = eye_ell_params.sel(ellipse_params='shortaxis')

    eyeTSminusstart = [(t-start_time).seconds for t in eyeTS]
    worldTSminusstart = [(t-start_time).seconds for t in worldTS]

    # saftey check
    plt.subplots(2, 1, figsize=(15, 15))
    plt.subplot(211)
    plt.title('raw/interpolated theta for ' + eyeext + ' side')
    plt.plot(eyeTSminusstart, eye_raw_theta.values, 'r--', label='raw theta')
    plt.plot(worldTSminusstart[:-1], eye_theta.values, 'b-', label='interp theta')
    plt.subplot(212)
    plt.title('raw/interpolated phi for ' + eyeext + ' side')
    plt.plot(eyeTSminusstart, eye_raw_phi.values, 'r--', label='raw phi')
    plt.plot(worldTSminusstart[:-1], eye_phi.values, 'b-', label='interp phi')
    plt.savefig(fig_dir + eyeext + 'rawinterp_phitheta.png', dpi=300)
    plt.close()

    worldvid = cv2.VideoCapture(worldvidpath)
    topvid = cv2.VideoCapture(top1vidpath)
    eyevid = cv2.VideoCapture(eyevidpath)

    # setup the file to save out of this
    savepath = os.path.join(fig_dir, str(file_name + '_worldshift_' + eyeext + '.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))*2))

    set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while(1):
        # read the frame for this pass through while loop
        wrld_ret, wrld_frame = worldvid.read()
        eye_ret, eye_frame = eyevid.read()
        top_ret, top_frame = topvid.read()

        if not wrld_ret:
            break
        if not eye_ret:
            break
        if not top_ret:
            break

        # create empty frame to shift the world in
        wrld_shift = np.zeros(set_size)

        # limit range to shift over
        phi_max = np.where(eye_phi > 15, -15, 15)
        theta_max = np.where(eye_theta > 20, -20, 20)

        # insert world frame into world shift with offset
        if np.isnan(eye_theta) is False and np.isnan(eye_phi) is False:
            wrld_shift[(range(61,180) - np.round(phi_max * pix_deg)), (range(81,240) - np.round(theta_max * pix_deg))] = wrld_frame

        # resize the frames before plotting
        wrld_frame_resz = cv2.resize(wrld_frame, set_size)
        wrld_shift_resz = cv2.resize(np.uint8(wrld_shift), set_size)
        eye_frame_resz = cv2.resize(eye_frame, set_size)
        top_frame_resz = cv2.resize(top_frame, set_size)

        # concat frames together into a 2x2 grid
        a = np.concatenate((cv2.cvtColor(eye_frame_resz, cv2.COLOR_BGR2GRAY), cv2.cvtColor(top_frame_resz, cv2.COLOR_BGR2GRAY)), axis=1)
        b = np.concatenate((cv2.cvtColor(wrld_frame_resz, cv2.COLOR_BGR2GRAY), wrld_shift_resz), axis=1)
        all_vids = np.concatenate((a, b), axis=0)

        out_vid.write(all_vids)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_vid.release()
    cv2.destroyAllWindows()
