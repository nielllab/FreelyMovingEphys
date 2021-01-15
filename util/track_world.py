"""
track_world.py

tracking world camera and finding pupil rotation

Jan. 10, 2021
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
from util.format_data import h5_to_xr

def smooth_tracking(y, box_pts=3):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def track_LED(config):
    # DLC tracking
    calib = config['calibration']
    dlc_config_eye = calib['eye_LED_config']
    dlc_config_world = calib['world_LED_config']
    led_dir = os.path.join(config['data_path'], config['LED_dir_name'])
    led_dir_avi = find('*IR*.avi', led_dir)
    led_dir_csv = find('*IR*BonsaiTSformatted.csv', led_dir)
    if led_dir_avi == []:
        led_dir_avi = find('*IR*.avi', config['data_path'])
        led_dir_csv = find('*IR*BonsaiTSformatted.csv', config['data_path'])
        led_dir_h5 = find('*IR*.h5', config['data_path'])
    # get the trial name
    t_name = os.path.split('_'.join(led_dir_avi[0].split('_')[:-1]))[1]
    # find the correct eye anbd world video and time files
    eye_csv = [i for i in led_dir_csv if 'REYE' in i and 'formatted' in i][0]
    eye_avi = [i for i in led_dir_avi if 'REYE' in i and 'deinter' in i][0]
    world_csv = [i for i in led_dir_csv if 'WORLD' in i and 'formatted' in i][0]
    world_avi = [i for i in led_dir_avi if 'WORLD' in i and 'calib' in i][0]
    # generate .h5 files
    run_DLC_on_LED(dlc_config_world, world_avi)
    run_DLC_on_LED(dlc_config_eye, eye_avi)
    # then, get the h5 files for this trial that were just written to file
    led_dir_h5 = find('*IR*.h5', led_dir)
    if led_dir_h5 == []:
        led_dir_h5 = find('*IR*.h5',config['data_path'])
    world_h5 = [i for i in led_dir_h5 if 'WORLD' in i and 'calib' in i][0]
    eye_h5 = [i for i in led_dir_h5 if 'REYE' in i and 'deinter' in i][0]
    # format everything into an xarray
    eyexr = h5_to_xr(eye_h5, eye_csv, 'REYE', config=config)
    worldxr = h5_to_xr(world_h5, world_csv, 'WORLD', config=config) # format in xarray
    # save out the paramters in nc files
    eyexr.to_netcdf(os.path.join(config['data_path'], str('led_eye_positions.nc')))
    worldxr.to_netcdf(os.path.join(config['data_path'], str('led_world_positions.nc')))
    # then make some plots in a pdf
    if config['save_figs'] is True:
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(config['data_path'], (t_name + 'LED_tracking.pdf')))
        
        eye_x = eyexr.sel(point_loc='light_x')
        plt.figure()
        plt.plot(eye_x); plt.title('light x position in eye')
        plt.ylabel('eye x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        eye_y = eyexr.sel(point_loc='light_y')
        plt.figure()
        plt.plot(eye_y); plt.title('light y position in eye')
        plt.ylabel('eye y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        world_x = worldxr.sel(point_loc='light_x')
        plt.figure()
        plt.plot(world_x); plt.title('light x position in worldcam')
        plt.ylabel('world x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        world_y = worldxr.sel(point_loc='light_y')
        plt.figure()
        plt.plot(world_y); plt.title('light y position in worldcam')
        plt.ylabel('world y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()

        # threshold out frames with low likelihood
        # seems to work well for thresh=0.99
        eye_x[eyexr.sel(point_loc='light_likelihood')<config['lik_thresh_strict']] = np.nan
        eye_y[eyexr.sel(point_loc='light_likelihood')<config['lik_thresh_strict']] = np.nan
        world_x[worldxr.sel(point_loc='light_likelihood')<config['lik_thresh_strict']] = np.nan
        world_y[worldxr.sel(point_loc='light_likelihood')<config['lik_thresh_strict']] = np.nan
        # eliminate frames in which there is very little movementin the worldcam (movements should be large!)
        orig_world_x = world_x.copy(); orig_world_y = world_y.copy()
        world_x = world_x[:-1]; world_y = world_y[:-1]
        world_x[np.logical_and(np.diff(orig_world_x)<1,np.diff(orig_world_x)>-1)] = np.nan
        world_y[np.logical_and(np.diff(orig_world_y)<1,np.diff(orig_world_y)>-1)] = np.nan

        plt.figure()
        plt.plot(eye_x); plt.title('light x position in eye (thresh applied)')
        plt.ylabel('eye x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(eye_y); plt.title('light y position in eye (thresh applied)')
        plt.ylabel('eye y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(world_x); plt.title('light x position in worldcam (thresh applied)')
        plt.ylabel('world x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(world_y); plt.title('light y position in worldcam (thresh applied)')
        plt.ylabel('world y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()

        # apply a smoothing convolution
        eye_x = smooth_tracking(eye_x); eye_y = smooth_tracking(eye_y)
        world_x = smooth_tracking(world_x); world_y = smooth_tracking(world_y)

        plt.figure()
        plt.plot(eye_x); plt.title('light x position in eye (conv applied)')
        plt.ylabel('eye x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(eye_y); plt.title('light y position in eye (conv applied)')
        plt.ylabel('eye y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(world_x); plt.title('light x position in worldcam (conv applied)')
        plt.ylabel('world x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(world_y); plt.title('light y position in worldcam (conv applied)')
        plt.ylabel('world y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()

        # plot eye vs world for x and y
        diff_in_len = len(world_x) - len(eye_x)
        plt.subplots(1,2)
        plt.subplot(121)
        plt.plot(eye_x,world_x[:-diff_in_len], '.')
        plt.ylabel('world x'); plt.xlabel('eye x')
        plt.title('x in eye vs world')
        plt.subplot(122)
        plt.plot(eye_y, world_y[:-diff_in_len], '.')
        plt.ylabel('world y'); plt.xlabel('eye y')
        plt.title('y in eye vs world')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.subplots(1,2)
        plt.subplot(121)
        plt.plot(world_x,world_y,'.')
        plt.ylabel('world y'); plt.xlabel('world x')
        plt.title('world x vs y')
        plt.subplot(122)
        plt.plot(eye_x, eye_y,'.')
        plt.ylabel('eye y'); plt.xlabel('eye x')
        plt.title('eye x vs y')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.subplots(1,2)
        plt.subplot(121)
        plt.plot(eye_x,world_y[:-diff_in_len],'.')
        plt.ylabel('world y'); plt.xlabel('eye x')
        plt.title('eye x vs world y')
        plt.subplot(122)
        plt.plot(eye_y, world_x[:-diff_in_len],'.')
        plt.ylabel('world x'); plt.xlabel('eye y')
        plt.title('eye y vs world x')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        pdf.close()
    
    if config['save_avi_vids'] is True:
        plot_IR_track(world_avi, worldxr, eye_avi, eyexr, t_name, config)
    
    print('done preprocessing IR LED calibration videos')

def plot_IR_track(world_vid, world_dlc, eye_vid, eye_dlc, trial_name, config):
    
    print('plotting avi of IR LED tracking')

    savepath = os.path.join(config['data_path'], (trial_name + '_IR_LED_tracking.avi'))
    
    world_vid_read = cv2.VideoCapture(world_vid)
    w_width = int(world_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
    w_height = int(world_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    eye_vid_read = cv2.VideoCapture(eye_vid)
    e_width = int(eye_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
    e_height = int(eye_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (e_width*2, e_height))

    if config['num_save_frames'] > int(world_vid_read.get(cv2.CAP_PROP_FRAME_COUNT)):
        num_save_frames = int(world_vid_read.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        num_save_frames = config['num_save_frames']

    for step in tqdm(range(0,num_save_frames)):
        w_ret, w_frame = world_vid_read.read()
        e_ret, e_frame = eye_vid_read.read()

        for ret in [w_ret, e_ret]:
            if not ret:
                break
        
        try:
            e_pt = eye_dlc.sel(frame=step)
            eye_pt_cent = (int(e_pt.sel(point_loc='light_x').values), int(e_pt.sel(point_loc='light_y').values))
            if e_pt.sel(point_loc='light_likelihood').values < config['lik_thresh_strict']: # bad points in red
                e_frame = cv2.circle(e_frame, eye_pt_cent, 8, (0,0,255), 1)
            elif e_pt.sel(point_loc='light_likelihood').values >= config['lik_thresh_strict']: # good points in green
                e_frame = cv2.circle(e_frame, eye_pt_cent, 8, (0,255,0), 1)
        except ValueError:
            pass
                
        try:
            w_pt = world_dlc.sel(frame=step)
            world_pt_cent = (int(w_pt.sel(point_loc='light_x').values), int(w_pt.sel(point_loc='light_y').values))
            if w_pt.sel(point_loc='light_likelihood').values < config['lik_thresh_strict']: # bad points in red
                w_frame = cv2.circle(w_frame, world_pt_cent, 8, (0,0,255), 1)
            elif w_pt.sel(point_loc='light_likelihood').values >= config['lik_thresh_strict']: # good points in green
                w_frame = cv2.circle(w_frame, world_pt_cent, 8, (0,255,0), 1)
        except ValueError:
            pass
                
        plotted = np.concatenate([e_frame, w_frame], axis=1)

        out_vid.write(plotted)

    out_vid.release()

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
