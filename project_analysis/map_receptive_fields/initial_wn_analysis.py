"""
initial_wn_analysis.py

run minimal analysis needed to get receptive fields in the worldcam

Jan. 21, 2021
"""

# package imports
import argparse, json, sys, os, subprocess, shutil
import cv2
import pandas as pd
import deeplabcut
import numpy as np
import xarray as xr
import warnings
import tkinter as tk
from tkinter import filedialog
from glob import glob
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from glob import glob
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages
# module imports
from util.params import extract_params
from util.format_data import h5_to_xr, format_frames, safe_xr_merge
from util.paths import find, check_path
from util.time import open_time, merge_xr_by_timestamps
from util.track_topdown import topdown_tracking, head_angle1, plot_top_vid, body_props, body_angle
from util.track_eye import plot_eye_vid, eye_tracking, find_pupil_rotation
from util.track_world import adjust_world, track_LED
from util.ephys import format_spikes
from util.track_ball import ball_tracking
from util.track_side import side_angle, side_tracking
from util.track_imu import read_8ch_imu, convert_acc_gyro
from util.deinterlace import deinterlace_data
from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids
from project_analysis.ephys.ephys_figures import *

def quick_whitenoise_analysis(wn_path):
    temp_config = {
        'data_path': wn_path,
        'save_path': wn_path,
        'flip_eye_during_deinter': True,
        'flip_world_during_deinter': True,
        'calibration': {
            'world_checker_npz': 'T:/freely_moving_ephys/camera_calibration_params/world_checkerboard_calib.npz'
        },
        'save_nc_vids': True,
        'use_BonsaiTS': True,
        'dwnsmpl': 0.25,
        'ephys_sample_rate': 30000,
        'run_with_form_time': True
    } # 'G:/freely_moving_ephys/ephys_recordings_copy_011721/calibration_params/world_checkerboard_calib.npz'

    world_vids = glob(os.path.join(wn_path, '*WORLD.avi'))
    world_times = glob(os.path.join(wn_path, '*WORLD_BonsaiTS.csv'))

    deinterlace_data(temp_config, world_vids, world_times)
    calibrate_new_world_vids(temp_config)

    trial_units = []; name_check = []; path_check = []
    for avi in find('*.avi', temp_config['data_path']):
        bad_list = ['plot','IR','rep11','betafpv','side_gaze'] # don't use trials that have these strings in their path
        if temp_config['run_with_form_time'] is True:
            if all(bad not in avi for bad in bad_list):
                split_name = avi.split('_')[:-1]
                trial = '_'.join(split_name)
                path_to_trial = os.path.join(os.path.split(trial)[0])
                trial_name = os.path.split(trial)[1]
        elif temp_config['run_with_form_time'] is False:
            if all(bad not in avi for bad in bad_list):
                trial_path_noext = os.path.splitext(avi)[0]
                path_to_trial, trial_name_long = os.path.split(trial_path_noext)
                trial_name = '_'.join(trial_name_long.split('_')[:3])
        try:
            if trial_name not in name_check:
                trial_units.append([path_to_trial, trial_name])
                path_check.append(path_to_trial); name_check.append(trial_name)
        except UnboundLocalError:
            pass

    for trial_unit in trial_units:
        temp_config['trial_path'] = trial_unit[0]
        t_name = trial_unit[1]
        trial_cam_csv = find(('*BonsaiTS*.csv'), temp_config['trial_path'])
        trial_cam_avi = find(('*.avi'), temp_config['trial_path'])

        trial_cam_csv = [x for x in trial_cam_csv if x != []]
        trial_cam_avi = [x for x in trial_cam_avi if x != []]

        # filter the list of files for the current trial to get the world view of this side
        world_csv = [i for i in trial_cam_csv if 'WORLD' in i and 'formatted' in i][0]
        world_avi = [i for i in trial_cam_avi if 'WORLD' in i and 'calib' in i][0]
        # make an xarray of timestamps without dlc points, since there aren't any for world camera
        worlddlc = h5_to_xr(pt_path=None, time_path=world_csv, view=('WORLD'), config=temp_config)
        worlddlc.name = 'WORLD_times'
        # make xarray of video frames
        if temp_config['save_nc_vids'] is True:
            xr_world_frames = format_frames(world_avi, temp_config); xr_world_frames.name = 'WORLD_video'
        # merge but make sure they're not off in lenght by one value, which happens occasionally
        print('saving nc file of world view...')
        if temp_config['save_nc_vids'] is True:
            trial_world_data = safe_xr_merge([worlddlc, xr_world_frames])
            trial_world_data.to_netcdf(os.path.join(temp_config['trial_path'], str(t_name+'_world.nc')), engine='netcdf4', encoding={'WORLD_video':{"zlib": True, "complevel": 4}})
        elif temp_config['save_nc_vids'] is False:
            worlddlc.to_netcdf(os.path.join(temp_config['trial_path'], str(t_name+'_world.nc')))

        print('generating ephys plots')

        pdf = PdfPages(os.path.join(wn_path, (t_name + '_prelim_wn_figures.pdf')))
    
        # generate summary plot
        samprate = 30000  # ephys sample rate
        ephys_file_path = glob(os.path.join(wn_path, '*_ephys_merge.json'))[0]
        world_file_path = glob(os.path.join(wn_path, '*_world.nc'))[0]
        world_data = xr.open_dataset(world_file_path)
        world_vid_raw = np.uint8(world_data['WORLD_video'])
        ephys_data = pd.read_json(ephys_file_path)
        ephysT0 = ephys_data.iloc[0,12]
        worldT = world_data.timestamps - ephysT0

        ephys_data['spikeTraw'] = ephys_data['spikeT'].copy()

        offset0 = 0.1
        drift_rate = -0.1/1000

        for i in ephys_data.index:
            ephys_data.at[i,'spikeT'] = np.array(ephys_data.at[i,'spikeTraw']) - (offset0 + np.array(ephys_data.at[i,'spikeTraw']) *drift_rate)
        goodcells = ephys_data.loc[ephys_data['group']=='good']
      
        if worldT[0]<-600:
            worldT = worldT + 8*60*60
        
        # resize worldcam to make more manageable
        world_vid = world_vid_raw.copy()

        cam_gamma = 2
        world_norm = (world_vid/255)**cam_gamma
        std_im = np.std(world_norm,axis=0)
        std_im[std_im<10/255] = 10/255
        img_norm = (world_norm-np.mean(world_norm,axis=0))/std_im
        img_norm = img_norm * (std_im>20/255)

        contrast = np.empty(worldT.size)
        for i in range(worldT.size):
            contrast[i] = np.std(img_norm[i,:,:])
        newc = interp1d(worldT,contrast,fill_value="extrapolate")

        dt = 0.025
        t = np.arange(0, np.max(worldT),dt)
        ephys_data['rate'] = np.nan
        ephys_data['rate'] = ephys_data['rate'].astype(object)
        for i,ind in enumerate(ephys_data.index):
            ephys_data.at[ind,'rate'], bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
        ephys_data['rate']= ephys_data['rate']/dt
        goodcells = ephys_data.loc[ephys_data['group']=='good']
        n_units = len(goodcells)
        
        contrast_interp = newc(t[0:-1])

        plt.figure()
        plt.plot(worldT[0:12000],contrast[0:12000])
        plt.xlabel('time')
        plt.ylabel('contrast')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(t[0:600],contrast_interp[0:600])
        plt.xlabel('secs'); plt.ylabel('contrast')
        pdf.savefig()
        plt.close()

        spike_corr = 1 #+ 0.125/1200  # correction factor for ephys timing drift, but it's now corrected in spikeT and doesn't need to be manually reset

        img_norm[img_norm<-2] = -2
        movInterp = interp1d(worldT,img_norm,axis=0, fill_value="extrapolate") # added extrapolate for cases where x_new is below interpolation range

        plt.figure()
        plt.plot(np.diff(worldT)); plt.xlabel('frame'); plt.ylabel('deltaT'); plt.title('world cam')
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots(figsize=(20,8))
        ax.fontsize = 20
        for i,ind in enumerate(goodcells.index):
            plt.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25)
            plt.xlim(0, 10); plt.xlabel('secs',fontsize = 20); plt.ylabel('unit #',fontsize=20)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        pdf.savefig()
        plt.close()

        # calculate contrast - response functions
        # mean firing rate in timebins correponding to contrast ranges
        resp = np.empty((n_units,12))
        crange = np.arange(0,1.2,0.1)
        for i,ind in enumerate(goodcells.index):
            for c,cont in enumerate(crange):
                resp[i,c] = np.mean(goodcells.at[ind,'rate'][(contrast_interp>cont) & (contrast_interp<(cont+0.1))])
        # plot individual contrast response functions in subplots
        ind_contrast_funcs_fig = plot_ind_contrast_funcs(n_units, goodcells, crange, resp)
        pdf.savefig()
        plt.close()

        # calculate spike-triggered average
        staAll, STA_single_lag_fig = plot_STA_single_lag(n_units, img_norm, goodcells, worldT, movInterp)
        pdf.savefig()
        plt.close()

        # print('getting spike-triggered average with range in lags')
        # calculate spike-triggered average
        print('getting spike-triggered average with range in lags')
        # calculate spike-triggered average
        fig = plot_STA_multi_lag(n_units, goodcells, worldT, movInterp)
        pdf.savefig()
        plt.close()

        print('getting spike-triggered variance')
        # calculate spike-triggered variance
        fig = plot_spike_triggered_variance(n_units, goodcells, t, movInterp, img_norm)
        pdf.savefig()
        plt.close()
        

        pdf.close()