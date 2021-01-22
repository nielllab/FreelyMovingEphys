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
# module imports
from util.params import extract_params
from util.format_data import h5_to_xr, format_frames
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
        'dwnsmpl': 0.5,
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
            try:
                trial_world_data = xr.merge([worlddlc, xr_world_frames])
            except ValueError:
                if len(worlddlc) > len(xr_world_frames):
                    trial_world_data = xr.merge([worlddlc[:-1], xr_world_frames])
                elif len(worlddlc) < len(xr_world_frames):
                    trial_world_data = xr.merge([worlddlc, xr_world_frames[:-1]])
            trial_world_data.to_netcdf(os.path.join(temp_config['trial_path'], str(t_name+'_world.nc')), engine='netcdf4', encoding={'WORLD_video':{"zlib": True, "complevel": 9}})
        elif temp_config['save_nc_vids'] is False:
            worlddlc.to_netcdf(os.path.join(temp_config['trial_path'], str(t_name+'_world.nc')))

        print('generating summary plot')
        # generate summary plot
        samprate = 30000  # ephys sample rate
        ephys_file_path = glob(os.path.join(wn_path, '*_ephys_merge.json'))[0]
        world_file_path = glob(os.path.join(wn_path, '*_world.nc'))[0]
        world_data = xr.open_dataset(world_file_path)
        world_vid_raw = np.uint8(world_data['WORLD_video'])
        ephys_data = pd.read_json(ephys_file_path)
        ephysT0 = ephys_data.iloc[0,12]
        worldT = world_data.timestamps - ephysT0

        ephys_data['spikeTraw'] = ephys_data['spikeT']

        offset0 = 0.1
        drift_rate = 0.1/1000

        for i in range(len(ephys_data)):
            ephys_data['spikeT'].iloc[i] = np.array(ephys_data['spikeTraw'].iloc[i]) - (offset0 + np.array(ephys_data['spikeTraw'].iloc[i]) *drift_rate)

        dt = 0.025
        t = np.arange(0, np.max(worldT),dt)
        ephys_data['rate'] = np.nan
        ephys_data['rate'] = ephys_data['rate'].astype(object)
        for i,ind in enumerate(ephys_data.index):
            ephys_data.at[ind,'rate'], bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
        ephys_data['rate']= ephys_data['rate']/dt
        goodcells = ephys_data.loc[ephys_data['group']=='good']
        n_units = len(goodcells)
        
        if worldT[0]<-600:
            worldT = worldT + 8*60*60
        contrast = np.empty(worldT.size)
        newc = interp1d(worldT,contrast)
        
        contrast_interp = newc(t[0:-1])

        # resize worldcam to make more manageable
        sz = world_vid_raw.shape
        downsamp = 0.5
        world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
        for f in range(sz[0]):
            world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
        worldT = world_data.timestamps.copy()

        cam_gamma = 2
        world_norm = (world_vid/255)**cam_gamma
        std_im = np.std(world_norm,axis=0)
        std_im[std_im<10/255] = 10/255
        img_norm = (world_norm-np.mean(world_norm,axis=0))/std_im

        spike_corr = 1 # + 0.125/1200  # correction factor for ephys timing drift, but it's now corrected in spikeT and doesn't need to be manually reset

        movInterp = interp1d(worldT,img_norm,axis=0, fill_value='extrapolate') # added extrapolate for cases where x_new is below interpolation range

        staAll = np.zeros((n_units,np.shape(img_norm)[1],np.shape(img_norm)[2]))
        lag = 0.125
        plt.figure(figsize = (12,np.ceil(n_units/2)))
        for c, ind in enumerate(goodcells.index):
            r = goodcells.at[ind,'rate']
            sta = 0; nsp = 0
            sp = goodcells.at[ind,'spikeT'].copy()
            if c==1:
                ensemble = np.zeros((len(sp),np.shape(img_norm)[1],np.shape(img_norm)[2]))
            for s in sp:
                if (s-lag >5) & ((s-lag)*spike_corr <np.max(worldT)):
                    nsp = nsp+1
                    im = movInterp((s-lag)*spike_corr)
                    if c==1:
                        ensemble[nsp-1,:,:] = im
                    sta = sta+im
            plt.subplot(np.ceil(n_units/4),4,c+1)
            plt.title(str(c) + ' ' + goodcells.at[ind,'KSLabel']  +  ' cont='+ str(goodcells.at[ind,'ContamPct']))
            if nsp > 0:
                sta = sta/nsp
            else:
                sta = np.nan
            #sta[abs(sta)<0.1]=0
            plt.imshow((sta-np.mean(sta)),vmin=-0.3,vmax=0.3,cmap = 'jet')
            staAll[c,:,:] = sta
        plt.tight_layout()

        plt.savefig(os.path.join(wn_path, t_name+'_intial_wn_analysis_summary_plot.png'))
