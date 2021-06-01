"""
initial_revchecker_analysis.py

run minimal analysis needed to get lfp traces during revchecker stim
"""

from glob import glob
import os, sys
from tqdm import tqdm
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

from util.paths import find, list_subdirs
from util.format_data import h5_to_xr, format_frames, safe_xr_merge
from util.deinterlace import deinterlace_data
from util.calibration import calibrate_new_world_vids
from project_analysis.ephys.ephys_figures import *
from project_analysis.ephys.ephys_utils import *

def quick_revchecker_analysis(rc_path, probe_type):
    temp_config = {
        'data_path': rc_path,
        'flip_eye_during_deinter': True,
        'flip_world_during_deinter': True,
        'calibration': {'world_checker_npz': 'E:/freely_moving_ephys/camera_calibration_params/world_checkerboard_calib.npz'},
        'dwnsmpl': 0.25
    }
    # get lists of worldcam videos
    world_vids = glob(os.path.join(rc_path, '*WORLD.avi'))
    world_times = glob(os.path.join(rc_path, '*WORLD_BonsaiTS.csv'))
    # deinterlace worldcam videos
    deinterlace_data(temp_config, world_vids, world_times)
    # run calibration
    # it might be okay to not run this for this quick analysis
    # (i.e. warped image would still give good seperation in kmeans clustering)
    calibrate_new_world_vids(temp_config)
    # get the path to each recording directory
    recording_name = '_'.join(os.path.splitext(os.path.split([i for i in find('*.avi', temp_config['data_path']) if all(bad not in i for bad in ['plot','IR','rep11','betafpv','side_gaze'])][0])[1])[0].split('_')[:-1])
    print('opening pdf')
    pdf = PdfPages(os.path.join(rc_path, (recording_name + '_prelim_revchecker_figures.pdf')))
    trial_cam_csv = find(('*BonsaiTS*.csv'), temp_config['data_path'])
    trial_cam_avi = find(('*.avi'), temp_config['data_path'])
    trial_cam_csv = [x for x in trial_cam_csv if x != []]
    trial_cam_avi = [x for x in trial_cam_avi if x != []]
    # filter the list of files for the current trial to get the world view of this side
    world_csv = [i for i in trial_cam_csv if 'WORLD' in i and 'formatted' in i][0]
    world_avi = [i for i in trial_cam_avi if 'WORLD' in i and 'calib' in i][0]
    # make an xarray of timestamps without dlc points, since there aren't any for world camera
    worlddlc = h5_to_xr(pt_path=None, time_path=world_csv, view=('WORLD'), config=temp_config)
    worlddlc.name = 'WORLD_times'
    # make xarray of video frames
    xr_world_frames = format_frames(world_avi, temp_config); xr_world_frames.name = 'WORLD_video'
    # merge but make sure they're not off in lenght by one value, which happens occasionally
    print('saving nc file of world view...')
    trial_world_data = safe_xr_merge([worlddlc, xr_world_frames])
    trial_world_data.to_netcdf(os.path.join(temp_config['data_path'], str(recording_name+'_world.nc')), engine='netcdf4', encoding={'WORLD_video':{"zlib": True, "complevel": 4}})
    print('running revchecker analysis')
    print('opening worldcam video and resizing')
    world_data = xr.open_dataset(os.path.join(temp_config['data_path'], str(recording_name+'_world.nc')))
    world_vid_raw = np.uint8(world_data['WORLD_video'])
    # resize worldcam to make more manageable
    sz = world_vid_raw.shape
    if sz[1]>160:
        downsamp = 0.5
        world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
        for f in range(sz[0]):
            world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
    # if the worldcam has already been resized when the NC file was written in preprocessing, don't resize
    else:
        world_vid = world_vid_raw
    world_vid_raw = None # clear large variable
    worldT = world_data.timestamps.copy()
    # read in the binary file of ephys recording
    print('loading ephys binary file and applying filters')
    ephys_binary_file = os.path.join(rc_path, recording_name + '_Ephys.bin')
    lfp_ephys = read_ephys_bin(ephys_binary_file, probe_type, do_remap=True)
    # subtract off average for each channel, then apply bandpass filter
    ephys_center_sub = lfp_ephys - np.mean(lfp_ephys,0)
    filt_ephys = butter_bandpass(ephys_center_sub, lowcut=1, highcut=300, fs=30000, order=6)
    plt.figure()
    plt.title('lfp with bandpass and remapping')
    plt.plot(filt_ephys)
    pdf.savefig(); plt.close()
    # k means clustering into two clusters
    # will seperate out the two checkerboard patterns
    # diff of labels will give each transition between checkerboard patterns (i.e. each reversal)
    print('kmeans clustering on revchecker worldcam')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 2
    num_frames = np.size(world_vid,0); vid_width = np.size(world_vid,1); vid_height = np.size(world_vid,2)
    kmeans_input = world_vid.reshape(num_frames,vid_width*vid_height)
    compactness, labels, centers = cv2.kmeans(kmeans_input.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    label_diff = np.diff(np.ndarray.flatten(labels))
    revind = list(abs(label_diff)) # need abs because reversing back will be -1
    # plot time between reversals, which should be centered around 1sec
    plt.figure()
    plt.title('time between reversals')
    plt.hist(np.diff(worldT[np.where(revind)]),bins=100); plt.xlim([0.9,1.1])
    plt.xlabel('time (s)')
    pdf.savefig(); plt.close()
    # get response of each channel centered around time of checkerboard reversal
    revchecker_window_start = 0.1 # in seconds
    revchecker_window_end = 0.5 # in seconds
    samprate = 30000 # Hz
    print('getting reversal response')
    all_resp = np.zeros([np.size(filt_ephys, 1), np.sum(revind), len(list(set(np.arange(1-revchecker_window_start, 1+revchecker_window_end, 1/samprate))))])
    true_rev_index = 0
    for rev_index, rev_label in tqdm(enumerate(revind)):
        if rev_label == True and worldT[rev_index] > 1:
            for ch_num in range(np.size(filt_ephys, 1)):
                # index of ephys data to start window with, aligned to set time before checkerboard will reverse
                bin_start = int((worldT[rev_index]-revchecker_window_start)*samprate)
                # index of ephys data to end window with, aligned to time after checkerboard reversed
                bin_end = int((worldT[rev_index]+revchecker_window_end)*samprate)
                # index into the filtered ephys data and store each trace for this channel of the probe
                if bin_end < np.size(filt_ephys, 0): # make sure it's a possible range
                    all_resp[ch_num, true_rev_index] = filt_ephys[bin_start:bin_end, ch_num]
            true_rev_index = true_rev_index + 1
    # mean of responses within each channel
    rev_resp_mean = np.mean(all_resp, 1)
    print('generating figures and csd')
    # plot traces over each other for two shanks
    colors = plt.cm.jet(np.linspace(0,1,32))
    num_channels = int([16 if '16' in probe_type else 64][0])
    if num_channels == 64:
        plt.subplots(1,2 ,figsize=(12,6))
        for ch_num in np.arange(0,64):
            if ch_num<=31:
                plt.subplot(1,2,1)
                plt.plot(rev_resp_mean[ch_num], color=colors[ch_num], linewidth=1)
                plt.title('ch1:32'); plt.axvline(x=(0.1*samprate))
                plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
            if ch_num>31:
                plt.subplot(1,2,2)
                plt.plot(rev_resp_mean[ch_num], color=colors[ch_num-32], linewidth=1)
                plt.title('ch33:64'); plt.axvline(x=(0.1*samprate))
                plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
        pdf.savefig(); plt.close()
    # channels arranged in columns
    fig, axes = plt.subplots(int(np.size(rev_resp_mean,0)/2),2, figsize=(7,20),sharey=True)
    ch_num = 0
    for ax in axes.T.flatten():
        ax.plot(rev_resp_mean[ch_num], linewidth=1)
        ax.axvline(x=(0.1*samprate), linewidth=1)
        ax.axis('off')
        ax.set_title(ch_num)
        ch_num = ch_num + 1
    pdf.savefig(); plt.close()
    # csd
    csd = np.ones([np.size(rev_resp_mean,0), np.size(rev_resp_mean,1)])
    csd_interval = 2
    for ch in range(2,np.size(rev_resp_mean,0)-2):
        csd[ch] = rev_resp_mean[ch] - 0.5*(rev_resp_mean[ch-csd_interval] + rev_resp_mean[ch+csd_interval])
    # csd between -1 and 1
    csd_interp = np.interp(csd, (csd.min(), csd.max()), (-1, +1))
    # visualize csd
    fig, ax = plt.subplots(1,1)
    plt.subplot(1,1,1)
    plt.imshow(csd_interp, cmap='jet')
    plt.axes().set_aspect('auto'); plt.colorbar()
    plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
    plt.xlabel('msec'); plt.ylabel('channel')
    plt.axvline(x=(0.1*samprate), color='k')
    plt.title('revchecker csd')
    pdf.savefig(); plt.close()
    pdf.close()
    print('revchecker analysis complete; pdf written to file')