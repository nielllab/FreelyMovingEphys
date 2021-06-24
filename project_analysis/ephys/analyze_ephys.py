"""
analyze_ephys.py
"""
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cv2, gc
import subprocess
import matplotlib as mpl 
import platform
if platform.system() == 'Linux':
    mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
else:
    mpl.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
from scipy.interpolate import interp1d
from numpy import nan
from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from util.aux_funcs import nanxcorr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import os
from scipy.ndimage import shift as imshift
from scipy import signal
from sklearn.cluster import KMeans

from project_analysis.ephys.ephys_figures import *
from project_analysis.ephys.ephys_utils import *

def find_files(rec_path, rec_name, free_move, cell, stim_type, mp4, probe_name):
    """
    assemble file paths and options into dictionary
    output dictionary is passed into func run_ephys_analysis
    INPUTS
        rec_path: path to the recording directory
        rec_name: name of the recording (e.g. 'date_subject_control_Rig2_hf1_wn')
        free_move: bool, True if this is a freely moving recording
        cell: unit index to highlight in figures/videos
        stim_type: None if freely moving, else any from ['white_noise', 'gratings', 'revchecker', 'sparse_noise']
        mp4: bool, True if videos of worldcam, eyecam, animated ephys raster + other plots should be written (this is somewhat slow to run)
        probe_name: name of probe, which should be a key in the .json in this repository, /matlab/channel_maps.json
    OUTPUTS
        file_dict: dictionay of the paths to important files and options to run ephys analysis with
    """
    print('finding ephys files')
    # get the files names in the provided path
    eye_file = os.path.join(rec_path, rec_name + '_REYE.nc')
    if not os.path.isfile(eye_file):
        eye_file = os.path.join(rec_path, rec_name + '_Reye.nc')
    world_file = os.path.join(rec_path, rec_name + '_world.nc')
    top_file = os.path.join(rec_path, rec_name + '_TOP1.nc')
    ephys_file = os.path.join(rec_path, rec_name + '_ephys_merge.json')
    imu_file = os.path.join(rec_path, rec_name + '_imu.nc')
    speed_file = os.path.join(rec_path, rec_name + '_speed.nc')
    ephys_bin_file = os.path.join(rec_path, rec_name + '_Ephys.bin')
    # shorten gratings stim, since 'grat' is the str used in ephys analysis
    # this can be eliminated if either name passed in or usage in run_ephys_analysis is changed
    if stim_type == 'gratings':
        stim_type = 'grat'
    # assemble dict
    if free_move is True:
        dict_out = {'cell':cell,'top':top_file,'eye':eye_file,'world':world_file,'ephys':ephys_file,
        'ephys_bin':ephys_bin_file,'speed':None,'imu':imu_file,'save':rec_path,'name':rec_name,
        'stim_type':stim_type,'mp4':mp4,'probe_name':probe_name}
    elif free_move is False:
        dict_out = {'cell':cell,'eye':eye_file,'world':world_file,'ephys':ephys_file,'ephys_bin':ephys_bin_file,
        'speed':speed_file,'imu':None,'save':rec_path,'name':rec_name,'stim_type':stim_type,
        'mp4':mp4,'probe_name':probe_name}

    return dict_out

def run_ephys_analysis(file_dict):
    """
    ephys analysis bringing together eyecam, worldcam, ephys data, imu data, and running ball optical mouse data
    runs on one recording at a time
    saves out an .h5 file for the rec structured as a dict of 
    h5 file is  best read in with pandas, or if pooling data across recordings, and then across sessions, with load_ephys func in /project_analysis/ephys/ephys_utils.py
    INPUTS
        file_dict: dictionary saved out from func find_files (see find_files docstring)
    """

    # set up props of this recording
    if file_dict['speed'] is None:
        free_move = True; has_imu = True; has_mouse = False
    else:
        free_move = False; has_imu = False; has_mouse = True

    # delete the existing h5 file, so that a new one can be written
    if os.path.isfile(os.path.join(file_dict['save'], (file_dict['name']+'_ephys_props.h5'))):
        os.remove(os.path.join(file_dict['save'], (file_dict['name']+'_ephys_props.h5')))

    print('opening pdfs')
    # three pdf outputs will be saved
    overview_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_overview_analysis_figures.pdf')))
    detail_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_detailed_analysis_figures.pdf')))
    diagnostic_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_diagnostic_analysis_figures.pdf')))

    print('opening and resizing worldcam data')
    world_data = xr.open_dataset(file_dict['world'])
    world_vid_raw = np.uint8(world_data['WORLD_video'])

    sz = world_vid_raw.shape
    if sz[1]>160:
        downsamp = 0.5
        world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
        for f in range(sz[0]):
            world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
    else:
        # if the worldcam has already been resized when the nc file was written in preprocessing, don't resize
        world_vid = world_vid_raw

    del world_vid_raw
    gc.collect()

    worldT = world_data.timestamps.copy()

    # plot worldcam timing
    worldcam_fig = plot_cam_time(worldT, 'world')
    diagnostic_pdf.savefig()
    plt.close()

    # plot mean world image
    plt.figure()
    plt.imshow(np.mean(world_vid,axis=0)); plt.title('mean world image')
    diagnostic_pdf.savefig()
    plt.close()

    if free_move is True:
        print('opening top data')
        # open the topdown camera nc file
        top_data = xr.open_dataset(file_dict['top'])
        # get the speed of the base of the animal's tail in the topdown tracking
        topx = top_data.TOP1_pts.sel(point_loc='tailbase_x').values; topy = top_data.TOP1_pts.sel(point_loc='tailbase_y').values
        topdX = np.diff(topx); topdY = np.diff(topy)
        top_speed = np.sqrt(topdX**2, topdY**2)
        topT = top_data.timestamps.copy()
        top_vid = np.uint8(top_data['TOP1_video'])

        del top_data
        gc.collect()
    
    # load IMU data
    if file_dict['imu'] is not None:
        print('opening imu data')
        imu_data = xr.open_dataset(file_dict['imu'])
        accT = imu_data.IMU_data.sample
        acc_chans = imu_data.IMU_data
        # raw gyro values
        gx = np.array(acc_chans.sel(channel='gyro_x_raw'))
        gy = np.array(acc_chans.sel(channel='gyro_y_raw'))
        gz = np.array(acc_chans.sel(channel='gyro_z_raw'))
        # gyro values in degrees
        gx_deg = np.array(acc_chans.sel(channel='gyro_x'))
        gy_deg = np.array(acc_chans.sel(channel='gyro_y'))
        gz_deg = np.array(acc_chans.sel(channel='gyro_z'))
        # pitch and roll in deg
        groll = np.array(acc_chans.sel(channel='roll'))
        gpitch = np.array(acc_chans.sel(channel='pitch'))
        # figure of gyro z
        plt.figure()
        plt.plot(gz[0:100*60])
        plt.title('gyro z')
        plt.xlabel('frame')
        diagnostic_pdf.savefig()
        plt.close()

    # load optical mouse nc file from running ball
    if file_dict['speed'] is not None:
        print('opening speed data')
        speed_data = xr.open_dataset(file_dict['speed'])
        spdVals = speed_data.BALL_data
        try:
            spd = spdVals.sel(move_params = 'speed_cmpersec')
            spd_tstamps = spdVals.sel(move_params = 'timestamps')
        except:
            spd = spdVals.sel(frame = 'speed_cmpersec')
            spd_tstamps = spdVals.sel(frame = 'timestamps')

    print('opening ephys data')
    ephys_data = pd.read_json(file_dict['ephys'])
    ephys_data['spikeTraw'] = ephys_data['spikeT']

    print('getting good cells')
    # select good cells from phy2
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    units = goodcells.index.values
    # get number of good units
    n_units = len(goodcells)
    # plot spike raster
    spikeraster_fig = plot_spike_rasters(goodcells)
    detail_pdf.savefig()
    plt.close()

    print('opening eyecam data')
    # load eye data
    eye_data = xr.open_dataset(file_dict['eye'])
    eye_vid = np.uint8(eye_data['REYE_video'])
    eyeT = eye_data.timestamps.copy()
    # plot eye timestamps
    reyecam_fig = plot_cam_time(eyeT, 'reye')
    diagnostic_pdf.savefig()
    plt.close()
    # plot eye postion across recording
    eye_params = eye_data['REYE_ellipse_params']
    eyepos_fig = plot_eye_pos(eye_params)
    detail_pdf.savefig()
    plt.close()
    # define theta, phi and zero-center
    th = np.array((eye_params.sel(ellipse_params = 'theta')-np.nanmean(eye_params.sel(ellipse_params = 'theta')))*180/3.14159)
    phi = np.array((eye_params.sel(ellipse_params = 'phi')-np.nanmean(eye_params.sel(ellipse_params = 'phi')))*180/3.14159)

    if file_dict['speed'] is not None:
        # plot optical mouse speeds
        optical_mouse_sp_fig = plot_optmouse_spd(spd_tstamps, spd)
        detail_pdf.savefig()
        plt.close()

    print('adjusting camera times to match ephys')
    # adjust eye/world/top times relative to ephys
    ephysT0 = ephys_data.iloc[0,12]
    eyeT = eye_data.timestamps  - ephysT0
    if eyeT[0]<-600:
        eyeT = eyeT + 8*60*60 # 8hr offset for some data
    worldT = world_data.timestamps - ephysT0
    if worldT[0]<-600:
        worldT = worldT + 8*60*60
    if free_move is True and has_imu is True:
        accTraw = imu_data.IMU_data.sample - ephysT0
    if free_move is False and has_mouse is True:
        speedT = spd_tstamps - ephysT0
    if free_move is True:
        topT = topT - ephysT0

    del eye_data
    gc.collect()

    # check that deinterlacing worked correctly
    # plot theta and theta_switch
    # want theta_switch to be jagged, theta to be smooth
    theta_switch_fig, th_switch = plot_param_switch_check(eye_params)
    diagnostic_pdf.savefig()
    plt.close()
    # plot eye variables
    eye_param_fig = plot_eye_params(eye_params, eyeT)
    detail_pdf.savefig()
    plt.close()

    # calculate eye veloctiy
    dEye = np.diff(th)

    # check accelerometer / eye temporal alignment
    if file_dict['imu'] is not None:
        print('checking accelerometer / eye temporal alignment')
        # eye velocity against head movements
        plt.figure
        plt.plot(eyeT[0:-1],-dEye,label = '-dEye')
        plt.plot(accTraw,gz*3-7.5,label = 'gz')
        plt.legend()
        plt.xlim(0,10); plt.xlabel('secs')
        diagnostic_pdf.savefig()
        plt.close()

        lag_range = np.arange(-0.2,0.2,0.002)
        cc = np.zeros(np.shape(lag_range))
        t1 = np.arange(5,len(dEye)/60-120,20).astype(int) # was np.arange(5,1600,20), changed for shorter videos
        t2 = t1 + 60
        offset = np.zeros(np.shape(t1))
        ccmax = np.zeros(np.shape(t1))
        acc_interp = interp1d(accTraw, (gz-3)*7.5)
        for tstart in tqdm(range(len(t1))):
            for l in range(len(lag_range)):
                try:
                    c, lag= nanxcorr(-dEye[t1[tstart]*60 : t2[tstart]*60] , acc_interp(eyeT[t1[tstart]*60:t2[tstart]*60]+lag_range[l]),1)
                    cc[l] = c[1]
                except: # occasional problem with operands that cannot be broadcast togther because of different shapes
                    cc[l] = np.nan
            offset[tstart] = lag_range[np.argmax(cc)]    
            ccmax[tstart] = np.max(cc)
        offset[ccmax<0.1] = np.nan
        acc_eyetime_alligment_fig = plot_acc_eyetime_alignment(eyeT, t1, offset, ccmax)
        diagnostic_pdf.savefig()
        plt.close()

        del ccmax
        gc.collect()

    if file_dict['imu'] is not None:
        print('fitting regression to timing drift')
        # fit regression to timing drift
        model = LinearRegression()
        dataT = np.array(eyeT[t1*60 + 30*60])
        model.fit(dataT[~np.isnan(offset)].reshape(-1,1),offset[~np.isnan(offset)]) 

        offset0 = model.intercept_
        drift_rate = model.coef_
        plot_regression_timing_fit_fig = plot_regression_timing_fit(dataT[~np.isnan(dataT)], offset[~np.isnan(dataT)], offset0, drift_rate)
        diagnostic_pdf.savefig()
        plt.close()
        print(offset0,drift_rate)

        del dataT
        gc.collect()
        
    elif file_dict['speed'] is not None:
        offset0 = 0.1
        drift_rate = -0.000114

    if file_dict['imu'] is not None:
        accT = accTraw - (offset0 + accTraw*drift_rate)
        del accTraw

    print('correcting ephys spike times for offset and timing drift')
    for i in ephys_data.index:
        ephys_data.at[i,'spikeT'] = np.array(ephys_data.at[i,'spikeTraw']) - (offset0 + np.array(ephys_data.at[i,'spikeTraw']) *drift_rate)
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    
    print('preparing worldcam video')
    if free_move:
        print('estimating eye-world calibration')
        xmap, ymap,fig = eye_shift_estimation(th, phi, eyeT, world_vid,worldT,60*60)
        xcorrection = xmap.copy()
        ycorrection = ymap.copy()

        # print('applying gamma to camera')
        # cam_gamma = 1
        # world_norm = (world_vid/255)#**cam_gamma
        
        print('shifting worldcam for eyes')
        thInterp =interp1d(eyeT,th, bounds_error = False)
        phiInterp =interp1d(eyeT,phi, bounds_error = False)
        thWorld = thInterp(worldT)
        phiWorld = phiInterp(worldT)
        for f in tqdm(range(np.shape(world_vid)[0])):
            world_vid[f,:,:] = imshift(world_vid[f,:,:],(-np.int8(thInterp(worldT[f])*ycorrection[0] + phiInterp(worldT[f])*ycorrection[1]),
                                                         -np.int8(thInterp(worldT[f])*xcorrection[0] + phiInterp(worldT[f])*xcorrection[1])))
        
        print('saving worldcam video corrected for eye movements')
        np.save(file=os.path.join(file_dict['save'], 'corrected_worldcam.npy'), arr=world_vid)

    std_im = np.std(world_vid,axis=0)

    img_norm = (world_vid-np.mean(world_vid,axis=0))/std_im
    std_im[std_im<20] = 0
    img_norm = img_norm * (std_im>0)

    # worldcam contrast
    contrast = np.empty(worldT.size)
    for i in range(worldT.size):
        contrast[i] = np.std(img_norm[i,:,:])
    plt.plot(contrast[2000:3000])
    plt.xlabel('time')
    plt.ylabel('contrast')
    diagnostic_pdf.savefig()
    plt.close()

    fig = plt.figure()
    plt.imshow(std_im)
    plt.colorbar(); plt.title('std img')
    diagnostic_pdf.savefig()
    plt.close()

    # make movie and sound
    this_unit = file_dict['cell']

    # set up interpolators for eye and world videos
    eyeInterp = interp1d(eyeT, eye_vid, axis=0, bounds_error=False)
    worldInterp = interp1d(worldT, world_vid, axis=0, bounds_error=False)
    if free_move:
        topInterp = interp1d(topT, top_vid, axis=0,bounds_error=False)
    
    if file_dict['imu'] is not None and free_move is True:
        trace_summary_fig = plot_trace_summary(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, top_speed, topT, tr = [15,45], accT=accT, gz=gz)
        detail_pdf.savefig()
        plt.close()
    elif file_dict['speed'] is not None and free_move is True:
        trace_summary_fig = plot_trace_summary(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, top_speed, topT, tr = [15,45], speedT=speedT, spd=spd)
        detail_pdf.savefig()
        plt.close()

    if file_dict['mp4']:
        if file_dict['imu'] is not None:
            print('making video figure')
            vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, accT=accT, gz=gz)
            print('making a reduced version of the video figure')
            vidfile1 = make_movie1(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, top_vid, topT, topInterp, accT=accT, gz=gz)
        elif file_dict['speed'] is not None:
            print('making video figure')
            vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, speedT=speedT, spd=spd)
            vidfile2 = make_movie2(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, speedT=speedT, spd=spd)
        print('making audio figure')
        audfile = make_sound(file_dict, ephys_data, units, this_unit)
        print('merging videos with sound')
        if file_dict['imu'] is not None:
            # from make_movie1 (no panels, just videos and raster)
            merge_mp4_name = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'_simple_panels_merge.mp4'))
            subprocess.call(['ffmpeg', '-i', vidfile1, '-i', audfile, '-c:v', 'copy', '-c:a', 'aac', '-y', merge_mp4_name])
        elif file_dict['speed'] is not None:
            merge_mp4_name = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'_simple_panels_merge.mp4'))
            subprocess.call(['ffmpeg', '-i', vidfile2, '-i', audfile, '-c:v', 'copy', '-c:a', 'aac', '-y', merge_mp4_name])
        # main video
        merge_mp4_name = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'_merge.mp4'))
        subprocess.call(['ffmpeg', '-i', vidfile, '-i', audfile, '-c:v', 'copy', '-c:a', 'aac', '-y', merge_mp4_name])

    if free_move is True and file_dict['imu'] is not None:
        plt.figure()
        plt.plot(eyeT[0:-1],np.diff(th),label = 'dTheta')
        plt.plot(accT-0.1,(gz-3)*10, label = 'gyro')
        plt.xlim(30,40); plt.ylim(-12,12); plt.legend(); plt.xlabel('secs')
        diagnostic_pdf.savefig()
        plt.close()

    print('plot eye and gaze (i.e. saccade and fixate)')
    if free_move and file_dict['imu'] is not None:
        gInterp = interp1d(accT,(gz-np.nanmean(gz))*7.5 , bounds_error = False)
        plt.figure(figsize = (8,4))
        plot_saccade_and_fixate_fig = plot_saccade_and_fixate(eyeT, dEye, gInterp, th)
        diagnostic_pdf.savefig()
        plt.close()
    
    plt.subplot(1,2,1)
    plt.imshow(std_im)
    plt.title('std dev of image')
    plt.subplot(1,2,2)
    plt.imshow(np.mean(world_vid,axis=0),vmin=0,vmax=255)
    plt.title('mean of image')
    diagnostic_pdf.savefig()
    plt.close()

    # set up timebase for subsequent analysis
    dt = 0.025
    t = np.arange(0, np.max(worldT),dt)

    # interpolate and plot contrast
    newc = interp1d(worldT,contrast)
    contrast_interp = newc(t[0:-1])
    plt.plot(t[0:600],contrast_interp[0:600])
    plt.xlabel('secs'); plt.ylabel('world contrast')
    diagnostic_pdf.savefig()
    plt.close()

    print('calculating firing rate')
    # calculate firing rate at new timebase
    ephys_data['rate'] = nan
    ephys_data['rate'] = ephys_data['rate'].astype(object)
    for i,ind in enumerate(ephys_data.index):
        ephys_data.at[ind,'rate'],bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
    ephys_data['rate']= ephys_data['rate']/dt
    goodcells = ephys_data.loc[ephys_data['group']=='good']

    print('calculating contrast reponse functions')
    # mean firing rate in timebins correponding to contrast ranges
    resp = np.empty((n_units,12))
    crange = np.arange(0,1.2,0.1)
    for i, ind in enumerate(goodcells.index):
        for c,cont in enumerate(crange):
            resp[i,c] = np.mean(goodcells.at[ind,'rate'][(contrast_interp>cont) & (contrast_interp<(cont+0.1))])
    # plot individual contrast response functions in subplots
    crf_cent, crf_tuning, crf_err, crf_fig = plot_spike_rate_vs_var(contrast, crange, goodcells, worldT, t, 'contrast')
    detail_pdf.savefig()
    plt.close()

    eyeR = eye_params.sel(ellipse_params = 'longaxis').copy()
    Rnorm = (eyeR - np.mean(eyeR))/np.std(eyeR)
    try: # test excluding this because of C++ assertion error raised during batch analysis
        plt.figure()
        plt.plot(eyeT,Rnorm)
        #plt.xlim([0,60])
        plt.xlabel('secs')
        plt.ylabel('normalized pupil R')
        diagnostic_pdf.savefig()
        plt.close()
    except:
        pass

    if file_dict['stim_type'] == 'revchecker':
        print('running revchecker analysis')
        print('loading ephys binary file and applying filters')
        # read in the binary file of ephys recording
        lfp_ephys = read_ephys_bin(file_dict['ephys_bin'], file_dict['probe_name'], do_remap=True)
        # subtract off average for each channel, then apply bandpass filter
        ephys_center_sub = lfp_ephys - np.mean(lfp_ephys,0)
        filt_ephys = butter_bandpass(ephys_center_sub, lowcut=1, highcut=300, fs=30000, order=6)
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
        diagnostic_pdf.savefig(); plt.close()
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
        del all_resp
        gc.collect()
        print('generating figures and csd')
        # plot traces over each other for two shanks
        colors = plt.cm.jet(np.linspace(0,1,32))
        num_channels = int([16 if '16' in file_dict['probe_name'] else 64][0])
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
            detail_pdf.savefig(); plt.close()
        # channels arranged in columns
        fig, axes = plt.subplots(int(np.size(rev_resp_mean,0)/2),2, figsize=(7,20),sharey=True)
        ch_num = 0
        for ax in axes.T.flatten():
            ax.plot(rev_resp_mean[ch_num], linewidth=1)
            ax.axvline(x=(0.1*samprate), linewidth=1)
            ax.axis('off')
            ax.set_title(ch_num)
            ch_num = ch_num + 1
        detail_pdf.savefig(); plt.close()
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
        detail_pdf.savefig(); plt.close()
        print('getting lfp relative depth')
        # assign the deepest deflection to lfp, the center of layer 4, to have depth 0
        # channels above will have negative depth, channels below will have positive depth
        # adding or subtracting "depth" with a step size of 1
        if num_channels == 64:
            shank0_layer4cent = np.argmin(np.min(rev_resp_mean[0:32,int(samprate*0.1):int(samprate*0.3)], axis=1))
            shank1_layer4cent = np.argmin(np.min(rev_resp_mean[32:64,int(samprate*0.1):int(samprate*0.3)], axis=1))
            shank0_ch_positions = list(range(32)) - shank0_layer4cent; shank1_ch_positions = list(range(32)) - shank1_layer4cent
            lfp_depth = [shank0_ch_positions, shank1_ch_positions]
            layer4_out = [shank0_layer4cent, shank1_layer4cent]
        elif num_channels == 16:
            layer4cent = np.argmin(np.min(rev_resp_mean, axis=1))
            lfp_depth = [list(range(16)) - layer4cent]
            layer4_out = [layer4cent]
    
    if file_dict['stim_type'] == 'grat':
        print('getting grating flow')
        nf = np.size(img_norm,0)-1
        u_mn = np.zeros((nf,1)); v_mn = np.zeros((nf,1))
        sx_mn = np.zeros((nf,1)) ; sy_mn = np.zeros((nf,1))
        flow_norm = np.zeros((nf,np.size(img_norm,1),np.size(img_norm,2),2 ))
        vidfile = os.path.join(file_dict['save'], (file_dict['name']+'_grating_flow'))
        # find screen
        meanx = np.mean(std_im>0,axis=0)
        xcent = np.int(np.sum(meanx*np.arange(len(meanx)))/ np.sum(meanx))
        meany = np.mean(std_im>0,axis=1)
        ycent = np.int(np.sum(meany*np.arange(len(meany)))/ np.sum(meany))
        xrg = 40;   yrg = 25; # pixel range to define monitor
        # animation of optic flow
        fig, ax = plt.subplots(1,1,figsize = (16,8))
        # now animate
        for f in tqdm(range(nf)):
            frm = np.uint8(32*(img_norm[f,:,:]+4))
            frm2 = np.uint8(32*(img_norm[f+1,:,:]+4))
            # frm = cv2.resize(frm, (0,0), fx=0.5); frm2 = cv2.resize(frm2, (0,0), fx=0.5) # added resizing frames to a downscaled resolution
            flow_norm[f,:,:,:] = cv2.calcOpticalFlowFarneback(frm,frm2, None, 0.5, 3, 30, 3, 7, 1.5, 0)
            #ax.cla()
            #ax.imshow(frm,vmin = 0, vmax = 255)
            u = flow_norm[f,:,:,0]; v = -flow_norm[f,:,:,1]  # negative to fix sign for y axis in images
            sx = cv2.Sobel(frm,cv2.CV_64F,1,0,ksize=11)
            sy = -cv2.Sobel(frm,cv2.CV_64F,0,1,ksize=11)# negative to fix sign for y axis in images
            sx[std_im<20]=0; sy[std_im<20]=0; # get rid of values outside of monitor
            sy[sx<0] = -sy[sx<0]  #make vectors point in positive x direction (so opposite sides of grating don't cancel)
            sx[sx<0] = -sx[sx<0]
            #sy[np.abs(sx)<500000] = np.abs(sy[np.abs(sx)<500000]) # deals with horitzontal cases - flips them right side up
            sy[np.abs(sx/sy)<0.15] = np.abs(sy[np.abs(sx/sy)<0.15])
            #ax.quiver(x[::nx,::nx],y[::nx,::nx],sx[::nx,::nx],sy[::nx,::nx], scale = 100000 )
            #u_mn[f]= np.mean(u); v_mn[f]= np.mean(v); sx_mn[f] = np.mean(sx); sy_mn[f] = np.mean(sy)
            u_mn[f]= np.mean(u[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg]); v_mn[f]= np.mean(v[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg]); 
            sx_mn[f] = np.mean(sx[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg]); sy_mn[f] = np.mean(sy[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg])
    
        scr_contrast = np.empty(worldT.size)
        for i in range(worldT.size):
            scr_contrast[i] = np.nanmean(np.abs(img_norm[i,ycent-25:ycent+25,xcent-40:xcent+40]))
        scr_contrast = signal.medfilt(scr_contrast,11)
        
        stimOn = np.double(scr_contrast>0.5)

        stim_start = np.array(worldT[np.where(np.diff(stimOn)>0)])
        grating_psth = plot_psth(goodcells,stim_start,-0.5,1.5,0.1,True)
        plt.title('grating psth')
        detail_pdf.savefig(); plt.close()
        
        stim_end = np.array(worldT[np.where(np.diff(stimOn)<0)])
        stim_end = stim_end[stim_end>stim_start[0]]
        stim_start = stim_start[stim_start<stim_end[-1]]
        grating_th = np.zeros(len(stim_start))
        grating_mag = np.zeros(len(stim_start))
        grating_dir = np.zeros(len(stim_start))
        for i in range(len(stim_start)):
            tpts = np.where((worldT>stim_start[i] + 0.025) & (worldT<stim_end[i]-0.025))
            mag = np.sqrt(sx_mn[tpts]**2 + sy_mn[tpts]**2)
            this = np.where(mag[:,0]>np.percentile(mag,25))
            goodpts = np.array(tpts)[0,this]

            stim_sx = np.nanmedian(sx_mn[tpts])
            stim_sy = np.nanmedian(sy_mn[tpts])
            stim_u = np.nanmedian(u_mn[tpts])
            stim_v = np.nanmedian(v_mn[tpts])
            grating_th[i] = np.arctan2(stim_sy,stim_sx)
            grating_mag[i] = np.sqrt(stim_sx**2 + stim_sy**2)
            grating_dir[i] = np.sign(stim_u*stim_sx + stim_v*stim_sy) # dot product of gratient and flow gives direction
        # grating_th = np.round(grating_th *10)/10

        grating_ori = grating_th.copy()
        grating_ori[grating_dir<0] = grating_ori[grating_dir<0] + np.pi
        grating_ori = grating_ori - np.min(grating_ori)
        np.unique(grating_ori)

        ori_cat = np.floor((grating_ori+np.pi/16)/(np.pi/4))

        km = KMeans(n_clusters=3).fit(np.reshape(grating_mag,(-1,1)))
        sf_cat = km.labels_
        order = np.argsort(np.reshape(km.cluster_centers_, 3))
        sf_catnew = sf_cat.copy()
        for i in range(3):
            sf_catnew[sf_cat == order[i]]=i
        sf_cat = sf_catnew.copy()
        
        plt.figure(figsize = (8,8))
        plt.scatter(grating_mag,grating_ori,c=ori_cat)
        plt.xlabel('grating magnitude'); plt.ylabel('theta')
        diagnostic_pdf.savefig()
        plt.close()

        ntrial = np.zeros((3,8))
        for i in range(3):
            for j in range(8):
                ntrial[i,j]= np.sum((sf_cat==i)&(ori_cat==j))
        plt.figure; plt.imshow(ntrial,vmin = 0, vmax = 2*np.mean(ntrial)); plt.colorbar()
        plt.xlabel('orientations'); plt.ylabel('sfs'); plt.title('trials per condition')
        diagnostic_pdf.savefig()
        plt.close()
        
        print('plotting grating orientation and tuning curves')
        edge_win = 0.025
        grating_rate = np.zeros((len(goodcells),len(stim_start)))
        spont_rate = np.zeros((len(goodcells),len(stim_start)))
        ori_tuning = np.zeros((len(goodcells),8,3))
        drift_spont = np.zeros(len(goodcells))
        plt.figure(figsize = (12,n_units*2))

        for c, ind in enumerate(goodcells.index):
            sp = goodcells.at[ind,'spikeT'].copy()
            for i in range(len(stim_start)):
                grating_rate[c,i] = np.sum((sp> stim_start[i]+edge_win) & (sp<stim_end[i])) / (stim_end[i] - stim_start[i]- edge_win)
            for i in range(len(stim_start)-1):
                spont_rate[c,i] = np.sum((sp> stim_end[i]+edge_win) & (sp<stim_start[i+1])) / (stim_start[i+1] - stim_end[i]- edge_win)  
            for ori in range(8):
                for sf in range(3):
                    ori_tuning[c,ori,sf] = np.mean(grating_rate[c,(ori_cat==ori) & (sf_cat ==sf)])
            drift_spont[c] = np.mean(spont_rate[c,:])
            plt.subplot(n_units,2,2*c+1)
            plt.scatter(grating_ori,grating_rate[c,:],c= sf_cat)
            plt.plot(3*np.ones(len(spont_rate[c,:])),spont_rate[c,:],'r.')
            plt.subplot(n_units,2,2*c+2)
            plt.plot(ori_tuning[c,:,0],label = 'low sf'); plt.plot(ori_tuning[c,:,1],label = 'mid sf');plt.plot(ori_tuning[c,:,2],label = 'hi sf')
            plt.plot([0,7],[drift_spont[c],drift_spont[c]],'r:', label = 'spont')
            
            try:
                plt.ylim(0,np.nanmax(ori_tuning[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
        plt.legend()
        detail_pdf.savefig()
        plt.close()
        
        del eyeInterp, worldInterp
        gc.collect()

    # create interpolator for movie data so we can evaluate at same timebins are firing rate
    #img_norm[img_norm<-2] = -2
    sz = np.shape(img_norm); downsamp = 0.5
    img_norm_sm = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)))
 
    for f in range(sz[0]):
        img_norm_sm[f,:,:] = cv2.resize(img_norm[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))

    movInterp = interp1d(worldT, img_norm_sm, axis=0, bounds_error=False)
    
    ch_count = int([16 if '16' in file_dict['probe_name'] else 64][0])
    print('getting spike-triggered average for lag=0.025')
    # calculate spike-triggered average
    staAll, STA_single_lag_fig = plot_STA_single_lag(n_units, img_norm_sm, goodcells, worldT, movInterp, ch_count)
    detail_pdf.savefig()
    plt.close()
    
    print('getting spike-triggered average with range in lags')
    # calculate spike-triggered average
    fig = plot_STA_multi_lag(n_units, goodcells, worldT, movInterp)
    detail_pdf.savefig()
    plt.close()

    print('getting spike-triggered variance')
    # calculate spike-triggered variance
    st_var, fig = plot_spike_triggered_variance(n_units, goodcells, t, movInterp, img_norm_sm)
    detail_pdf.savefig()
    plt.close()

    print('doing GLM RF estimate')
    if (free_move is True) | (file_dict['stim_type'] == 'white_noise'):
        ### simplified setup for GLM
        ### these are general parameters (spike rates, eye position)
        n_units = len(goodcells)
        print('get timing')
        model_dt = 0.025;
        model_t = np.arange(0,np.max(worldT),model_dt)
        model_nsp = np.zeros((n_units,len(model_t)))
        
        # get spikes / rate
        print('get spikes')
        bins = np.append(model_t,model_t[-1]+model_dt)
        for i,ind in enumerate(goodcells.index):
            model_nsp[i,:],bins = np.histogram(goodcells.at[ind,'spikeT'],bins)
        
        #get eye position
        print('get eye')
        thInterp =interp1d(eyeT,th, bounds_error = False)
        phiInterp =interp1d(eyeT,phi, bounds_error = False)
        model_th = thInterp(model_t+model_dt/2)
        model_phi = phiInterp(model_t+model_dt/2)
        del thInterp, phiInterp
        
        # get active times
        if free_move:
            interp = interp1d(accT,(gz-np.mean(gz))*7.5,bounds_error=False)
            model_gz = interp(model_t)
            model_active = np.convolve(np.abs(model_gz),np.ones(np.int(1/model_dt)),'same')
            use = np.where((np.abs(model_th)<10) & (np.abs(model_phi)<10)& (model_active>40) )[0]
        else:
            use = np.where((np.abs(model_th)<10) & (np.abs(model_phi)<10))[0]
            
        ### get video ready for GLM
        downsamp = 0.25;
        print('setting up video') 
        movInterp = None; model_vid_sm = 0
        movInterp = interp1d(worldT,img_norm,'nearest',axis = 0,bounds_error = False) 
        
        testimg = movInterp(model_t[0])
        testimg = cv2.resize(testimg,(int(np.shape(testimg)[1]*downsamp), int(np.shape(testimg)[0]*downsamp)))
        testimg = testimg[5:-5,5:-5]; #remove area affected by eye movement correction
        model_vid_sm = np.zeros((len(model_t),np.int(np.shape(testimg)[0]*np.shape(testimg)[1])))
        for i in tqdm(range(len(model_t))):
            model_vid = movInterp(model_t[i] + model_dt/2)
            smallvid = cv2.resize(model_vid,(np.int(np.shape(img_norm)[2]*downsamp),np.int(np.shape(img_norm)[1]*downsamp)),interpolation = cv2.INTER_AREA)
            smallvid = smallvid[5:-5,5:-5];
            #smallvid = smallvid - np.mean(smallvid)
            model_vid_sm[i,:] = np.reshape(smallvid,np.shape(smallvid)[0]*np.shape(smallvid)[1])
        
        nks = np.shape(smallvid); nk = nks[0]*nks[1]
        model_vid_sm[np.isnan(model_vid_sm)]=0
        
        del movInterp
        gc.collect()
        
        glm_receptive_field, glm_cc, fig = fit_glm_vid(model_vid_sm,model_nsp,model_dt, use,nks)
        detail_pdf.savefig()
        plt.close()

        del model_vid_sm
        gc.collect()

    spikeraster_fig = plot_spike_rasters(goodcells)
    detail_pdf.savefig()
    plt.close()

    print('plotting head and eye movements')
    # calculate saccade-locked psth
    spike_corr = 1 #+ 0.125/1200  # correction factor for ephys timing drift

    plt.figure()
    plt.hist(dEye, bins=21, range=(-10,10), density=True)
    plt.xlabel('eye dtheta'); plt.ylabel('fraction')
    detail_pdf.savefig()
    plt.close()
    
    if free_move is True:
        dhead = interp1d(accT,(gz-np.mean(gz))*7.5, bounds_error=False)
        dgz = dEye + dhead(eyeT[0:-1])
        
        plt.figure()
        plt.hist(dhead(eyeT),bins=21,range = (-10,10))
        plt.xlabel('dhead')
        detail_pdf.savefig()
        plt.close()
        
        plt.figure()
        plt.hist(dgz,bins=21,range = (-10,10))
        plt.xlabel('dgaze')
        detail_pdf.savefig()
        plt.close()
        
        # ValueEror length mistamtch fix
        # this should be done in a better way
        plt.figure()
        if len(dEye[0:-1:10]) == len(dhead(eyeT[0:-1:10])):
            plt.plot(dEye[0:-1:10],dhead(eyeT[0:-1:10]),'.')
        elif len(dEye[0:-1:10]) > len(dhead(eyeT[0:-1:10])):
            len_diff = len(dEye[0:-1:10]) - len(dhead(eyeT[0:-1:10]))
            plt.plot(dEye[0:-1:10][:-len_diff],dhead(eyeT[0:-1:10]),'.')
        elif len(dEye[0:-1:10]) < len(dhead(eyeT[0:-1:10])):
            len_diff = len(dhead(eyeT[0:-1:10])) - len(dEye[0:-1:10])
            plt.plot(dEye[0:-1:10],dhead(eyeT[0:-1:10])[:-len_diff],'.')
        plt.xlabel('dEye'); plt.ylabel('dHead'); plt.xlim((-10,10)); plt.ylim((-10,10))
        plt.plot([-10,10],[10,-10], 'r')
        detail_pdf.savefig()
        plt.close()
      
    print('plotting saccade-locked psths')
    trange = np.arange(-1,1.1,0.025)
    if free_move is True:
        sthresh = 5
        upsacc = eyeT[ (np.append(dEye,0)>sthresh)]
        downsacc = eyeT[ (np.append(dEye,0)<-sthresh)]
    else:
        sthresh = 3
        upsacc = eyeT[np.append(dEye,0)>sthresh]
        downsacc = eyeT[np.append(dEye,0)<-sthresh]   

    upsacc_avg, downsacc_avg, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
    plt.title('all dEye')
    detail_pdf.savefig()
    plt.close()

    if free_move is True:
        # plot gaze shifting eye movements
        sthresh = 3
        upsacc = eyeT[ (np.append(dEye,0)>sthresh) & (np.append(dgz,0)>sthresh)]
        downsacc = eyeT[ (np.append(dEye,0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
        upsacc_avg_gaze_shift_dEye, downsacc_avg_gaze_shift_dEye, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('gaze shift dEye');  detail_pdf.savefig() ;  plt.close()
        
        # plot compensatory eye movements    
        sthresh = 3
        upsacc = eyeT[ (np.append(dEye,0)>sthresh) & (np.append(dgz,0)<1)]
        downsacc = eyeT[ (np.append(dEye,0)<-sthresh) & (np.append(dgz,0)>-1)]
        upsacc_avg_comp_dEye, downsacc_avg_comp_dEye, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('comp dEye'); detail_pdf.savefig() ;  plt.close()
        
    
        # plot gaze shifting head movements
        sthresh = 3
        upsacc = eyeT[ (np.append(dhead(eyeT[0:-1]),0)>sthresh) & (np.append(dgz,0)>sthresh)]
        downsacc = eyeT[ (np.append(dhead(eyeT[0:-1]),0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
        upsacc_avg_gaze_shift_dHead, downsacc_avg_gaze_shift_dHead, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('gaze shift dhead') ; detail_pdf.savefig() ;  plt.close()
        
        # plot compensatory eye movements    
        sthresh = 3
        upsacc = eyeT[ (np.append(dhead(eyeT[0:-1]),0)>sthresh) & (np.append(dgz,0)<1)]
        downsacc = eyeT[ (np.append(dhead(eyeT[0:-1]),0)<-sthresh) & (np.append(dgz,0)>-1)]
        upsacc_avg_comp_dHead, downsacc_avg_comp_dHead, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('comp dhead') ; detail_pdf.savefig() ;  plt.close()

    # normalize and plot eye radius
    eyeR = eye_params.sel(ellipse_params = 'longaxis').copy()
    Rnorm = (eyeR - np.mean(eyeR))/np.std(eyeR)
    plt.figure()
    plt.plot(eyeT,Rnorm)
    #plt.xlim([0,60])
    plt.xlabel('secs')
    plt.ylabel('normalized pupil R')
    diagnostic_pdf.savefig()
    plt.close()

    print('plotting spike rate vs pupil radius and position')
    # plot rate vs pupil
    R_range = np.arange(10,51,5)
    spike_rate_vs_pupil_radius_cent, spike_rate_vs_pupil_radius_tuning, spike_rate_vs_pupil_radius_err, spike_rate_vs_pupil_radius_fig = plot_spike_rate_vs_var(eyeR, R_range, goodcells, eyeT, t, 'pupil radius')
    detail_pdf.savefig()
    plt.close()

    # normalize eye position
    eyeTheta = eye_params.sel(ellipse_params = 'theta').copy()
    thetaNorm = (eyeTheta - np.mean(eyeTheta))/np.std(eyeTheta)
    plt.plot(eyeT[0:3600],thetaNorm[0:3600])
    plt.xlabel('secs'); plt.ylabel('normalized eye theta')
    diagnostic_pdf.savefig()
    plt.close()

    eyePhi = eye_params.sel(ellipse_params = 'phi').copy()
    phiNorm = (eyePhi - np.mean(eyePhi))/np.std(eyePhi)

    print('plotting spike rate vs theta/phi')
    # plot rate vs theta
    th_range = np.arange(-30,31,5)
    spike_rate_vs_theta_cent, spike_rate_vs_theta_tuning, spike_rate_vs_theta_err, spike_rate_vs_theta_fig = plot_spike_rate_vs_var(th, th_range, goodcells, eyeT, t, 'eye theta')
    detail_pdf.savefig()
    plt.close()

    phi_range = np.arange(-30,31,5)
    spike_rate_vs_phi_cent, spike_rate_vs_phi_tuning, spike_rate_vs_phi_err, spike_rate_vs_phi_fig = plot_spike_rate_vs_var(phi, phi_range, goodcells, eyeT, t, 'eye phi')
    detail_pdf.savefig()
    plt.close()
    
    if free_move is True:
        print('plotting spike rate vs gyro and speed')
        active_interp = interp1d(model_t, model_active, bounds_error=False)
        active_accT = active_interp(accT.values)
        use = np.where(active_accT > 40)

        gx_range = np.arange(-5,6,1)
        active_gx = ((gx-np.mean(gx))*7.5)[use]
        spike_rate_vs_gx_cent, spike_rate_vs_gx_tuning, spike_rate_vs_gx_err, spike_rate_vs_gx_fig = plot_spike_rate_vs_var(active_gx, gx_range, goodcells, accT[use], t, 'gyro x')
        detail_pdf.savefig()
        plt.close()
        
        gy_range = np.arange(-5,6,1)
        active_gy = ((gy-np.mean(gy))*7.5)[use]
        spike_rate_vs_gy_cent, spike_rate_vs_gy_tuning, spike_rate_vs_gy_err, spike_rate_vs_gy_fig = plot_spike_rate_vs_var(active_gy, gy_range, goodcells, accT[use], t, 'gyro y')
        detail_pdf.savefig()
        plt.close()

        gz_range = np.arange(-7,8,1)
        active_gz = ((gz-np.mean(gz))*7.5)[use]
        spike_rate_vs_gz_cent, spike_rate_vs_gz_tuning, spike_rate_vs_gz_err, spike_rate_vs_gz_fig = plot_spike_rate_vs_var(active_gz, gz_range, goodcells, accT[use], t, 'gyro z')
        detail_pdf.savefig()
        plt.close()

    if free_move is False and has_mouse is True:
        print('plotting spike rate vs speed')
        spd_range = [0, 0.01, 0.1, 0.2, 0.5, 1.0]
        spike_rate_vs_spd_cent, spike_rate_vs_spd_tuning, spike_rate_vs_spd_err, spike_rate_vs_spd_fig = plot_spike_rate_vs_var(spd, spd_range, goodcells, speedT, t, 'speed')
        detail_pdf.savefig()
        plt.close()

    if free_move is True:
        print('plotting spike rate vs pitch/roll')
        # roll vs spike rate
        roll_range = np.arange(-30,31,10)
        spike_rate_vs_roll_cent, spike_rate_vs_roll_tuning, spike_rate_vs_roll_err, spike_rate_vs_roll_fig = plot_spike_rate_vs_var(groll[use], roll_range, goodcells, accT[use], t, 'roll')
        detail_pdf.savefig()
        plt.close()
        # pitch vs spike rate
        pitch_range = np.arange(-30,31,10)
        spike_rate_vs_pitch_cent, spike_rate_vs_pitch_tuning, spike_rate_vs_pitch_err, spike_rate_vs_pitch_fig = plot_spike_rate_vs_var(gpitch[use], pitch_range, goodcells, accT[use], t, 'pitch')
        detail_pdf.savefig()
        plt.close()
        print('plotting pitch/roll vs th/phi')
        # subtract mean from roll and pitch to center around zero
        pitch = gpitch - np.mean(gpitch)
        roll = groll - np.mean(groll)
        # pitch vs theta
        pitchi1d = interp1d(accT, pitch, bounds_error=False)
        pitch_interp = pitchi1d(eyeT)
        plt.figure()
        plt.plot(pitch_interp[::100], th[::100], '.'); plt.xlabel('pitch'); plt.ylabel('theta')
        plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
        diagnostic_pdf.savefig()
        plt.close()
        # roll vs phi
        rolli1d = interp1d(accT, roll, bounds_error=False)
        roll_interp = rolli1d(eyeT)
        plt.figure()
        plt.plot(roll_interp[::100], phi[::100], '.'); plt.xlabel('roll'); plt.ylabel('phi')
        plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')
        diagnostic_pdf.savefig()
        plt.close()
        # roll vs theta
        plt.figure()
        plt.plot(roll_interp[::100], th[::100], '.'); plt.xlabel('roll'); plt.ylabel('theta')
        plt.ylim([-60,60]); plt.xlim([-60,60])
        diagnostic_pdf.savefig()
        plt.close()
        # pitch vs phi
        plt.figure()
        plt.plot(pitch_interp[::100], phi[::100], '.'); plt.xlabel('pitch'); plt.ylabel('phi')
        plt.ylim([-60,60]); plt.xlim([-60,60])
        diagnostic_pdf.savefig()
        plt.close()
        # histogram of pitch values
        plt.figure()
        plt.hist(pitch, bins=50); plt.xlabel('pitch')
        diagnostic_pdf.savefig()
        plt.close()
        # histogram of pitch values
        plt.figure()
        plt.hist(roll, bins=50); plt.xlabel('roll')
        diagnostic_pdf.savefig()
        plt.close()
        # histogram of th values
        plt.figure()
        plt.hist(th, bins=50); plt.xlabel('theta')
        diagnostic_pdf.savefig()
        plt.close()
        # histogram of pitch values
        plt.figure()
        plt.hist(phi, bins=50); plt.xlabel('phi')
        diagnostic_pdf.savefig()
        plt.close()

    print('generating summary plots')
    # generate summary plot
    if file_dict['stim_type'] == 'grat':
        summary_fig = plot_summary(n_units, goodcells, crange, resp, file_dict, staAll, trange, upsacc_avg, downsacc_avg, ori_tuning=ori_tuning, drift_spont=drift_spont)
    else:
        summary_fig = plot_summary(n_units, goodcells, crange, resp, file_dict, staAll, trange, upsacc_avg, downsacc_avg)
    overview_pdf.savefig()
    plt.close()

    hist_dt = 1
    hist_t = np.arange(0, np.max(worldT),hist_dt)
    plt.figure(figsize = (12,n_units*2))
    plt.subplot(n_units+3,1,1)
    if has_imu:
        plt.plot(accT,gz)
        plt.xlim(0, np.max(worldT)); plt.ylabel('gz'); plt.title('gyro')
    elif has_mouse:
        plt.plot(speedT,spd)
        plt.xlim(0, np.max(worldT)); plt.ylabel('cm/sec'); plt.title('mouse speed')  

    plt.subplot(n_units+3,1,2)
    plt.plot(eyeT,eye_params.sel(ellipse_params = 'longaxis'))
    plt.xlim(0, np.max(worldT)); plt.ylabel('rad (pix)'); plt.title('pupil diameter')

    plt.subplot(n_units+3,1,3)
    plt.plot(worldT,contrast)
    plt.xlim(0, np.max(worldT)); plt.ylabel('contrast a.u.'); plt.title('contrast')

    for i,ind in enumerate(goodcells.index):
        rate,bins = np.histogram(ephys_data.at[ind,'spikeT'],hist_t)
        plt.subplot(n_units+3,1,i+4)
        plt.plot(bins[0:-1],rate)
        plt.xlabel('secs')
        plt.ylabel('sp/sec'); plt.xlim(bins[0],bins[-1]); plt.title('unit ' + str(ind))
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    # clear up space in memory
    del ephys_data
    gc.collect()

    print('closing pdfs')
    overview_pdf.close(); detail_pdf.close(); diagnostic_pdf.close()

    print('organizing data and saving .h5')
    split_base_name = file_dict['name'].split('_')

    date = split_base_name[0]; mouse = split_base_name[1]; exp = split_base_name[2]; rig = split_base_name[3]
    try:
        stim = '_'.join(split_base_name[4:])
    except:
        stim = split_base_name[4:]
    session_name = date+'_'+mouse+'_'+exp+'_'+rig
    unit_data = pd.DataFrame([])

    if file_dict['stim_type'] == 'grat':
        for unit_num, ind in enumerate(goodcells.index):
            unit_df = pd.DataFrame([]).astype(object)
            cols = [stim+'_'+i for i in ['c_range',
                                        'crf_cent',
                                        'crf_tuning',
                                        'crf_err',
                                        'spike_triggered_average',
                                        'sta_shape',
                                        'spike_triggered_variance',
                                        'upsacc_avg',
                                        'downsacc_avg',
                                        'spike_rate_vs_pupil_radius_cent',
                                        'spike_rate_vs_pupil_radius_tuning',
                                        'spike_rate_vs_pupil_radius_err',
                                        'spike_rate_vs_theta_cent',
                                        'spike_rate_vs_theta_tuning',
                                        'spike_rate_vs_theta_err',
                                        'grating_psth',
                                        'grating_ori',
                                        'ori_tuning',
                                        'drift_spont',
                                        'spont_rate',
                                        'grating_rate',
                                        'trange',
                                        'theta',
                                        'phi',
                                        'spike_rate_vs_spd_cent',
                                        'spike_rate_vs_spd_tuning',
                                        'spike_rate_vs_spd_err',
                                        'spike_rate_vs_phi_cent',
                                        'spike_rate_vs_phi_tuning',
                                        'spike_rate_vs_phi_err']]
            unit_df = pd.DataFrame(pd.Series([crange,
                                    crf_cent,
                                    crf_tuning[unit_num],
                                    crf_err[unit_num],
                                    np.ndarray.flatten(staAll[unit_num]),
                                    np.shape(staAll[unit_num]),
                                    np.ndarray.flatten(st_var[unit_num]),
                                    upsacc_avg[unit_num],
                                    downsacc_avg[unit_num],
                                    spike_rate_vs_pupil_radius_cent,
                                    spike_rate_vs_pupil_radius_tuning[unit_num],
                                    spike_rate_vs_pupil_radius_err[unit_num],
                                    spike_rate_vs_theta_cent,
                                    spike_rate_vs_theta_tuning[unit_num],
                                    spike_rate_vs_theta_err[unit_num],
                                    grating_psth[unit_num],
                                    grating_ori[unit_num],
                                    ori_tuning[unit_num],
                                    drift_spont[unit_num],
                                    spont_rate[unit_num],
                                    grating_rate[unit_num],
                                    trange,
                                    th,
                                    phi,
                                    spike_rate_vs_spd_cent,
                                    spike_rate_vs_spd_tuning[unit_num],
                                    spike_rate_vs_spd_err[unit_num],
                                    spike_rate_vs_phi_cent,
                                    spike_rate_vs_phi_tuning[unit_num],
                                    spike_rate_vs_phi_err[unit_num]]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
    if file_dict['stim_type'] == 'revchecker':
        for unit_num, ind in enumerate(goodcells.index):
            cols = [stim+'_'+i for i in ['c_range',
                                        'crf_cent',
                                        'crf_tuning',
                                        'crf_err',
                                        'spike_triggered_average',
                                        'sta_shape',
                                        'spike_triggered_variance',
                                        'upsacc_avg',
                                        'downsacc_avg',
                                        'spike_rate_vs_pupil_radius_cent',
                                        'spike_rate_vs_pupil_radius_tuning',
                                        'spike_rate_vs_pupil_radius_err',
                                        'spike_rate_vs_theta_cent',
                                        'spike_rate_vs_theta_tuning',
                                        'spike_rate_vs_theta_err',
                                        'trange',
                                        'revchecker_mean_resp_per_ch',
                                        'csd',
                                        'lfp_rel_depth',
                                        'theta',
                                        'phi',
                                        'spike_rate_vs_spd_cent',
                                        'spike_rate_vs_spd_tuning',
                                        'spike_rate_vs_spd_err',
                                        'spike_rate_vs_phi_cent',
                                        'spike_rate_vs_phi_tuning',
                                        'spike_rate_vs_phi_err',
                                        'layer4center']]
            unit_df = pd.DataFrame(pd.Series([crange,
                                    crf_cent,
                                    crf_tuning[unit_num],
                                    crf_err[unit_num],
                                    np.ndarray.flatten(staAll[unit_num]),
                                    np.shape(staAll[unit_num]),
                                    np.ndarray.flatten(st_var[unit_num]),
                                    upsacc_avg[unit_num],
                                    downsacc_avg[unit_num],
                                    spike_rate_vs_pupil_radius_cent,
                                    spike_rate_vs_pupil_radius_tuning[unit_num],
                                    spike_rate_vs_pupil_radius_err[unit_num],
                                    spike_rate_vs_theta_cent,
                                    spike_rate_vs_theta_tuning[unit_num],
                                    spike_rate_vs_theta_err[unit_num],
                                    trange,
                                    rev_resp_mean,
                                    csd_interp,
                                    lfp_depth,
                                    th,
                                    phi,
                                    spike_rate_vs_spd_cent,
                                    spike_rate_vs_spd_tuning[unit_num],
                                    spike_rate_vs_spd_err[unit_num],
                                    spike_rate_vs_phi_cent,
                                    spike_rate_vs_phi_tuning[unit_num],
                                    spike_rate_vs_phi_err[unit_num],
                                    layer4_out]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
    elif file_dict['stim_type'] != 'grat' and file_dict['stim_type'] != 'revchecker' and free_move is False and file_dict['stim_type'] != 'white_noise':
        for unit_num, ind in enumerate(goodcells.index):
            cols = [stim+'_'+i for i in ['c_range',
                                        'crf_cent',
                                        'crf_tuning',
                                        'crf_err',
                                        'spike_triggered_average',
                                        'sta_shape',
                                        'spike_triggered_variance',
                                        'upsacc_avg',
                                        'downsacc_avg',
                                        'spike_rate_vs_pupil_radius_cent',
                                        'spike_rate_vs_pupil_radius_tuning',
                                        'spike_rate_vs_pupil_radius_err',
                                        'spike_rate_vs_theta_cent',
                                        'spike_rate_vs_theta_tuning',
                                        'spike_rate_vs_theta_err',
                                        'trange',
                                        'theta',
                                        'phi',
                                        'spike_rate_vs_spd_cent',
                                        'spike_rate_vs_spd_tuning',
                                        'spike_rate_vs_spd_err',
                                        'spike_rate_vs_phi_cent',
                                        'spike_rate_vs_phi_tuning',
                                        'spike_rate_vs_phi_err']]
            unit_df = pd.DataFrame(pd.Series([crange,
                                    crf_cent,
                                    crf_tuning[unit_num],
                                    crf_err[unit_num],
                                    np.ndarray.flatten(staAll[unit_num]),
                                    np.shape(staAll[unit_num]),
                                    np.ndarray.flatten(st_var[unit_num]),
                                    upsacc_avg[unit_num],
                                    downsacc_avg[unit_num],
                                    spike_rate_vs_pupil_radius_cent,
                                    spike_rate_vs_pupil_radius_tuning[unit_num],
                                    spike_rate_vs_pupil_radius_err[unit_num],
                                    spike_rate_vs_theta_cent,
                                    spike_rate_vs_theta_tuning[unit_num],
                                    spike_rate_vs_theta_err[unit_num],
                                    trange,
                                    th,
                                    phi,
                                    spike_rate_vs_spd_cent,
                                    spike_rate_vs_spd_tuning[unit_num],
                                    spike_rate_vs_spd_err[unit_num],
                                    spike_rate_vs_phi_cent,
                                    spike_rate_vs_phi_tuning[unit_num],
                                    spike_rate_vs_phi_err[unit_num]]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
    elif file_dict['stim_type'] == 'white_noise':
        for unit_num, ind in enumerate(goodcells.index):
            cols = [stim+'_'+i for i in ['c_range',
                                        'crf_cent',
                                        'crf_tuning',
                                        'crf_err',
                                        'spike_triggered_average',
                                        'sta_shape',
                                        'spike_triggered_variance',
                                        'upsacc_avg',
                                        'downsacc_avg',
                                        'spike_rate_vs_pupil_radius_cent',
                                        'spike_rate_vs_pupil_radius_tuning',
                                        'spike_rate_vs_pupil_radius_err',
                                        'spike_rate_vs_theta_cent',
                                        'spike_rate_vs_theta_tuning',
                                        'spike_rate_vs_theta_err',
                                        'trange',
                                        'theta',
                                        'phi',
                                        'glm_receptive_field',
                                        'glm_cc',
                                        'spike_rate_vs_spd_cent',
                                        'spike_rate_vs_spd_tuning',
                                        'spike_rate_vs_spd_err',
                                        'spike_rate_vs_phi_cent',
                                        'spike_rate_vs_phi_tuning',
                                        'spike_rate_vs_phi_err']]
            unit_df = pd.DataFrame(pd.Series([crange,
                                    crf_cent,
                                    crf_tuning[unit_num],
                                    crf_err[unit_num],
                                    np.ndarray.flatten(staAll[unit_num]),
                                    np.shape(staAll[unit_num]),
                                    np.ndarray.flatten(st_var[unit_num]),
                                    upsacc_avg[unit_num],
                                    downsacc_avg[unit_num],
                                    spike_rate_vs_pupil_radius_cent,
                                    spike_rate_vs_pupil_radius_tuning[unit_num],
                                    spike_rate_vs_pupil_radius_err[unit_num],
                                    spike_rate_vs_theta_cent,
                                    spike_rate_vs_theta_tuning[unit_num],
                                    spike_rate_vs_theta_err[unit_num],
                                    trange,
                                    th,
                                    phi,
                                    glm_receptive_field[unit_num],
                                    glm_cc[unit_num],
                                    spike_rate_vs_spd_cent,
                                    spike_rate_vs_spd_tuning[unit_num],
                                    spike_rate_vs_spd_err[unit_num],
                                    spike_rate_vs_phi_cent,
                                    spike_rate_vs_phi_tuning[unit_num],
                                    spike_rate_vs_phi_err[unit_num]]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
    elif free_move is True:
        for unit_num, ind in enumerate(goodcells.index):
            cols = [stim+'_'+i for i in ['c_range',
                                        'crf_cent',
                                        'crf_tuning',
                                        'crf_err',
                                        'spike_triggered_average',
                                        'sta_shape',
                                        'spike_triggered_variance',
                                        'upsacc_avg',
                                        'downsacc_avg',
                                        'upsacc_avg_gaze_shift_dEye',
                                        'downsacc_avg_gaze_shift_dEye',
                                        'upsacc_avg_comp_dEye',
                                        'downsacc_avg_comp_dEye',
                                        'upsacc_avg_gaze_shift_dHead',
                                        'downsacc_avg_gaze_shift_dHead',
                                        'upsacc_avg_comp_dHead',
                                        'downsacc_avg_comp_dHead',
                                        'spike_rate_vs_pupil_radius_cent',
                                        'spike_rate_vs_pupil_radius_tuning',
                                        'spike_rate_vs_pupil_radius_err',
                                        'spike_rate_vs_theta_cent',
                                        'spike_rate_vs_theta_tuning',
                                        'spike_rate_vs_theta_err',
                                        'spike_rate_vs_gz_cent',
                                        'spike_rate_vs_gz_tuning',
                                        'spike_rate_vs_gz_err',
                                        'spike_rate_vs_gx_cent',
                                        'spike_rate_vs_gx_tuning',
                                        'spike_rate_vs_gx_err',
                                        'spike_rate_vs_gy_cent',
                                        'spike_rate_vs_gy_tuning',
                                        'spike_rate_vs_gy_err',
                                        'trange',
                                        'dHead',
                                        'dEye',
                                        'eyeT',
                                        'theta',
                                        'phi',
                                        'gz',
                                        'spike_rate_vs_roll_cent',
                                        'spike_rate_vs_roll_tuning',
                                        'spike_rate_vs_roll_err',
                                        'spike_rate_vs_pitch_cent',
                                        'spike_rate_vs_pitch_tuning',
                                        'spike_rate_vs_pitch_err',
                                        'glm_receptive_field',
                                        'glm_cc',
                                        'spike_rate_vs_phi_cent',
                                        'spike_rate_vs_phi_tuning',
                                        'spike_rate_vs_phi_err',
                                        'accT',
                                        'roll',
                                        'pitch',
                                        'roll_interp',
                                        'pitch_interp']]
            unit_df = pd.DataFrame(pd.Series([crange,
                                    crf_cent,
                                    crf_tuning[unit_num],
                                    crf_err[unit_num],
                                    np.ndarray.flatten(staAll[unit_num]),
                                    np.shape(staAll[unit_num]),
                                    np.ndarray.flatten(st_var[unit_num]),
                                    upsacc_avg[unit_num],
                                    downsacc_avg[unit_num],
                                    upsacc_avg_gaze_shift_dEye[unit_num],
                                    downsacc_avg_gaze_shift_dEye[unit_num],
                                    upsacc_avg_comp_dEye[unit_num],
                                    downsacc_avg_comp_dEye[unit_num],
                                    upsacc_avg_gaze_shift_dHead[unit_num],
                                    downsacc_avg_gaze_shift_dHead[unit_num],
                                    upsacc_avg_comp_dHead[unit_num],
                                    downsacc_avg_comp_dHead[unit_num],
                                    spike_rate_vs_pupil_radius_cent,
                                    spike_rate_vs_pupil_radius_tuning[unit_num],
                                    spike_rate_vs_pupil_radius_err[unit_num],
                                    spike_rate_vs_theta_cent,
                                    spike_rate_vs_theta_tuning[unit_num],
                                    spike_rate_vs_theta_err[unit_num],
                                    spike_rate_vs_gz_cent,
                                    spike_rate_vs_gz_tuning[unit_num],
                                    spike_rate_vs_gz_err[unit_num],
                                    spike_rate_vs_gx_cent,
                                    spike_rate_vs_gx_tuning[unit_num],
                                    spike_rate_vs_gx_err[unit_num],
                                    spike_rate_vs_gy_cent,
                                    spike_rate_vs_gy_tuning[unit_num],
                                    spike_rate_vs_gy_err[unit_num],
                                    trange,
                                    dhead,
                                    dEye,
                                    eyeT,
                                    th,
                                    phi,
                                    gz,
                                    spike_rate_vs_roll_cent,
                                    spike_rate_vs_roll_tuning[unit_num],
                                    spike_rate_vs_roll_err[unit_num],
                                    spike_rate_vs_pitch_cent,
                                    spike_rate_vs_pitch_tuning[unit_num],
                                    spike_rate_vs_pitch_err[unit_num],
                                    glm_receptive_field[unit_num],
                                    glm_cc[unit_num],
                                    spike_rate_vs_phi_cent,
                                    spike_rate_vs_phi_tuning[unit_num],
                                    spike_rate_vs_phi_err[unit_num],
                                    accT,
                                    roll,
                                    pitch,
                                    roll_interp,
                                    pitch_interp]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
            
    data_out = pd.concat([goodcells, unit_data],axis=1)

    data_out.to_hdf(os.path.join(file_dict['save'], (file_dict['name']+'_ephys_props.h5')), 'w')

    print('clearing memory')

    del data_out
    gc.collect()

    print('done')