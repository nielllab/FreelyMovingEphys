"""
analyze_ephys.py

make ephys figures
called by analysis jupyter notebook

Jan. 20, 2021
"""
# package imports
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
import time
import subprocess
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl 
import wavio
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe' # use for windows lab computers
# mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg' # user has to change to this line on ubuntu
from scipy.interpolate import interp1d
from numpy import nan
from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from util.aux_funcs import nanxcorr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from scipy.ndimage import shift as imshift
from scipy import signal
from sklearn.cluster import KMeans
# module imports
from project_analysis.ephys.ephys_figures import *

def find_files(rec_path, rec_name, free_move, cell, stim_type):
    print('find ephys files')

    # get the files names in the provided path
    eye_file = os.path.join(rec_path, rec_name + '_Reye.nc')
    world_file = os.path.join(rec_path, rec_name + '_world.nc')
    ephys_file = os.path.join(rec_path, rec_name + '_ephys_merge.json')
    imu_file = os.path.join(rec_path, rec_name + '_imu.nc')
    speed_file = os.path.join(rec_path, rec_name + '_speed.nc')

    if stim_type == 'gratings':
        stim_type = 'grat'
    elif stim_type == 'white_noise':
        pass
    elif stim_type == 'sparse_noise':
        pass
    else:
        stim_type = None

    if free_move is True:
        dict_out = {'cell':cell,'eye':eye_file,'world':world_file,'ephys':ephys_file,'speed':None,'imu':imu_file,'save':rec_path,'name':rec_name,'stim_type':stim_type}
    elif free_move is False:
        dict_out = {'cell':cell,'eye':eye_file,'world':world_file,'ephys':ephys_file,'speed':speed_file,'imu':None,'save':rec_path,'name':rec_name,'stim_type':stim_type}

    return dict_out

def run_ephys_analysis(file_dict):

    if file_dict['speed'] is None:
        free_move = True; has_imu = True; has_mouse = True
    else:
        free_move = False; has_imu = False; has_mouse = True

    print('opening pdfs')
    # three pdf outputs will be saved
    overview_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_overview_analysis_figures.pdf')))
    detail_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_detailed_analysis_figures.pdf')))
    diagnostic_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_diagnostic_analysis_figures.pdf')))

    print('opening data')
    # load worldcam
    world_data = xr.open_dataset(file_dict['world'])
    world_vid_raw = np.uint8(world_data['WORLD_video'])

    # resize worldcam to make more manageable
    sz = world_vid_raw.shape
    downsamp = 0.5
    world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
    for f in range(sz[0]):
        world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
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

    # load IMU data
    if file_dict['imu'] is not None:
        imu_data = xr.open_dataset(file_dict['imu'])
        accT = imu_data.timestamps
        acc_chans = imu_data.IMU_data
        try:
            gx = np.array(acc_chans.sel(channel='gyro_x'))
            gy = np.array(acc_chans.sel(channel='gyro_y'))
            gz = np.array(acc_chans.sel(channel='gyro_z'))
        except:
            gx = np.array(acc_chans.sel(sample='gyro_x'))
            gy = np.array(acc_chans.sel(sample='gyro_y'))
            gz = np.array(acc_chans.sel(sample='gyro_z'))

    # load optical mouse data
    if file_dict['speed'] is not None:
        speed_data = xr.open_dataset(file_dict['speed'])
        spdVals = speed_data.BALL_data
        try:
            spd = spdVals.sel(move_params = 'cm_per_sec')
            spd_tstamps = spdVals.sel(move_params = 'timestamps')
        except:
            spd = spdVals.sel(frame = 'cm_per_sec')
            spd_tstamps = spdVals.sel(frame = 'timestamps')


    # read ephys data
    ephys_data = pd.read_json(file_dict['ephys'])
    ephys_data['spikeTraw'] = ephys_data['spikeT']

    # select good cells from phy2
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    goodcells.shape
    units = goodcells.index.values

    # get number of good units
    n_units = len(goodcells)

    # spike rasters
    spikeraster_fig = plot_spike_rasters(goodcells)
    detail_pdf.savefig()
    plt.close()

    # load eye data
    eye_data = xr.open_dataset(file_dict['eye'])
    eye_vid = np.uint8(eye_data['REYE_video'])
    eyeT = eye_data.timestamps.copy()

    # plot eye timestamps
    reyecam_fig = plot_cam_time(worldT, 'reye')
    diagnostic_pdf.savefig()
    plt.close()

    # plot eye postion across recording
    eye_params = eye_data['REYE_ellipse_params']
    eyepos_fig = plot_eye_pos(eye_params)
    detail_pdf.savefig()
    plt.close()

    if file_dict['speed'] is not None:
        # plot optical mouse speeds
        optical_mouse_sp_fig = plot_optmouse_spd(spd_tstamps, spd)
        detail_pdf.savefig()
        plt.close()

    # adjust eye/world/top times relative to ephys
    ephysT0 = ephys_data.iloc[0,12]
    eyeT = eye_data.timestamps  - ephysT0
    if eyeT[0]<-600:
        eyeT = eyeT + 8*60*60 # 8hr offset for some data
    worldT = world_data.timestamps - ephysT0
    if worldT[0]<-600:
        worldT = worldT + 8*60*60
    if free_move is True and has_imu is True:
        accTraw = imu_data.timestamps-ephysT0
    if free_move is False and has_mouse is True:
        speedT = spd_tstamps-ephysT0

    # check that deinterlacing worked correctly
    # plot theta and theta switch
    # want theta switch to be jagged, theta to be smooth
    theta_switch_fig, th_switch = plot_param_switch_check(eye_params)
    diagnostic_pdf.savefig()
    plt.close()

    # plot eye variables
    eye_param_fig = plot_eye_params(eye_params, eyeT)
    detail_pdf.savefig()
    plt.close()

    # calculate eye veloctiy
    dEye = np.diff(np.rad2deg(eye_params.sel(ellipse_params='theta')))

    print('checking accelerometer / eye temporal alignment')
    # check accelerometer / eye temporal alignment
    if file_dict['imu'] is not None:
        lag_range = np.arange(-0.2,0.2,0.002)
        cc = np.zeros(np.shape(lag_range))
        t1 = np.arange(5,1600,20)
        t2 = t1 + 60
        offset = np.zeros(np.shape(t1))
        ccmax = np.zeros(np.shape(t1))
        acc_interp = interp1d(accTraw, (gz-3)*7.5)
        for tstart in range(len(t1)):
            for l in range(len(lag_range)):
                try:
                    c, lag= nanxcorr(dEye[t1[tstart]*60 : t2[tstart]*60] + 0.5/60, acc_interp(eyeT[t1[tstart]*60:t2[tstart]*60]+lag_range[l]),1)
                    cc[l] = c[1]
                except: # occasional probelm with operands that cannot be broadcast togther because of different shapes
                    cc[l] = np.nan
            offset[tstart] = lag_range[np.argmax(cc)]    
            ccmax[tstart] = np.max(cc)
        offset[ccmax<0.1] = np.nan
        acc_eyetime_alligment_fig = plot_acc_eyetime_alignment(eyeT, t1, offset, ccmax)
        diagnostic_pdf.savefig()
        plt.close()

    print('fitting regression to timing drift')
    # fit regression to timing drift
    if file_dict['imu'] is not None:
        model = LinearRegression()
        dataT = np.array(eyeT[t1*60 + 30])
        model.fit(dataT[offset>0].reshape(-1,1),offset[offset>0])
        offset0 = model.intercept_
        drift_rate = model.coef_
        plot_regression_timing_fit_fig = plot_regression_timing_fit(dataT, offset, offset0, drift_rate)
        diagnostic_pdf.savefig()
        plt.close()

    elif file_dict['speed'] is not None:
        offset0 = 0.1
        drift_rate = - 0.1/1000

    if file_dict['imu'] is not None:
        accT = accTraw - (offset0 + accTraw*drift_rate)

    for i in range(len(ephys_data)):
        ephys_data['spikeT'].iloc[i] = np.array(ephys_data['spikeTraw'].iloc[i]) - (offset0 + np.array(ephys_data['spikeTraw'].iloc[i]) *drift_rate)

    print('finding contrast of normalized worldcam')
    # normalize world movie and calculate contrast
    cam_gamma = 2
    world_norm = (world_vid/255)**cam_gamma
    std_im = np.std(world_norm,axis=0)
    std_im[std_im<10/255] = 10/255
    img_norm = (world_norm-np.mean(world_norm,axis=0))/std_im

    contrast = np.empty(worldT.size)
    for i in range(worldT.size):
        contrast[i] = np.std(img_norm[i,:,:])
    plt.plot(worldT[0:6000],contrast[0:6000])
    plt.xlabel('time')
    plt.ylabel('contrast')
    diagnostic_pdf.savefig()
    plt.close()

    fig = plt.figure()
    plt.imshow(std_im)
    plt.colorbar(); plt.title('std img')
    diagnostic_pdf.savefig()
    plt.close()

    # set up interpolators for eye and world videos
    eyeInterp = interp1d(eyeT,eye_vid,axis=0)
    worldInterp = interp1d(worldT,world_vid,axis=0)

    # make movie and sound
    print('making video figure')
    this_unit = file_dict['cell']

    if file_dict['imu'] is not None:
        vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, accT=accT, gz=gz)
    elif file_dict['speed'] is not None:
        vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, speedT=speedT, spd=spd)

    print('making audio figure')
    audfile = make_sound(file_dict, ephys_data, units, this_unit)
    
    # merge video and audio
    merge_mp4_name = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'_merge.mp4'))

    print('merging movie with sound')
    subprocess.call(['ffmpeg', '-i', vidfile, '-i', audfile, '-c:v', 'copy', '-c:a', 'aac', merge_mp4_name])

    th = np.array((eye_params.sel(ellipse_params = 'theta')-np.nanmean(eye_params.sel(ellipse_params = 'theta')))*180/3.14159)
    phi = np.array((eye_params.sel(ellipse_params = 'phi')-np.nanmean(eye_params.sel(ellipse_params = 'phi')))*180/3.14159)

    if free_move is True and file_dict['imu'] is not None:
        plt.figure()
        plt.plot(eyeT[0:-1],np.diff(th),label = 'dTheta')
        plt.plot(accT-0.1,(gz-3)*10, label = 'gyro')
        plt.xlim(30,40); plt.ylim(-12,12); plt.legend(); plt.xlabel('secs')
        diagnostic_pdf.savefig()
        plt.close()

    print('plot eye and gaze (i.e. saccade and fixate)')
    dEye = np.diff(th)
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
    contrast_interp.shape
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
    # calculate contrast - response functions
    # mean firing rate in timebins correponding to contrast ranges
    resp = np.empty((n_units,12))
    crange = np.arange(0,1.2,0.1)
    for i,ind in enumerate(goodcells.index):
        for c,cont in enumerate(crange):
            resp[i,c] = np.mean(goodcells.at[ind,'rate'][(contrast_interp>cont) & (contrast_interp<(cont+0.1))])
    plt.plot(crange,np.transpose(resp))
    #plt.ylim(0,10)
    plt.xlabel('contrast')
    plt.ylabel('sp/sec')
    plt.title('mean firing rate in timebins correponding to contrast ranges')
    detail_pdf.savefig()
    plt.close()

    print('plotting individual contrast response functions')
    # plot individual contrast response functions in subplots
    ind_contrast_funcs_fig = plot_ind_contrast_funcs(n_units, goodcells, crange, resp)
    detail_pdf.savefig()
    plt.close()

    # create interpolator for movie data so we can evaluate at same timebins are firing rate
    img_norm[img_norm<-2] = -2
    movInterp = interp1d(worldT,img_norm,axis=0)

    print('getting spike-triggered average for lag=0.125')
    # calculate spike-triggered average
    staAll, STA_single_lag_fig = plot_STA_single_lag(n_units, img_norm, goodcells, worldT, movInterp)
    detail_pdf.savefig()
    plt.close()

    if file_dict['stim_type'] == 'grat':
        print('getting grating flow')
        nf = np.size(img_norm,0)-1
        u_mn = np.zeros((nf,1)); v_mn = np.zeros((nf,1))
        sx_mn = np.zeros((nf,1)) ; sy_mn = np.zeros((nf,1))
        flow_norm = np.zeros((nf,np.size(img_norm,1),np.size(img_norm,2),2 ))
        vidfile = os.path.join(file_dict['save'], (file_dict['name']+'_grating_flow'))

        fig, ax = plt.subplots(1,1,figsize = (16,8))
        # now animate
        #writer = FFMpegWriter(fps=30)
        #with writer.saving(fig, vidfile, 100):
        for f in tqdm(range(nf)):
            frm = np.uint8(32*(img_norm[f,:,:]+4))
            frm2 = np.uint8(32*(img_norm[f+1,:,:]+4))
            # frm = cv2.resize(frm, (0,0), fx=0.5); frm2 = cv2.resize(frm2, (0,0), fx=0.5) # added resizing frames to a downscaled resolution
            flow_norm[f,:,:,:] = cv2.calcOpticalFlowFarneback(frm,frm2, None, 0.5, 3, 30, 3, 7, 1.5, 0)
            #ax.cla()
            #ax.imshow(frm,vmin = 0, vmax = 255)
            u = flow_norm[f,:,:,0]; v = -flow_norm[f,:,:,1]  # negative to fix sign for y axis in images
            sx = cv2.Sobel(frm,cv2.CV_64F,1,0,ksize=7)
            sy = -cv2.Sobel(frm,cv2.CV_64F,0,1,ksize=7)# negative to fix sign for y axis in images
            sx[std_im<0.05]=0; sy[std_im<0.05]=0; # get rid of values outside of monitor
            sy[sx<0] = -sy[sx<0]  #make vectors point in positive x direction (so opposite sides of grating don't cancel)
            sx[sx<0] = -sx[sx<0]
            #ax.quiver(x[::nx,::nx],y[::nx,::nx],sx[::nx,::nx],sy[::nx,::nx], scale = 100000 )
            u_mn[f]= np.mean(u); v_mn[f]= np.mean(v); sx_mn[f] = np.mean(sx); sy_mn[f] = np.mean(sy)
            #plt.title(str(np.round(np.arctan2(sy_mn[f],sx_mn[f])*180/np.pi))
            #writer.grab_frame()

        stimOn = contrast>0.5
        stimOn = signal.medfilt(stimOn,11)

        stim_start = np.array(worldT[np.where(np.diff(stimOn)>0)])
        stim_end = np.array(worldT[np.where(np.diff(stimOn)<0)])
        stim_end = stim_end[stim_end>stim_start[0]]
        stim_start = stim_start[stim_start<stim_end[-1]]
        grating_th = np.zeros(len(stim_start))
        grating_mag = np.zeros(len(stim_start))
        grating_dir = np.zeros(len(stim_start))
        for i in range(len(stim_start)):
            stim_u = np.median(u_mn[np.where((worldT>stim_start[i] + 0.025) & (worldT<stim_end[i]-0.025))])
            stim_v = np.median(v_mn[np.where((worldT>stim_start[i] + 0.025) & (worldT<stim_end[i]-0.025))])
            stim_sx = np.median(sx_mn[np.where((worldT>stim_start[i] + 0.025) & (worldT<stim_end[i]-0.025))])
            stim_sy = np.median(sy_mn[np.where((worldT>stim_start[i] + 0.025) & (worldT<stim_end[i]-0.025))])
            grating_th[i] = np.arctan2(stim_sy,stim_sx)
            grating_mag[i] = np.sqrt(stim_sx**2 + stim_sy**2)
            grating_dir[i] = np.sign(stim_u*stim_sx + stim_v*stim_sy) # dot product of gratient and flow gives direction
        #grating_th = np.round(grating_th *10)/10

        grating_ori = grating_th.copy()
        grating_ori[grating_dir<0] = grating_ori[grating_dir<0] + np.pi
        grating_ori = grating_ori - np.min(grating_ori)
        np.unique(grating_ori)
        plt.figure(figsize = (8,8))

        ori_cat = np.floor((grating_ori+np.pi/8)/(np.pi/4))

        # might be a bad idea...
        # replace all NaN values in grating_mag with 0, same for pos/neg inf
        # any NaN value raises ValueError in KMeans below
        grating_mag = np.nan_to_num(grating_mag, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

        km = KMeans(n_clusters=3).fit(np.reshape(grating_mag,(-1,1)))
        sf_cat = km.labels_
        order = np.argsort(np.reshape(km.cluster_centers_, 3))
        sf_catnew = sf_cat.copy()
        for i in range(3):
            sf_catnew[sf_cat == order[i]]=i
        sf_cat = sf_catnew.copy()
        plt.scatter(grating_mag,grating_ori,c=ori_cat)
        detail_pdf.savefig()
        plt.plot()

        print('plotting grading orientation and tuning curves')
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
            plt.legend()
            try:
                plt.ylim(0,np.nanmax(ori_tuning[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
        detail_pdf.savefig()
        plt.close()

    print('getting spike-triggered average with range in lags')
    # calculate spike-triggered average
    fig = plot_STA_multi_lag(n_units, goodcells, worldT, movInterp)
    detail_pdf.savefig()
    plt.close()

    print('getting spike-triggered variance')
    # calculate spike-triggered variance
    fig = plot_spike_triggered_variance(n_units, goodcells, t, movInterp, img_norm)
    detail_pdf.savefig()
    plt.close()

    print('getting rasters around saccades')
    # calculate saccade-locked psth
    spike_corr = 1 #+ 0.125/1200  # correction factor for ephys timing drift
    dEye= np.diff(th)

    trange = np.arange(-1,1.1,0.1)
    sthresh = 8
    upsacc = eyeT[np.append(dEye,0)>sthresh]/spike_corr
    upsacc = upsacc[upsacc>5]
    upsacc = upsacc[upsacc<np.max(t)-5]
    downsacc = eyeT[np.append(dEye,0)<-sthresh]/spike_corr
    downsacc = downsacc[downsacc>5]
    downsacc = downsacc[downsacc<np.max(t)-5]
    upsacc_avg = np.zeros((units.size,trange.size))
    downsacc_avg = np.zeros((units.size,trange.size))

    upsacc_avg, downsacc_avg, saccade_lock_fig = plot_saccade_locked(n_units, goodcells, t, upsacc, upsacc_avg, trange, downsacc, downsacc_avg)
    detail_pdf.savefig()
    plt.close()

    # rasters around positive saccades
    raster_around_upsacc_fig = plot_rasters_around_saccades(n_units, goodcells, upsacc)
    detail_pdf.savefig()
    plt.close()

    #rasters around negative saccades
    raster_around_downsacc_fig = plot_rasters_around_saccades(n_units, goodcells, downsacc)
    detail_pdf.savefig()
    plt.close()

    # normalize and plot eye radius
    eyeR = eye_params.sel(ellipse_params = 'longaxis').copy()
    Rnorm = (eyeR - np.mean(eyeR))/np.std(eyeR)
    plt.plot(eyeT,Rnorm)
    plt.xlim([0,60])
    plt.xlabel('secs')
    plt.ylabel('normalized pupil R')
    diagnostic_pdf.savefig()
    plt.close()

    print('plotting spike rate vs pupil radius')
    # plot rate vs pupil
    n_units = len(goodcells)
    R_range = np.arange(-4,4,0.5)
    useEyeT = eyeT[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    useR = Rnorm[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    spike_rate_vs_pupil_radius_fig = plot_spike_rate_vs_pupil_radius(n_units, useR, R_range, goodcells, useEyeT, t, 'rad')
    detail_pdf.savefig()
    plt.close()

    # normalize eye position
    eyeTheta = eye_params.sel(ellipse_params = 'theta').copy()
    thetaNorm = (eyeTheta - np.mean(eyeTheta))/np.std(eyeTheta)
    plt.plot(eyeT[0:3600],thetaNorm[0:3600])
    plt.xlabel('secs'); plt.ylabel('normalized eye theta')
    diagnostic_pdf.savefig()
    plt.close()

    print('plotting spike rate vs theta')
    # plot rate vs theta
    n_units = len(goodcells)
    th_range = np.arange(-2,3,0.5)
    useEyeT = eyeT[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    useTh = thetaNorm[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    spike_rate_vs_pupil_radius_fig = plot_spike_rate_vs_pupil_radius(n_units, useTh, th_range, goodcells, useEyeT, t, 'rad')
    detail_pdf.savefig()
    plt.close()

    print('generating summary plot')
    # generate summary plot
    summary_fig = plot_summary(n_units, goodcells, crange, resp, file_dict, ori_tuning, drift_spont, staAll, trange, upsacc_avg, downsacc_avg)
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
        plt.ylabel('sp/sec'); plt.xlim(bins[0],bins[-1]); plt.title('unit ' + str(i))
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    overview_pdf.close(); detail_pdf.close(); diagnostic_pdf.close()

    print('organizing data and saving as xarray')

    split_base_name = file_dict['name'].split('_')

    date = split_base_name[0]; mouse = split_base_name[1]; exp = split_base_name[2]; rig = split_base_name[3]
    try:
        stim = '_'.join(split_base_name[4:])
    except:
        stim = split_base_name[4:]
    var_names = ['_'.join([mouse, date, exp, rig, stim, 'unit'+str(i)]) for i in range(1,n_units+1)]
    
    unit_names = [(file_dict['name']+'_unit'+str(i)) for i in range(1,n_units+1)]
    if file_dict['stim_type'] == 'grat':
        all_units = {}
        for unit_num, ind in enumerate(goodcells.index):
            unit = unit_num+1
            unit_dict = {
                'contrast_range': crange,
                'orientation_tuning':ori_tuning[unit_num],
                'drift_spont': drift_spont[unit_num],
                'contrast_response': resp[unit_num],
                'waveform': goodcells.at[ind,'waveform'],
                'trange': trange,
                'upsacc_avg': upsacc_avg[unit_num],
                'downsacc_avg':downsacc_avg[unit_num]
            }
            all_units[var_names[unit_num]] = unit_dict
    elif file_dict['stim_type'] != 'grat':
        all_units = {}
        for unit_num, ind in enumerate(goodcells.index):
            unit = unit_num+1
            unit_dict = {
                'contrast_range': crange,
                'sta':staAll[unit_num],
                'contrast_response': resp[unit_num],
                'waveform': goodcells.at[ind,'waveform'],
                'trange': trange,
                'upsacc_avg': upsacc_avg[unit_num],
                'downsacc_avg':downsacc_avg[unit_num]
            }
            all_units[var_names[unit_num]] = unit_dict

    np.save(os.path.join(file_dict['save'], (file_dict['name']+'_ephys_props.npy')), all_units)
    # have to open like d1.item().get('name_of_unit')

    print('analysis complete; pdfs closed and .npy saved to file')
