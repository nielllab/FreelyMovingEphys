"""
analyze_ephys.py

make ephys figures
called by analysis jupyter notebook

Dec. 12, 2020
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
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
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

def find_files(rec_path, rec_name, free_move, cell):
    # get the files names in the provided path
    eye_file = os.path.join(rec_path, rec_name + '_Reye.nc')
    world_file = os.path.join(rec_path, rec_name + '_world.nc')
    ephys_file = os.path.join(rec_path, rec_name + '_ephys_merge.json')
    imu_file = os.path.join(rec_path, rec_name + '_imu.nc')
    speed_file = os.path.join(rec_path, rec_name + '_speed.nc')

    if free_move is True:
        dict_out = {'cell':cell,'eye':eye_file,'world':world_file,'ephys':ephys_file,'speed':None,'imu':imu_file,'save':rec_path,'name':rec_name}
    elif free_move is False:
        dict_out = {'cell':cell,'eye':eye_file,'world':world_file,'ephys':ephys_file,'speed':speed_file,'imu':None,'save':rec_path,'name':rec_name}

    return dict_out

def run_ephys_analysis(file_dict):

    if file_dict['speed'] is None:
        free_move = True; has_imu = True; has_mouse = True
    else:
        free_move = False; has_imu = False; has_mouse = True

    # three pdf outputs will be saved
    overview_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_overview_analysis_figures.pdf')))
    detail_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_detailed_analysis_figures.pdf')))
    diagnostic_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_diagnostic_analysis_figures.pdf')))

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

    if file_dict['imu'] is not None:
        imu_data = xr.open_dataset(file_dict['imu'])
        accT = imu_data.timestamps
        acc_chans = imu_data.IMU_data
        gx = np.array(acc_chans.sel(channel='gyro_x'))
        gy = np.array(acc_chans.sel(channel='gyro_y'))
        gz = np.array(acc_chans.sel(channel='gyro_z'))

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
    theta_switch_fig = plot_param_switch_check(eye_params)
    diagnostic_pdf.savefig()
    plt.close()

    # plot eye variables
    eye_param_fig = plot_eye_params(eye_params, eyeT)
    detail_pdf.savefig()
    plt.close()

    # calculate eye veloctiy
    dEye = np.diff(np.rad2deg(eye_params.sel(ellipse_params='theta')))

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
                c, lag= nanxcorr(dEye[t1[tstart]*60 : t2[tstart]*60] + 0.5/60, acc_interp(eyeT[t1[tstart]*60:t2[tstart]*60]+lag_range[l]),1)
                cc[l] = c[1]
            offset[tstart] = lag_range[np.argmax(cc)]    
            ccmax[tstart] = np.max(cc)
        offset[ccmax<0.1] = nan
        plt.subplot(1,2,1)
        plt.plot(eyeT[t1*60],offset)
        plt.xlabel('secs'); plt.ylabel('offset - secs')
        plt.subplot(1,2,2)
        plt.plot(eyeT[t1*60],ccmax)
        plt.xlabel('secs'); plt.ylabel('max cc')
        diagnostic_pdf.savefig()
        plt.close()

    # fit regression to timing drift
    if file_dict['imu'] is not None:
        model = LinearRegression()
        dataT = np.array(eyeT[t1*60 + 30])
        model.fit(dataT[offset>0].reshape(-1,1),offset[offset>0])
        offset0 = model.intercept_
        drift_rate = model.coef_
        plt.plot(dataT,offset,'.')
        plt.plot(dataT, offset0 + dataT*drift_rate)
        plt.xlabel('secs'); plt.ylabel('offset - secs')
        print(offset0)
        print(drift_rate)
    elif file_dict['speed'] is not None:
        offset0 = 0.1
        drift_rate = 0.1/1000

    if file_dict['imu'] is not None:
        accT = accTraw - (offset0 + accTraw*drift_rate)

    for i in range(len(ephys_data)):
        ephys_data['spikeT'].iloc[i] = np.array(ephys_data['spikeTraw'].iloc[i]) - (offset0 + np.array(ephys_data['spikeTraw'].iloc[i]) *drift_rate)

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
    this_unit = file_dict['cell']

    if file_dict['imu'] is not None:
        vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, accT=accT, gz=gz)
    elif file_dict['speed'] is not None:
        vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, speedT=speedT, spd=spd)

    audfile = make_sound(file_dict, ephys_data, this_unit)
    
    # merge video and audio
    merge_mp4_name = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+this_unit+'_merge.mp4'))
    subprocess.call(['ffmpeg', '-i', vidfile, '-i', audfile, '-c:v', 'copy', '-c:a', 'aac', merge_mp4_name]) 

    if free_move:
        plt.figure()
        plt.plot(eyeT[0:-1],np.diff(th_switch),label = 'dTheta')
        plt.plot(accT-0.1,(gz-3)*10, label = 'gyro')
        plt.xlim(30,40); plt.ylim(-12,12); plt.legend(); plt.xlabel('secs')
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

    # calculate firing rate at new timebase
    ephys_data['rate'] = nan
    ephys_data['rate'] = ephys_data['rate'].astype(object)
    for i,ind in enumerate(ephys_data.index):
        ephys_data.at[ind,'rate'],bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
    ephys_data['rate']= ephys_data['rate']/dt
    goodcells = ephys_data.loc[ephys_data['group']=='good']

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

    # plot individual contrast response functions in subplots
    fig = plt.figure(figsize = (6,np.ceil(n_units/2)))
    for i, ind in enumerate(goodcells.index):
        plt.subplot(np.ceil(n_units/4),4,i+1)
        plt.plot(crange[2:-1],resp[i,2:-1])
    # plt.ylim([0 , max(resp[i,1:-3])*1.2])
        plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec'); plt.ylim([0,np.nanmax(resp[i,2:-1])])
    plt.tight_layout()
    plt.title('individual contrast reponse')
    detail_pdf.savefig()
    plt.close()

    # create interpolator for movie data so we can evaluate at same timebins are firing rat
    img_norm[img_norm<-2] = -2
    movInterp = interp1d(worldT,img_norm,axis=0)

    # calculate spike-triggered average
    spike_corr = 1 + 0.125/1200  # correction factor for ephys timing drift

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
                im = movInterp((s-lag)*spike_corr);
                if c==1:
                    ensemble[nsp-1,:,:] = im
                sta = sta+im;
        plt.subplot(np.ceil(n_units/4),4,c+1)
        sta = sta/nsp
        #sta[abs(sta)<0.1]=0
        plt.imshow((sta-np.mean(sta) ),vmin=-0.3,vmax=0.3,cmap = 'jet')
        staAll[c,:,:] = sta;
    plt.title('soike triggered average (lag=0.125)')
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    # calculate spike-triggered average
    spike_corr = 1 + 0.125/1200
    sta = 0
    lag = 0.075
    lagRange = np.arange(0,0.25,0.05)
    plt.figure(figsize = (12,2*n_units))
    for c, ind in enumerate(goodcells.index):
        sp = goodcells.at[ind,'spikeT'].copy()
        for  lagInd, lag in enumerate(lagRange):
            sta = 0; nsp = 0
            for s in sp:
                if (s-lag >5) & ((s-lag)*spike_corr <np.max(worldT)):
                    nsp = nsp+1
                    sta = sta+movInterp((s-lag)*spike_corr)
            plt.subplot(n_units,6,(c*6)+lagInd + 1)
            sta = sta/nsp
        #sta[abs(sta)<0.1]=0
            plt.imshow(sta ,vmin=-0.35,vmax=0.35,cmap = 'jet')
            plt.title(str(c) + ' ' + str(np.round(lag*1000)) + 'msec')
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    # calculate spike-triggered variance
    sta = 0
    lag = 0.125
    plt.figure(figsize = (12,np.ceil(n_units/2)))
    for c, ind in enumerate(goodcells.index):
        r = goodcells.at[ind,'rate']
        sta = 0
        for i in range(5,t.size-10):
            sta = sta+r[i]*(movInterp(t[i]-lag))**2
        plt.subplot(np.ceil(n_units/4),4,c+1)
        sta = sta/np.sum(r)
        plt.imshow(sta - np.mean(img_norm**2,axis=0),vmin=-1,vmax=1)
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    # calculate saccade-locked psth
    # spike_corr = 1 + 0.125/1200  # correction factor for ephys timing drift
    dEye= np.diff(th_switch)

    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
    trange = np.arange(-1,1.1,0.1)
    sthresh = 8
    upsacc = eyeT[np.append(dEye,0)>sthresh]/spike_corr
    upsacc = upsacc[upsacc>5]
    upsacc = upsacc[upsacc<np.max(t)-5]
    downsacc= eyeT[np.append(dEye,0)<-sthresh]/spike_corr
    downsacc = downsacc[downsacc>5]
    downsacc = downsacc[downsacc<np.max(t)-5]
    upsacc_avg = np.zeros((units.size,trange.size))
    downsacc_avg = np.zeros((units.size,trange.size))
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'])
        for s in upsacc:
            upsacc_avg[i,:] = upsacc_avg[i,:]+ rateInterp(np.array(s)+trange)/upsacc.size
        for s in downsacc:
            downsacc_avg[i,:]= downsacc_avg[i,:]+ rateInterp(np.array(s)+trange)/downsacc.size
        plt.subplot(np.ceil(n_units/4),4,i+1)
        plt.plot(trange,upsacc_avg[i,:])
        plt.plot(trange,downsacc_avg[i,:],'r')
        plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2),'r')
        plt.ylim([0, np.max(upsacc_avg[i,:])*1.8])
        plt.ylabel('sp/sec')
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    # rasters around positive saccades
    fig = plt.figure(figsize = (12,n_units)) 
    for i, ind in enumerate(goodcells.index): 
        sp = np.array(goodcells.at[units[i],'spikeT']) *spike_corr
        plt.subplot(np.ceil(n_units/4),4,i+1) 
        n = 0 
        for s in upsacc: 
            n= n+1 
            sd = np.abs(sp-np.array(s))<10 
            sacc_sp = sp[sd] 
            plt.vlines(sacc_sp-np.array(s),n-0.25,n+0.25) 
        plt.xlim(-1,1); #plt.ylim(0,50)
    detail_pdf.savefig()
    plt.close()

    #rasters around negative saccades
    fig = plt.figure(figsize = (12,n_units))
    for i, ind in enumerate(goodcells.index):
        sp = np.array(goodcells.at[units[i],'spikeT'])
        plt.subplot(np.ceil(n_units/4),4,i+1)
        n = 0
        for s in downsacc:
            n= n+1
            sd = np.abs(sp-np.array(s))<10
            sacc_sp = sp[sd]
            plt.vlines(sacc_sp-np.array(s),n-0.25,n+0.25)
        plt.xlim(-1,1)
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

    # plot rate vs pupil
    n_units = len(goodcells)
    R_range = np.arange(-4,4,0.5)
    useEyeT = eyeT[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    useR = Rnorm[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    R_scatter = np.zeros((n_units,len(useR)))
    R_tuning = np.zeros((n_units,len(R_range)-1))
    R_tuning_err =R_tuning.copy()
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'])
        R_scatter[i,:] = rateInterp(useEyeT)
        for j in range(len(R_range)-1):
            usePts =(useR>R_range[j]) & (useR<R_range[j+1])
            R_tuning[i,j] = np.mean(R_scatter[i,usePts])
            R_tuning_err[i,j] = np.std(R_scatter[i,usePts])/np.sqrt(np.count_nonzero(usePts))
    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
    for i in range(n_units):
        plt.subplot(np.ceil(n_units/4),4,i+1)
        plt.errorbar(R_range[:-1],R_tuning[i,:],yerr=R_tuning_err[i,:])
        plt.ylim(0,np.nanmax(R_tuning[i,2:-2]*1.2))
        plt.xlim([-2, 2])
        plt.xlabel('normalized pupil R'); plt. ylabel('sp/sec'); plt.title(i)
    plt.tight_layout()
    details_pdf.savefig()
    plt.close()

    # normalize eye position
    eyeTheta = eye_params.sel(ellipse_params = 'theta').copy()
    thetaNorm = (eyeTheta - np.mean(eyeTheta))/np.std(eyeTheta)
    plt.plot(eyeT[0:3600],thetaNorm[0:3600])
    plt.xlabel('secs'); plt.ylabel('normalized eye theta')
    diagnostic_pdf.savefig()
    plt.close()

    # plot rate vs theta
    n_units = len(goodcells)
    th_range = np.arange(-2,3,0.5)
    useEyeT = eyeT[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    useTh = thetaNorm[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    th_scatter = np.zeros((n_units,len(useR)))
    th_tuning = np.zeros((n_units,len(th_range)-1))
    th_tuning_err =th_tuning.copy()
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'])
        th_scatter[i,:] = rateInterp(useEyeT)
        for j in range(len(th_range)-1):
            usePts =(useTh>th_range[j]) & (useTh<th_range[j+1])
            th_tuning[i,j] = np.mean(th_scatter[i,usePts])
            th_tuning_err[i,j] = np.std(th_scatter[i,usePts])/np.sqrt(np.count_nonzero(usePts))
    fig = plt.figure(figsize = (3*np.ceil(n_units/2),6))
    for i in range(n_units):
        plt.subplot(2,np.ceil(n_units/2),i+1)
        plt.errorbar(th_range[:-1],th_tuning[i,:],yerr=th_tuning_err[i,:])
        plt.ylim(0,np.nanmax(th_tuning[i,:]*1.2))
        plt.xlim([-2, 2])
        plt.xlabel('normalized pupil theta'); plt. ylabel('sp/sec'); plt.title(i)
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    # generate summary plot
    samprate = 30000  # ephys sample rate
    plt.figure(figsize = (12,np.ceil(n_units)*2))
    for i, ind in enumerate(goodcells.index): 
        # plot waveform
        plt.subplot(n_units,4,i*4 + 1)
        wv = goodcells.at[ind,'waveform']
        plt.plot(np.arange(len(wv))*1000/samprate,goodcells.at[ind,'waveform'])
        plt.xlabel('msec'); plt.title(str(i) + ' ' + goodcells.at[ind,'KSLabel']  +  ' cont='+ str(goodcells.at[ind,'ContamPct']))
        
        # plot CRF
        plt.subplot(n_units,4,i*4 + 2)
        plt.plot(crange[2:-1],resp[i,2:-1])
        plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec'); plt.ylim([0,np.nanmax(resp[i,2:-1])])
                                    
        #plot STA or tuning curve
        plt.subplot(n_units,4,i*4 + 3)
        if stim_type == 'grat':
            plt.plot(np.arange(8)*45, ori_tuning[i,:,0],label = 'low sf'); plt.plot(np.arange(8)*45,ori_tuning[i,:,1],label = 'mid sf');plt.plot(np.arange(8)*45,ori_tuning[i,:,2],label = 'hi sf');
            plt.plot([0,315],[drift_spont[i],drift_spont[i]],'r:', label = 'spont')
        # plt.legend()
            plt.ylim(0,np.nanmax(ori_tuning[i,:,:]*1.2)); plt.xlabel('orientation (deg)')
        else:
            sta = staAll[i,:,:]
            staRange = np.max(np.abs(sta))*1.2
            if staRange<0.25:
                staRange=0.25
            plt.imshow(staAll[i,:,:],vmin = -staRange, vmax= staRange, cmap = 'jet')
                                  
    #plot eye movements
    plt.subplot(n_units,4,i*4 + 4)
    plt.plot(trange,upsacc_avg[i,:])
    plt.plot(trange,downsacc_avg[i,:],'r')
    plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2),'r')
    plt.ylim([0, np.max(upsacc_avg[i,:])*1.8])
    plt.ylabel('sp/sec')
    plt.tight_layout()
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
