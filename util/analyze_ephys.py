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

def find_files(rec_path, rec_name, free_move, cell):
    # get the files names in the provided path
    eye_file = os.path.join(rec_path, rec_name + '_Reye.nc')
    world_file = os.path.join(rec_path, rec_name + '_world.nc')
    ephys_file = os.path.join(rec_path, rec_name + '_ephys_merge.json')
    imu_file = os.path.join(rec_path, rec_name + '_imu.nc')
    speed_file = os.path.join(rec_path, rec_name + '_speed.nc')

    if free_move is True:
        dict_out = {'cell':cell,'eye':eye_file,'world':world_file,'ephys':ephys_file,'imu':imu_file,'save':rec_path,'name':rec_name}
    elif free_move is False:
        dict_out = {'cell':cell,'eye':eye_file,'world':world_file,'ephys':ephys_file,'speed':speed_file,'save':rec_path,'name':rec_name}

    return dict_out

def ephys_figures(file_dict):

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
    fig, axs = plt.subplots(1,2,figsize=(8,3))
    axs[0].plot(np.diff(worldT)); axs[0].set_xlabel('frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('world cam')
    axs[1].hist(np.diff(worldT),100);axs[1].set_xlabel('deltaT')
    diagnostic_pdf.savefig()
    plt.close()

    # plot mean world image
    plt.figure()
    plt.imshow(np.mean(world_vid,axis=0)); plt.title('mean worldcam')
    diagnostic_pdf.savefig()
    plt.close()

    if file_dict['imu']:
        imu_data = xr.open_dataset(file_dict['imu'])
        accT = imu_data.timestamps
        acc_chans = imu_data['__xarray_dataarray_variable__']
        gx = np.array(acc_chans.sel(channel='gyro_x'))
        gy = np.array(acc_chans.sel(channel='gyro_y'))
        gz = np.array(acc_chans.sel(channel='gyro_z'))

    if file_dict['speed']:
        speed_data = xr.open_dataset(speed_file)
        spdVals = speed_data['__xarray_dataarray_variable__']
        spd = spdVals.sel(move_params = 'cm_per_sec')  # need to check conversion factor *10 ?
        spd_tstamps = spdVals.sel(move_params = 'timestamps')
        plt.plot(spd_tstamps,spd)
        plt.xlabel('sec'); plt.ylabel('running speed cm/sec')

    # read ephys data
    ephys_data = pd.read_json(file_dict['ephys'])

    # select good cells from phy2
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    goodcells.shape
    units = goodcells.index.values

    # get number of good units
    n_units = len(goodcells)

    #spike rasters
    fig, ax = plt.subplots(figsize=(20,8))
    ax.fontsize = 20
    for i,ind in enumerate(goodcells.index):
    plt.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25)
    plt.xlim(0, 10); plt.xlabel('secs',fontsize = 20); plt.ylabel('unit #',fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    detail_pdf.savefig()
    plt.close()

    # load eye data
    eye_data = xr.open_dataset(file_dict['eye'])
    eye_vid = np.uint8(eye_data['REYE_video'])
    eyeT = eye_data.timestamps.copy()

    # plot eye timestamps
    fig, axs = plt.subplots(1,2,figsize=(8,3))
    axs[0].plot(np.diff(eyeT)); axs[0].set_xlabel('frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('eye cam')
    axs[1].hist(np.diff(eyeT),100)
    diagnostic_pdf.savefig()
    plt.close()

    # plot eye postion across recording
    eye_params = eye_data['REYE_ellipse_params']
    plt.plot(eye_params.sel(ellipse_params = 'theta')*180/3.1415,eye_params.sel(ellipse_params = 'phi')*180/3.1415,'.')
    plt.xlabel('theta'); plt.ylabel('phi')
    detail_pdf.savefig()
    plt.close()

    # adjust eye/world/top times relative to ephys
    eyeT = eye_data.timestamps  - ephysT0
    if eyeT[0]<-600:
        eyeT = eyeT + 8*60*60 # 8hr offset for some data
    worldT = world_data.timestamps - ephysT0
    if worldT[0]<-600:
        worldT = worldT + 8*60*60
    if free_move:
        accT = imu_data.timestamps-ephysT0
    if free_move==False:
        speedT = spd_tstamps-ephysT0

    # plot eye variables
    fig,axs = plt.subplots(4,1,figsize = (4,8))
    for i,val in enumerate(eye_params.ellipse_params[0:4]):
        axs[i].plot(eyeT,eye_params.sel(ellipse_params = val))
        axs[i].set_ylabel(val.values)
    detail_pdf.savefig()
    plt.close()

    # calculate eye veloctiy
    dEye = np.diff(eye_params.sel(ellipse_params = 'theta'))*180/np.pi

    # normalize world movie and calculate contrast
    std_im = np.std(world_vid,axis=0)
    std_im[std_im<10] = 10
    img_norm = (world_vid-np.mean(world_vid,axis=0))/std_im

    contrast = np.empty(worldT.size)
    for i in range(worldT.size):
        contrast[i] = np.std(img_norm[i,:,:])
    plt.plot(worldT[0:6000],contrast[0:6000])
    plt.xlabel('time')
    plt.ylabel('contrast')
    detail_pdf.savefig()
    plt.close()

    # set up interpolators for eye and world videos
    eyeInterp = interp1d(eyeT,eye_vid,axis=0)
    worldInterp = interp1d(worldT,world_vid,axis=0)

    # make movie and sound
    # unit to plot/record
    this_unit = file_dict['cell']

    # set up figure
    fig = plt.figure(figsize = (8,12))
    gs = fig.add_gridspec(9,4)
    axEye = fig.add_subplot(gs[0:2,0:2])
    axWorld = fig.add_subplot(gs[0:2,2:4])

    axRad = fig.add_subplot(gs[2,:])
    axTheta = fig.add_subplot(gs[3,:])
    axdTheta = fig.add_subplot(gs[4,:])
    axGyro = fig.add_subplot(gs[5,:])
    axContrast = fig.add_subplot(gs[6,:])
    axR = fig.add_subplot(gs[7:9,:])
    #axRad = fig.add_subplot(gs[3,:])

    #timerange and center frame (only)
    tr = [0, 30]
    fr = np.mean(tr) # time for frame
    eyeFr = np.abs(eyeT-fr).argmin(dim = "frame")
    worldFr = np.abs(worldT-fr).argmin(dim = "frame")

    axEye.cla(); axEye.axis('off'); 
    axEye.imshow(eye_vid[eyeFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    #axEye.plot(eye_params.sel(ellipse_params = 'X0')[fr]/2,eye_params.sel(ellipse_params = 'Y0')[fr]/2,'r.')
    #axEye.set_xlim(0,160); axEye.set_ylim(0,120)

    axWorld.cla();  axWorld.axis('off'); 
    axWorld.imshow(world_vid[worldFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")

    # plot contrast
    axContrast.plot(worldT,contrast)
    axContrast.set_xlim(tr[0],tr[1]); axContrast.set_ylim(0,2)
    axContrast.set_ylabel('image contrast')

    #plot radius
    axRad.cla()
    axRad.plot(eyeT,eye_params.sel(ellipse_params = 'longaxis'))
    axRad.set_xlim(tr[0],tr[1]); 
    axRad.set_ylabel('pupil radius'); axRad.set_xlabel('frame #'); axRad.set_ylim(0,40)

    #plot eye position
    axTheta.cla()
    axTheta.plot(eyeT,(eye_params.sel(ellipse_params = 'theta')-np.nanmean(eye_params.sel(ellipse_params = 'theta')))*180/3.14159)
    axTheta.set_xlim(tr[0],tr[1]); 
    axTheta.set_ylabel('theta - deg'); axTheta.set_ylim(-30,30)

    # plot eye velocity
    axdTheta.cla()
    axdTheta.plot(eyeT[0:-1],dEye*60); ax.set_ylabel('dtheta')
    #sacc = np.transpose(np.where(np.abs(dEye)>10))
    #axdTheta.plot(sacc,np.sign(dEye[sacc])*20,'.')
    axdTheta.set_xlim(tr[0],tr[1]); 
    axdTheta.set_ylim(-900,900); axdTheta.set_ylabel('eye vel - deg/sec')

    # plot gyro
    if file_dict['imu']:
        axGyro.plot(accT,gz)
        axGyro.set_xlim(tr[0],tr[1]); axGyro.set_ylim(0,5)
        axGyro.set_ylabel('gyro V')

    if file_dict['speed']:
        axGyro.plot(speedT,spd)
        axGyro.set_xlim(tr[0],tr[1]); axGyro.set_ylim(0,20)
        axGyro.set_ylabel('speed cm/sec')   
        
    # plot spikes
    axR.fontsize = 20
    for i,ind in enumerate(goodcells.index):
        axR.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25,'k',linewidth=0.5)
    axR.vlines(goodcells.at[units[this_unit],'spikeT'],this_unit-0.25,this_unit+0.25,'b',linewidth=0.5)

    axR.set_xlim(tr[0],tr[1]); axR.set_ylim(-0.5 , n_units); axR.set_xlabel('secs'); axR.set_ylabel('unit #')
    axR.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)

    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    vidfile = name_base + '_' + str(this_unit) + '.mp4'
    # now animate
    writer = FFMpegWriter(fps=30)
    with writer.saving(fig, vidfile, 100):
        for t in np.arange(tr[0],tr[1],1/30):
            
            # show eye and world frames
            axEye.cla(); axEye.axis('off'); 
            axEye.imshow(eyeInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            #axEye.set_xlim(0,160); axEye.set_ylim(0,120)
            
            axWorld.cla(); axWorld.axis('off'); 
            axWorld.imshow(worldInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            
            #plot line for time, then remove
            ln = axR.vlines(t,-0.5,30,'b')
            writer.grab_frame()
            ln.remove()

    # generate wave file
    sp =np.array(ephys_data.at[units[this_unit],'spikeT'])-tr[0]
    sp = sp[sp>0]
    datarate = 30000

    # compute waveform samples
    tmax = tr[1]-tr[0]
    t = np.linspace(0, tr[1]-tr[0], (tr[1]-tr[0])*datarate,endpoint=False)
    x = np.zeros(np.size(t))
    for spt in sp[sp<tmax]:
        x[np.int64(spt*datarate) : np.int64(spt*datarate +30)] = 1
        x[np.int64(spt*datarate)+31 : np.int64(spt*datarate +60)] =- 1
    plt.plot(x)
    
    # Write the samples to a file
    audfile = name_base + '_' + str(this_unit) + '.wav'
    wavio.write(audfile, x, datarate, sampwidth=1)
    
    # merge video and audio
    subprocess.call(['ffmpeg', '-i', vidfile, '-i', audfile, '-c:v', 'copy', '-c:a', 'aac', vidfile[0:-4] + '_merge.mp4']) 

    if free_move:
        plt.plot(eyeT[0:-1],np.diff(th_switch),label = 'dTheta')
        plt.plot(accT-0.1,(gz-3)*10, label = 'gyro')
        plt.xlim(30:40); plt.ylim(-12,12); plt.legend(); plt.xlabel('secs')
        diagnostic_pdf.savefig()
        plt.close()

    # set up timebase for subsequent analysis
    dt = 0.025
    t = np.arange(0, np.max(worldT),dt)

    # interpolate and plot contrast
    newc =interp1d(worldT,contrast)
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
    lag = 0.125;
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
    spike_corr = 1 + 0.125/1200  # correction factor for ephys timing drift
    dEye= np.diff(th_switch)

    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
    trange = np.arange(-1,1.1,0.1)
    sthresh = 5;
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
                                    
        #plot STA
        plt.subplot(n_units,4,i*4 + 3)
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

    overview_pdf.close(); detail_pdf.close(); diagnostic_pdf.close()
