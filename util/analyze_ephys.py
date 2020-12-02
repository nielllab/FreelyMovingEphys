"""
analyze_ephys.py

camera and ephys figures

Dec. 02, 2020
"""
# package imports
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import cv2
import pickle
import time
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl 
import wavio
mpl.rcParams['animation.ffmpeg_path'] = r'C:/Program Files/ffmpeg/bin/ffmpeg.exe'
from scipy.interpolate import interp1d
from numpy import nan
import matplotlib.backends.backend_pdf
# module imports
from util.paths import find

def plot_worldcam_timing(worldT):
    fig, axs = plt.subplots(1,2)
    axs[0].plot(np.diff(worldT)); axs[0].set_xlabel('frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('world cam')
    axs[1].hist(np.diff(worldT),100);axs[1].set_xlabel('deltaT')
    return fig

def plot_mean_world_img(world_vid):
    fig, axs = plt.figure()
    axs = plt.imshow(np.mean(world_vid,axis=0))
    axs = plt.title('mean world image')
    return fig

def plot_eye_frame(eye_vid):
    fig, axs = plt.figure()
    axs = plt.imshow(eye_vid[0,:,:])
    axs = plt.title('eye cam, frame=0')
    return fig

def plot_eyecam_timing(eyeT):
    fig, axs = plt.subplots(1,2)
    axs[0].plot(np.diff(eyeT)); axs[0].set_xlabel('frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('eye cam')
    axs[1].hist(np.diff(eyeT),100)
    return fig

def plot_eye_position(eye_params):
    fig, axs = plt.figure()
    axs = plt.plot(eye_params.sel(ellipse_params = 'theta')*180/3.1415,eye_params.sel(ellipse_params = 'phi')*180/3.1415,'.')
    axs = plt.xlabel('theta'); axs = plt.ylabel('phi')
    return fig

def plot_eye_variables(eye_params, eyeT):
    fig, axs = plt.subplots(4,1)
    for i,val in enumerate(eye_params.ellipse_params[0:4]):
        axs[i].plot(eyeT,eye_params.sel(ellipse_params = val))
        axs[i].set_ylabel(val.values)
    return fig

def plot_sound_video(eye_vid, eyeT, world_vid, worldT, eyeInterp, worldInterp, goodcells, ephys_data, trial_name, datarate):
    #unit to plot/record
    this_unit = 0

    #set up figure
    fig = plt.figure(figsize = (6,8))
    gs = fig.add_gridspec(4,2)
    axEye = fig.add_subplot(gs[0,0])
    axWorld = fig.add_subplot(gs[0,1])
    axTheta = fig.add_subplot(gs[1,:])
    axdTheta = fig.add_subplot(gs[2,:])
    axR = fig.add_subplot(gs[3,:])
    #axRad = fig.add_subplot(gs[3,:])

    #timerange and center frame (only)
    tr = [5, 35]
    fr = np.mean(tr) # time for frame
    eyeFr = np.abs(eyeT-fr).argmin(dim = "frame")
    worldFr = np.abs(worldT-fr).argmin(dim = "frame")

    axEye.cla(); axEye.axis('off'); 
    axEye.imshow(eye_vid[eyeFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    #axEye.plot(eye_params.sel(ellipse_params = 'X0')[fr]/2,eye_params.sel(ellipse_params = 'Y0')[fr]/2,'r.')
    axEye.set_xlim(0,160); axEye.set_ylim(0,120)

    axWorld.cla(); axWorld.axis('off'); 
    axWorld.imshow(np.flipud(world_vid[worldFr,:,:]),'gray',vmin=0,vmax=255,aspect = "equal")
   
    #plot eye position
    axTheta.cla()
    axTheta.plot(eyeT,0.25*eye_params.sel(ellipse_params = 'theta')*180/3.14159)
    axTheta.set_xlim(tr[0],tr[1]); 
    axTheta.set_ylabel('theta - deg'); axTheta.set_ylim(-45,45)

    # plot eye velocity
    axdTheta.cla()
    axdTheta.plot(eyeT[0:-1],0.25*dEye*60); ax.set_ylabel('dtheta')
    #sacc = np.transpose(np.where(np.abs(dEye)>10))
    #axdTheta.plot(sacc,np.sign(dEye[sacc])*20,'.')
    axdTheta.set_xlim(tr[0],tr[1]); 
    axdTheta.set_ylim(-23*60,30*60); axdTheta.set_ylabel('eye vel - deg/sec')

    #plot radius?
    #axRad.cla()
    #axRad.plot(eye_params.sel(ellipse_params = 'longaxis')[frameRange])
    #axRad.set_xlim(0,frameRange[-1]-frameRange[0]); 
    #axRad.set_ylabel('radius'); axRad.set_xlabel('frame #'); axRad.set_ylim(0,40)

    # plot spikes
    axR.fontsize = 20
    for i,ind in enumerate(goodcells.index):
        axR.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25,'k',linewidth=0.5)
    axR.vlines(goodcells.at[units[this_unit],'spikeT'],this_unit-0.25,this_unit+0.25,'b',linewidth=0.5)

    axR.set_xlim(tr[0],tr[1]); axR.set_ylim(-0.5 , 5); axR.set_xlabel('secs'); axR.set_ylabel('unit #')
    axR.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)

    plt.tight_layout()

    # now animate
    writer = FFMpegWriter(fps=30)
    with writer.saving(fig, os.path.join(trial_path, (trial_name + '_ephys_plot.mp4')), 100):
        for t in np.arange(tr[0],tr[1],1/30):
            
            # show eye and world frames
            axEye.cla(); axEye.axis('off'); 
            axEye.imshow(eyeInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            axEye.set_xlim(0,160); axEye.set_ylim(0,120)
            
            axWorld.cla(); axWorld.axis('off'); 
            axWorld.imshow(np.flipud(worldInterp(t)),'gray',vmin=0,vmax=255,aspect = "equal")
            
            #plot line for time, then remove
            ln = axR.vlines(t,-0.5,30,'b')
            writer.grab_frame()
            ln.remove()

    # generate wave file
    sp = np.array(ephys_data.at[units[this_unit],'spikeT'])-tr[0]
    sp = sp[sp>0]
    f = 440.0 # sound frequency (Hz)
    # compute waveform samples
    tmax = tr[1]-tr[0]
    t = np.linspace(0, tr[1]-tr[0], (tr[1]-tr[0])*datarate,endpoint=False)
    x = np.zeros(np.size(t))
    for spt in sp[sp<tmax]:
        x[np.int64(spt*datarate) : np.int64(spt*datarate +30)] = 1
        x[np.int64(spt*datarate)+31 : np.int64(spt*datarate +60)] =- 1
    plt.plot(x)

    # write the samples to a file
    wavio.write(os.path.join(trial_path, (trial_name + '_ephys_plot.wav')), x, datarate, sampwidth=1)

def merge_sound_video(trial_path, trial_name):
    wav_path = os.path.join(trial_path, (trial_name + '_ephys_plot.wav'))
    mp4_path = os.path.join(trial_path, (trial_name + '_ephys_plot.mp4'))
    save_path = os.path.join(trial_path, (trial_name + '_ephys_plot_merged.mp4'))
    subprocess.call(['ffmpeg', '-i', mp4_path, '-i', wav_path, '-c:v', 'copy', '-c:a', 'aac', save_path])

def plot_norm_world(std_im, world_vid):
    fig, axs = plt.subplots(1,2)
    axs[0] = plt.imshow(std_im)
    axs[0] = plt.title('std dev of image')
    axs[1] = plt.imshow(np.mean(world_vid,axis=0),vmin=0,vmax=255)
    axs[1] = plt.title('mean of image')
    return fig

def plot_world_contrast(worldT, contrast):
    fig, axs = plt.figure()
    axs = plt.plot(worldT[0:6000],contrast[0:6000])
    axs = plt.xlabel('time')
    axs = plt.ylabel('contrast')
    return fig

def plot_interp_contrast(contrast_interp):
    fig, axs = plt.figure()
    axs = plt.plot(contrast_interp[0:600])
    return fig

def plot_firing_rate(goodcells):
    fig, axs = plt.subplots(7,2)
    for i, ind in enumerate(goodcells.index):
        ax[i+1] = plt.plot(t[0:-1],goodcells.at[ind,'rate'])
    return fig

def plot_scatter_vs_contrast(goodcells):
    fig, axs = plt.subplots(12,4)
    for i, ind in enumerate(goodcells.index):
        axs[i+1] = plt.plot(contrast_interp,goodcells.at[ind,'rate'],'.')
    axs = plt.xlabel('contrast')
    axs = plt.ylabel('rate')
    return fig

def plot_contrast_response(goodcells):
    resp = np.empty((6,12))
    crange = np.arange(0,1.2,0.1)
    for i,ind in enumerate(goodcells.index):
        for c,cont in enumerate(crange):
            resp[i,c] = np.mean(goodcells.at[ind,'rate'][(contrast_interp>cont) & (contrast_interp<(cont+0.1))])
    fig, axs = plt.figure(()
    axs = plt.plot(crange,np.transpose(resp))
    axs = plt.xlabel('contrast')
    axs = plt.ylabel('sp/sec')
    return fig

def plot_ind_contrast_response(goodcells)
    n_units = len(goodcells)
    fig = plt.figure(figsize = (3*np.ceil(n_units/2),6))
    for i, ind in enumerate(goodcells.index):
        plt.subplot(2,np.ceil(n_units/2),i+1)
        plt.plot(crange[2:-1],resp[i,2:-1])
        plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec')
        plt.ylim([0,np.nanmax(resp[i,2:-1])])
    plt.tight_layout()
    return fig

def plot_STA(goodcells):
    sta = 0
    lag = 0.075
    fig = plt.figure()
    for c, ind in enumerate(goodcells.index):
        r = goodcells.at[ind,'rate']
        sta = 0
        for i in range(100,t.size-100):
            if r[i]>0:
                sta = sta+r[i]*(movInterp(t[i]-lag))
        plt.subplot(3,5,c+1)
        sta = sta/np.sum(r)
        #sta[abs(sta)<0.1]=0
        plt.imshow((sta-np.mean(sta) ),vmin=-0.7,vmax=0.7,cmap = 'jet')
    plt.tight_layout()
    return fig

def plot_STA1(good_cells):
    spike_corr = 1 + 0.125/1200
    sta = 0
    lag = 0.075
    lagRange = np.arange(0,0.25,0.05)
    fig = plt.figure()
    for c, ind in enumerate(goodcells.index):
        sp = goodcells.at[ind,'spikeT'].copy()
        for  lagInd, lag in enumerate(lagRange):
            sta = 0; nsp = 0
            for s in sp:
                if (s-lag >5) & ((s-lag)*spike_corr <np.max(worldT)):
                    nsp = nsp+1
                    sta = sta+movInterp((s-lag)*spike_corr)
            plt.subplot(6,6,(c*6)+lagInd + 1)
            sta = sta/nsp
        #sta[abs(sta)<0.1]=0
            plt.imshow((sta-np.mean(sta) ),vmin=-0.35,vmax=0.35,cmap = 'jet')
    plt.tight_layout()
    return fig

def plot_STV(goodcells):
    sta = 0
    lag = 0.1
    fig = plt.figure()
    for c, ind in enumerate(goodcells.index):
        r = goodcells.at[ind,'rate']
        sta = 0
        for i in range(5,t.size-10):
            sta = sta+r[i]*(movInterp(t[i]-lag))**2
        plt.subplot(3,5,c+1)
        sta = sta/np.sum(r)
        plt.imshow(sta - np.mean(img_norm**2,axis=0),vmin=-1,vmax=1)
    plt.tight_layout()
    return fig

def plot_saccade(units, eyeT, dEye, goodcells, t, s):
    fig, axs = plt.subplot(3,5)
    trange = np.arange(-1,1.1,0.1)
    sthresh = 6;
    upsacc = eyeT[np.append(dEye,0)>sthresh]
    upsacc = upsacc[upsacc>5]
    downsacc= eyeT[np.append(dEye,0)<-sthresh]
    downsacc = downsacc[downsacc>5]
    upsacc_avg = np.zeros((units.size,trange.size))
    downsacc_avg = np.zeros((units.size,trange.size))
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'])
        for s in upsacc:
            upsacc_avg[i,:] = upsacc_avg[i,:]+ rateInterp(np.array(s)+trange)/upsacc.size
        for s in downsacc:
            downsacc_avg[i,:]= downsacc_avg[i,:]+ rateInterp(np.array(s)+trange)/downsacc.size
        axs[i+1] = plt.plot(trange,upsacc_avg[i,:])
        axs[i+1] = plt.plot(trange,downsacc_avg[i,:],'r')
        axs[i+1] = plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2),'r')
        axs[i+1] = plt.ylim([0, np.max(upsacc_avg[i,:])*1.8])
        axs[i+1] = plt.ylabel('sp/sec')
    plt.tight_layout()
    return fig

def plot_upsacc(goodcells):
    fig = plt.figure()
    for i, ind in enumerate(goodcells.index):
        sp = np.array(goodcells.at[units[i],'spikeT'])
        plt.subplot(2,3,i+1)
        n = 0
        for s in upsacc:
            n= n+1
            sd = np.abs(sp-np.array(s))<10
            sacc_sp = sp[sd]
            plt.vlines(sacc_sp-np.array(s),n-0.25,n+0.25)
        plt.xlim(-1,1)
    return fig

def plot_downsacc(goodcells)
    fig = plt.figure()
    for i, ind in enumerate(goodcells.index):
        sp = np.array(goodcells.at[units[i],'spikeT'])
        plt.subplot(2,3,i+1)
        n = 0
        for s in downsacc:
            n= n+1
            sd = np.abs(sp-np.array(s))<10
            sacc_sp = sp[sd]
            plt.vlines(sacc_sp-np.array(s),n-0.25,n+0.25)
        plt.xlim(-1,1)
    return fig

def plot_norm_eye_rads():
    fig = plt.plot(eyeT,Rnorm)
    plt.xlim([0,60])
    plt.xlabel('secs')
    plt.ylabel('normalized pupil R')
    return fig

def plot_rate_vs_pupil(goodcells, eyeT, t, Rnorm):
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
    fig = plt.figure(figsize = (3*np.ceil(n_units/2),6))
    for i in range(n_units):
        plt.subplot(2,np.ceil(n_units/2),i+1)
        plt.errorbar(R_range[:-1],R_tuning[i,:],yerr=R_tuning_err[i,:])
        plt.ylim(0,np.nanmax(R_tuning[i,:]*1.2))
        plt.xlim([-2, 3])
        plt.xlabel('normalized pupil R'); plt. ylabel('sp/sec'); plt.title(i)  
    plt.tight_layout()
    return fig

def plot_norm_eye_theta(eyeT, thetaNorm):
    fig = plt.plot(eyeT[0:3600],thetaNorm[0:3600])
    plt.xlabel('secs'); plt.ylabel('normalized eye theta')
    return fig

def plot_rate_vs_theta(goodcells, eyeT, t, thetaNorm)
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
    return fig

# headfixed figures
def headfixed_figures(config):

    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(config['trial_path'], (config['recording_name'] + '_figures.pdf')))

    eye_file = find(config['recording_name'] + '*eye.nc', config['trial_path'])
    world_file = find(config['recording_name'] + '*world.nc', config['trial_path'])
    ephys_file = find(config['recording_name'] + '*ephys_merge.json', config['trial_path'])

    world_data = xr.open_dataset(world_file)
    world_vid_raw = np.uint8(world_data['WORLD_video'])

    # resize worldcam to make more manageable
    sz = world_vid_raw.shape
    downsamp = 0.5
    world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
    for f in range(sz[0]):
        world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
    worldT = world_data.timestamps.copy()

    # plot worldcam timing
    pdf.savefig(plot_worldcam_timing(worldT))

    # plot mean world image
    pdf.savefig(plot_mean_world_img(world_vid))

    # read ephys data
    ephys_data = pd.read_json(ephys_file)

    # get first ephys timestamp
    ephysT0 = ephys_data.iloc[0,12]
    # select good cells
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    units = goodcells.index.values

    #load eye data
    eye_data = xr.open_dataset(eye_file)
    eye_vid = np.uint8(eye_data['REYE_video'])
    eyeT = eye_data.timestamps.copy()

    # plot first frame of eyecam
    pdf.savefig(plot_eye_frame(eye_vid))
    
    # plot eye timestamps
    pdf.savefig(plot_eyecam_timing(eyeT))

    # plot eye postion across recording
    eye_params = eye_data['REYE_ellipse_params']
    pdf.savefig(plot_eye_postion(eye_params))

    # adjust eye/world/top times relative to ephys
    eyeT = eye_data.timestamps  - ephysT0
    if eyeT[0]<-600:
        eyeT = eyeT + 8*60*60 # 8hr offset for some data
    worldT = world_data.timestamps - ephysT0
    if worldT[0]<-600:
        worldT = worldT + 8*60*60
    
    # plot eye variables
    pdf.savefig(plot_eye_variables(eye_params, eyeT))

    # calculate eye veloctiy
    dEye = np.diff(eye_params.sel(ellipse_params = 'theta'))*180/3.14159

    # normalize world movie and calculate contrast
    img_norm = (world_vid-np.mean(world_vid,axis=0))/np.std(world_vid,axis=0)
    contrast = np.empty(worldT.size)
    for i in range(worldT.size):
        contrast[i] = np.std(img_norm[i,:,:])
    pdf.savefig(plot_world_contrast(worldT, contrast))

    # set up interpolators for eye and world videos
    eyeInterp = interp1d(eyeT,eye_vid,axis=0)
    worldInterp = interp1d(worldT,world_vid,axis=0)

    # write .mp4 and .wav files
    plot_sound_video(eye_vid, eyeT, world_vid, worldT, eyeInterp, worldInterp, goodcells, ephys_data, trial_name, datarate)
    # then merge the two files into an mp4 with sound
    merge_sound_video(trial_path, trial_name)

    # normalize world video
    std_im = np.std(world_vid,axis=0)
    std_im[std_im<10] = 10
    img_norm = (world_vid-np.mean(world_vid,axis=0))/std_im
    # and plot
    pdf.savefig(plot_norm_world(std_im, world_vid))

    #set up timebase for subsequent analysis
    dt = 0.025
    t = np.arange(0, np.max(worldT),dt)

    # interpolate and plot contrast
    newc =interp1d(worldT,contrast)
    contrast_interp = newc(t[0:-1])
    pdf.savefig(plot_interp_contrast(contrast_interp[0:600]))

    # calculate firing rate at new timebase
    ephys_data['rate'] = nan
    ephys_data['rate'] = ephys_data['rate'].astype(object)
    for i,ind in enumerate(ephys_data.index):
        ephys_data.at[ind,'rate'],bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
    ephys_data['rate']= ephys_data['rate']/dt
    goodcells = ephys_data.loc[ephys_data['group']=='good']

    # plot firing rates
    pdf.savefig(plot_firing_rate(goodcells))

    #scatter of contrast vs rate
    pdf.savefig(plot_scatter_vs_contrast(goodcells))

    # calculate contrast - response functions
    # mean firing rate in timebins correponding to contrast ranges
    pdf.savefig(plot_contrast_response(goodcells))

    # plot individual contrast response functions in subplots
    pdf.savefig(plot_ind_contrast_response())

    # create interpolator for movie data so we can evaluate at same timebins are firing rat
    img_norm[img_norm<-2] = -2
    movInterp = interp1d(worldT,img_norm,axis=0)

    # calculate spike-triggered average
    pdf.savefig(plot_STA(goodcells))
    pdf.savefig(plot_STA1(goodcells))

    # calculate spike-triggered variance
    pdf.savefig(plot_SPV(goodcells))

    # calculate saccade-locked psth
    pdf.savefig(plot_saccade(units, eyeT, dEye, goodcells, t, s))

    # plot upsacc, downsac
    pdf.savefig(plot_upsacc(goodcells))
    pdf.savefig(plot_downsacc(goodcells))

    # normalize and plot eye radius
    eyeR = eye_params.sel(ellipse_params = 'longaxis').copy()
    Rnorm = (eyeR - np.mean(eyeR))/np.std(eyeR)
    pdf.savefig(plot_norm_eye_rads(eye, Rnorm))
    pdf.savefig(plt.hist(Rnorm))

    # plot rate vs pupil
    pdf.savefig(plot_rate_vs_pupil(goodcells, eyeT, t, Rnorm))

    eyeTheta = eye_params.sel(ellipse_params = 'theta').copy()
    thetaNorm = (eyeTheta - np.mean(eyeTheta))/np.std(eyeTheta)
    pdf.savefig(plot_nor_eye_theta(eyeT, thetaNorm))

    # plot rate vs theta
    pdf.savefig(plot_rate_vs_theta(goodcells, eyeT, t, thetaNorm))

    print('figures saved')

    pdf.close()

def freely_moving_figures(config):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(config['trial_path'], (config['recording_name'] + '_figures.pdf')))

    eye_file = find(config['recording_name'] + '*eye.nc', config['trial_path'])
    world_file = find(config['recording_name'] + '*world.nc', config['trial_path'])
    ephys_file = find(config['recording_name'] + '*ephys_merge.json', config['trial_path'])
    top_file = find(config['recording_name'] + '*TOP1.nc', config['trial_path'])

    name_base = ephys_file[0:-5]

    top_data = xr.open_dataset(top_file)
    top_vid = np.uint8(top_data['TOP1_video'])
    topT = top_data['timestamps']

    # get points from top cam to calculate locomotion
    # base of spine seems most reliable
    ptNames = top_data['TOP1_pts']
    spinex = ptNames.sel(point_loc = 'spine_x').copy()
    spiney = ptNames.sel(point_loc = 'spine_y').copy()
    spinep = ptNames.sel(point_loc = 'spine_likelihood').copy()

    #smooth pts before calculating speed
    box = np.ones(31)/31
    spinex_sm = np.convolve(spinex, box, mode='same')
    spiney_sm = np.convolve(spiney,box,mode = 'same')

    #speed
    spd = np.sqrt(np.diff(spinex_sm)**2 + np.diff(spiney_sm)**2)*60/10
    spd[0:10]=0

    pdf.savefig(plt.plot(spd[0:1800])
    pdf.savefig(plot_speed(spinex, spinex_sm))

    # load worldcam
    world_data = xr.open_dataset(world_file)
    world_vid_raw = np.uint8(world_data['WORLD_video'])

    # resize worldcam to make more manageable
    sz = world_vid_raw.shape
    downsamp = 0.25
    world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
    for f in range(sz[0]):
        world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
    worldT = world_data.timestamps.copy()

    # plot worldcam timing
    pdf.savefig(plot_worldcam_timing(worldT))

    # plot mean world image
    pdf.savefig(plot_mean_world_img(world_vid))

    # read ephys data
    ephys_data = pd.read_json(ephys_file)
    ephysT0 = ephys_data.iloc[0,12]

    # select good cells from phy2
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    units = goodcells.index.values

    # load eye data
    eye_data = xr.open_dataset(eye_file)
    eye_vid = np.uint8(eye_data['REYE_video'])
    pdf.savefig(plot_eye_frame(eye_vid))
    eyeT = eye_data.timestamps.copy()

    # plot eye timestamps
    pdf.savefig(plot_eyecam_timing(eyeT))

    #plot eye postion across recording
    eye_params = eye_data['REYE_ellipse_params']
    pdf.savefig(plot_eye_position(eye_params))

    # adjust eye/world/top times relative to ephys
    eyeT = eye_data.timestamps  - ephysT0
    if eyeT[0]<-600:
        eyeT = eyeT + 8*60*60 # 8hr offset for some data
    worldT = world_data.timestamps - ephysT0
    if worldT[0]<-600:
        worldT = worldT + 8*60*60
    topT = top_data.timestamps - ephysT0
    
    pdf.savefig(plot_eye_variables(eye_params, eyeT))

    # calculate eye veloctiy
    dEye = np.diff(eye_params.sel(ellipse_params = 'theta'))*180/3.14159
    eyeR = eye_params.sel(ellipse_params = 'longaxis').copy()

def plot_speed(spinex, spinex_sm):
    fig = plt.figure()
    plt.plot(spinex[100:200])
    plt.plot(spinex_sm[100:200])
    return fig

