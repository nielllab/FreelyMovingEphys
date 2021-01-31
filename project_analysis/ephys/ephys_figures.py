"""
ephys_figures.py

make individual ephys figures

Jan. 12, 2021
"""
# package imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import cv2
import pickle
import time
import subprocess
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl 
import wavio
from scipy.interpolate import interp1d
from numpy import nan
from matplotlib.backends.backend_pdf import PdfPages
import os

# diagnostic figure of camera timing
def plot_cam_time(camT, camname):
    fig, axs = plt.subplots(1,2)
    axs[0].plot(np.diff(camT)); axs[0].set_xlabel('frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title(camname+' cam timing')
    axs[1].hist(np.diff(camT),100);axs[1].set_xlabel('deltaT')
    return fig

# spike raster of ephys data
def plot_spike_rasters(goodcells):
    fig, ax = plt.subplots()
    ax.fontsize = 20
    for i,ind in enumerate(goodcells.index):
        plt.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25)
        plt.xlim(0, 10); plt.xlabel('secs',fontsize = 20); plt.ylabel('unit #',fontsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    return fig

# plot theta vs phi across recording
def plot_eye_pos(eye_params):
    fig = plt.figure()
    plt.plot(np.rad2deg(eye_params.sel(ellipse_params='theta')),np.rad2deg(eye_params.sel(ellipse_params='phi')),'.')
    plt.xlabel('theta'); plt.ylabel('phi'); plt.title('eye position accross recording')
    return fig

# optical mouse speed
def plot_optmouse_spd(spd_tstamps, spd):
    fig = plt.figure()
    plt.plot(spd_tstamps,spd)
    plt.xlabel('sec'); plt.ylabel('running speed cm/sec')
    return fig

# check that deinterlacing worked well
# plot theta and theta switch
# want theta switch to be jagged, theta to be smooth
def plot_param_switch_check(eye_params):
    th = np.array(np.rad2deg(eye_params.sel(ellipse_params = 'theta')-np.nanmean(eye_params.sel(ellipse_params = 'theta'))))
    phi = np.array(np.rad2deg(eye_params.sel(ellipse_params = 'phi')-np.nanmean(eye_params.sel(ellipse_params = 'phi'))))
    th_switch = np.zeros(np.shape(th))
    th_switch[0:-1:2] = np.array(th[1::2])
    th_switch[1::2] = np.array(th[0:-1:2])
    fig, ax = plt.subplots(121)
    plt.subplot(1,2,1)
    plt.plot(th[(35*60):(40*60)]); plt.title('theta')
    plt.subplot(1,2,2)
    plt.plot(th_switch[(35*60):(40*60)]); plt.title('theta switch')
    return fig, th_switch

# plot the four main paramters of the eye tracking
def plot_eye_params(eye_params, eyeT):
    fig, axs = plt.subplots(4,1)
    for i,val in enumerate(eye_params.ellipse_params[0:4]):
        axs[i].plot(eyeT,eye_params.sel(ellipse_params = val))
        axs[i].set_ylabel(val.values)
    plt.tight_layout()
    return fig

def make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, accT=None, gz=None, speedT=None, spd=None):
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
    axdTheta.plot(eyeT[0:-1],dEye*60); axdTheta.set_ylabel('dtheta')
    #sacc = np.transpose(np.where(np.abs(dEye)>10))
    #axdTheta.plot(sacc,np.sign(dEye[sacc])*20,'.')
    axdTheta.set_xlim(tr[0],tr[1]); 
    axdTheta.set_ylim(-900,900); axdTheta.set_ylabel('eye vel - deg/sec')

    # plot gyro
    if file_dict['imu'] is not None:
        axGyro.plot(accT,gz)
        axGyro.set_xlim(tr[0],tr[1]); axGyro.set_ylim(0,5)
        axGyro.set_ylabel('gyro V')

    if file_dict['speed'] is not None:
        axGyro.plot(speedT,spd)
        axGyro.set_xlim(tr[0],tr[1]); axGyro.set_ylim(0,20)
        axGyro.set_ylabel('speed cm/sec')   
        
    # plot spikes
    axR.fontsize = 20
    for i,ind in enumerate(goodcells.index):
        axR.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25,'k',linewidth=0.5)
    axR.vlines(goodcells.at[units[this_unit],'spikeT'],this_unit-0.25,this_unit+0.25,'b',linewidth=0.5)

    n_units = len(goodcells)

    axR.set_xlim(tr[0],tr[1]); axR.set_ylim(-0.5 , n_units); axR.set_xlabel('secs'); axR.set_ylabel('unit #')
    axR.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)

    vidfile = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'.mp4'))
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

    return vidfile

def make_sound(file_dict, ephys_data, units, this_unit):
    tr = [0, 30]
    # generate wave file
    sp = np.array(ephys_data.at[units[this_unit],'spikeT'])-tr[0]
    sp = sp[sp>0]
    datarate = 30000

    # compute waveform samples
    tmax = tr[1]-tr[0]
    t = np.linspace(0, tr[1]-tr[0], (tr[1]-tr[0])*datarate,endpoint=False)
    x = np.zeros(np.size(t))
    for spt in sp[sp<tmax]:
        x[np.int64(spt*datarate) : np.int64(spt*datarate +30)] = 1
        x[np.int64(spt*datarate)+31 : np.int64(spt*datarate +60)] =- 1
    
    # Write the samples to a file
    audfile = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'.wav'))
    wavio.write(audfile, x, datarate, sampwidth=1)

    return audfile

def plot_acc_eyetime_alignment(eyeT, t1, offset, ccmax):
    fig = plt.subplot(1,2,1)
    plt.plot(eyeT[t1*60],offset)
    plt.xlabel('secs'); plt.ylabel('offset - secs')
    plt.subplot(1,2,2)
    plt.plot(eyeT[t1*60],ccmax)
    plt.xlabel('secs'); plt.ylabel('max cc')
    return fig

def plot_regression_timing_fit(dataT, offset, offset0, drift_rate):
    fig = plt.figure()
    plt.plot(dataT,offset,'.')
    plt.plot(dataT, offset0 + dataT*drift_rate)
    plt.xlabel('secs'); plt.ylabel('offset - secs')
    plt.title('offset0='+str(offset0)+' drift_rate='+str(drift_rate))
    return fig

def plot_saccade_and_fixate(eyeT, dEye, gInterp, th):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.plot(eyeT[0:-1],dEye, label = 'dEye')
    plt.plot(eyeT, gInterp(eyeT), label = 'dHead')
    plt.xlim(37,39); plt.ylim(-10,10); plt.legend(); plt.ylabel('deg'); plt.xlabel('secs')
    plt.subplot(1,2,2)
    plt.plot(eyeT[0:-1],np.nancumsum(gInterp(eyeT[0:-1])), label = 'head')
    plt.plot(eyeT[0:-1],np.nancumsum(gInterp(eyeT[0:-1])-dEye),label ='gaze')
    plt.plot(eyeT[1:],th[0:-1],label ='eye')
    plt.xlim(35,40); plt.ylim(-30,30); plt.legend(); plt.ylabel('deg'); plt.xlabel('secs')
    plt.tight_layout()
    return fig

def plot_ind_contrast_funcs(n_units, goodcells, crange, resp):
    fig = plt.figure(figsize = (6,np.ceil(n_units/2)))
    for i, ind in enumerate(goodcells.index):
        plt.subplot(np.ceil(n_units/4),4,i+1)
        plt.plot(crange[2:-1],resp[i,2:-1])
    # plt.ylim([0 , max(resp[i,1:-3])*1.2])
        plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec'); plt.ylim([0,np.nanmax(resp[i,2:-1])])
    plt.title('individual contrast reponse')
    plt.tight_layout()
    return fig

def plot_STA_single_lag(n_units, img_norm, goodcells, worldT, movInterp):
    spike_corr = 1
    staAll = np.zeros((n_units,np.shape(img_norm)[1],np.shape(img_norm)[2]))
    lag = 0.075
    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
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
        if nsp > 0:
            sta = sta/nsp
        else:
            sta = np.nan
        plt.imshow((sta-np.mean(sta) ),vmin=-0.3,vmax=0.3,cmap = 'jet')
        staAll[c,:,:] = sta
    plt.tight_layout()
    return staAll, fig

def plot_STA_multi_lag(n_units, goodcells, worldT, movInterp):
    spike_corr = 1; sta = 0; lag = 0.075
    lagRange = np.arange(0,0.25,0.05)
    fig = plt.figure(figsize = (12,2*n_units))
    for c, ind in enumerate(goodcells.index):
        sp = goodcells.at[ind,'spikeT'].copy()
        for  lagInd, lag in enumerate(lagRange):
            sta = 0; nsp = 0
            for s in sp:
                if (s-lag >5) & ((s-lag)*spike_corr <np.max(worldT)):
                    nsp = nsp+1
                    sta = sta+movInterp((s-lag)*spike_corr)
            plt.subplot(n_units,6,(c*6)+lagInd + 1)
            if nsp > 0:
                sta = sta/nsp
            else:
                sta = np.nan
            plt.imshow(sta ,vmin=-0.35,vmax=0.35,cmap = 'jet')
            plt.title(str(c) + ' ' + str(np.round(lag*1000)) + 'msec')
    plt.tight_layout()
    return fig

def plot_spike_triggered_variance(n_units, goodcells, t, movInterp, img_norm):
    sta = 0; lag = 0.125
    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
    for c, ind in enumerate(goodcells.index):
        r = goodcells.at[ind,'rate']
        sta = 0
        for i in range(5,t.size-10):
            sta = sta+r[i]*(movInterp(t[i]-lag))**2
        plt.subplot(np.ceil(n_units/4),4,c+1)
        sta = sta/np.sum(r)
        plt.imshow(sta - np.mean(img_norm**2,axis=0),vmin=-1,vmax=1)
    plt.tight_layout()
    return fig

def plot_saccade_locked(n_units, goodcells, t, upsacc, upsacc_avg, trange, downsacc, downsacc_avg):
    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
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
    return upsacc_avg, downsacc_avg, fig

def plot_rasters_around_saccades(n_units, goodcells, sacc):
    fig = plt.figure(figsize = (12,n_units))
    units = goodcells.index.values
    for i, ind in enumerate(goodcells.index):
        sp = np.array(goodcells.at[units[i],'spikeT'])
        plt.subplot(np.ceil(n_units/4),4,i+1)
        n = 0
        for s in sacc:
            n = n+1
            sd = np.abs(sp-np.array(s))<10
            sacc_sp = sp[sd]
            plt.vlines(sacc_sp-np.array(s),n-0.25,n+0.25)
        plt.xlim(-1,1)
    return fig

def plot_spike_rate_vs_var(n_units, use, var_range, goodcells, useEyeT, t, var_name):
    scatter = np.zeros((n_units,len(use)))
    tuning = np.zeros((n_units,len(var_range)-1))
    tuning_err = tuning.copy()
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'])
        scatter[i,:] = rateInterp(useEyeT)
        for j in range(len(var_range)-1):
            usePts =(use>var_range[j]) & (use<var_range[j+1])
            tuning[i,j] = np.mean(scatter[i,usePts])
            tuning_err[i,j] = np.std(scatter[i,usePts])/np.sqrt(np.count_nonzero(usePts))
    if var_name == 'th':
        fig = plt.figure(figsize = (3*np.ceil(n_units/2),6))
    elif var_name == 'rad':
        fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
    for i in range(n_units):
        if var_name == 'th':
            plt.subplot(np.ceil(n_units/4),4,i+1)
        elif var_name == 'rad':
            plt.subplot(2,np.ceil(n_units/2),i+1)
        plt.errorbar(var_range[:-1],tuning[i,:],yerr=tuning_err[i,:])
        try:
            plt.ylim(0,np.nanmax(tuning[i,:]*1.2))
        except ValueError:
            plt.ylim(0,1)
        plt.xlim([-2, 2])
        if var_name == 'th':
            plt.xlabel('normalized pupil theta'); plt. ylabel('sp/sec'); plt.title(i)
        elif var_name == 'rad':
            plt.xlabel('normalized pupil radius'); plt. ylabel('sp/sec'); plt.title(i)
    plt.tight_layout()
    return fig

def plot_summary(n_units, goodcells, crange, resp, file_dict, staAll, trange, upsacc_avg, downsacc_avg, ori_tuning=None, drift_spont=None):
    samprate = 30000  # ephys sample rate
    fig = plt.figure(figsize = (12,np.ceil(n_units)*2))
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
        if ori_tuning is not None:
            plt.plot(np.arange(8)*45, ori_tuning[i,:,0],label = 'low sf')
            plt.plot(np.arange(8)*45,ori_tuning[i,:,1],label = 'mid sf')
            plt.plot(np.arange(8)*45,ori_tuning[i,:,2],label = 'hi sf')
            plt.plot([0,315],[drift_spont[i],drift_spont[i]],'r:', label = 'spont')
        # plt.legend()
            try:
                plt.ylim(0,np.nanmax(ori_tuning[i,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.xlabel('orientation (deg)')
        else:
            sta = staAll[i,:,:]
            staRange = np.max(np.abs(sta))*1.2
            if staRange<0.25:
                staRange=0.25
            plt.imshow(staAll[i,:,:],vmin = -staRange, vmax= staRange, cmap = 'jet')
                      
        # plot eye movements
        plt.subplot(n_units,4,i*4 + 4)
        plt.plot(trange,upsacc_avg[i,:])
        plt.plot(trange,downsacc_avg[i,:],'r')
        plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2),'r')
        plt.ylim([0, np.max(upsacc_avg[i,:])*1.8])
        plt.ylabel('sp/sec')
    plt.tight_layout()
    return fig