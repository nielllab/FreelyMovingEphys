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
import scipy.sparse as sparse
import scipy.linalg as linalg
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# calculate and plot psth relative to timepoints
def plot_psth(goodcells,onsets,lower,upper,dt,drawfig):
    n_units = len(goodcells)
    bins = np.arange(lower,upper+dt,dt)
    fig = plt.figure(figsize = (10,np.ceil(n_units/2)))
    psth_all = np.zeros((n_units,len(bins)-1))
    for i, ind in enumerate(goodcells.index):
        plt.subplot(int(np.ceil(n_units/4)),4,i+1)
        psth = np.zeros(len(bins)-1)
        for t in onsets:
            hist,edges = np.histogram(goodcells.at[ind,'spikeT']-t,bins)
            psth = psth+hist
        psth = psth/len(onsets)
        psth = psth/dt
        plt.plot(bins[0:-1]+ dt/2,psth)
        plt.ylim(0,np.nanmax(psth)*1.2)
        psth_all[i,:]=psth
    plt.xlabel('time'); plt.ylabel('sp/sec')    
    plt.tight_layout()
    
    if drawfig is False:
        plt.close()
    return psth_all
        
# diagnostic figure of camera timing
def plot_cam_time(camT, camname):
    fig, axs = plt.subplots(1,2)
    axs[0].plot(np.diff(camT)[0:-1:10]); axs[0].set_xlabel('every 10th frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title(camname+' cam timing')
    axs[1].hist(np.diff(camT),100);axs[1].set_xlabel('deltaT')
    return fig

# spike raster of ephys data
def plot_spike_rasters(goodcells):
    fig, ax = plt.subplots()
    ax.fontsize = 20
    for i,ind in enumerate(goodcells.index):
        sp = np.array(goodcells.at[ind,'spikeT'])
        plt.vlines(sp[sp<10],i-0.25,i+0.25)
        plt.xlim(0, 10); plt.xlabel('secs',fontsize = 20); plt.ylabel('unit #',fontsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    return fig

# plot theta vs phi across recording
def plot_eye_pos(eye_params):
    fig = plt.figure()
    th = np.array(eye_params.sel(ellipse_params='theta'))
    plt.plot(np.rad2deg(th)[0:-1:10],np.rad2deg(eye_params.sel(ellipse_params='phi'))[0:-1:10],'.')
    good_pts = np.sum(~np.isnan(th))/len(th)
    plt.xlabel('theta'); plt.ylabel('phi'); plt.title(f'eye position fraction good = {good_pts:.3}')
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
        axs[i].plot(eyeT[0:-1:10],eye_params.sel(ellipse_params = val)[0:-1:10])
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

def plot_trace_summary(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, top_speed, topT, tr = [15,45], accT=None, gz=None, speedT=None, spd=None):
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
    fr = np.mean(tr) # time for frame
    eyeFr = np.abs(eyeT-fr).argmin(dim = "frame")
    worldFr = np.abs(worldT-fr).argmin(dim = "frame")

    axEye.cla(); axEye.axis('off'); 
    axEye.imshow(eye_vid[eyeFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")

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
    axRad.set_ylabel('pupil radius'); axRad.set_xlabel('frame #'); axRad.set_ylim(0,50)

    #plot eye position
    axTheta.cla()
    axTheta.plot(eyeT,(eye_params.sel(ellipse_params = 'theta')-np.nanmean(eye_params.sel(ellipse_params = 'theta')))*180/3.14159)
    axTheta.set_xlim(tr[0],tr[1])
    axTheta.set_ylabel('theta (deg)'); axTheta.set_ylim(-30,30)

    # plot eye velocity
    axdTheta.cla()
    axdTheta.plot(topT,top_speed); axdTheta.set_ylabel('topdown speed')
    axdTheta.set_xlim(tr[0],tr[1])

    # plot gyro
    if file_dict['imu'] is not None:
        axGyro.plot(accT,gz)
        axGyro.set_xlim(tr[0],tr[1]); axGyro.set_ylim(0,5)
        axGyro.set_ylabel('gyro z velocity')

    if file_dict['speed'] is not None:
        axGyro.plot(speedT,spd)
        axGyro.set_xlim(tr[0],tr[1]); axGyro.set_ylim(0,20)
        axGyro.set_ylabel('ball speed cm/sec')   
        
    # plot spikes
    axR.fontsize = 20
    for i,ind in enumerate(goodcells.index):
        axR.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25,'k',linewidth=0.5)
    axR.vlines(goodcells.at[units[this_unit],'spikeT'],this_unit-0.25,this_unit+0.25,'b',linewidth=0.5)
    n_units = len(goodcells)
    axR.set_xlim(tr[0],tr[1]); axR.set_ylim(-0.5 , n_units); axR.set_xlabel('secs'); axR.set_ylabel('unit #')
    axR.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)

    return fig

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
    plt.plot(eyeT[0:-1],np.nancumsum(gInterp(eyeT[0:-1])+dEye),label ='gaze')
    plt.plot(eyeT[1:],th[0:-1],label ='eye')
    plt.xlim(35,40); plt.ylim(-30,30); plt.legend(); plt.ylabel('deg'); plt.xlabel('secs')
    plt.tight_layout()
    return fig

def eye_shift_estimation(th,phi, eyeT, world_vid,worldT,max_frames):
    
    # get eye displacement for each worldcam frame
    th_interp = interp1d(eyeT,th,bounds_error = False)
    phi_interp = interp1d(eyeT, phi, bounds_error = False)
    dth = np.diff(th_interp(worldT))
    dphi = np.diff(phi_interp(worldT))
    
    # calculate x-y shift for each worldcam frame  
    number_of_iterations = 5000
    termination_eps = 1e-4
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    warp_mode = cv2.MOTION_TRANSLATION
    cc = np.zeros(max_frames); xshift = np.zeros(max_frames); yshift = np.zeros(max_frames);
    warp_all = np.zeros((6,max_frames))
    # get shift between adjacent frames
    for i in tqdm(range(max_frames)):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try: 
            (cc[i], warp_matrix) = cv2.findTransformECC (world_vid[i,:,:],world_vid[i+1,:,:],warp_matrix, warp_mode, criteria, inputMask = None, gaussFiltSize = 1)
            xshift[i] = warp_matrix[0,2]; yshift[i] = warp_matrix[1,2]
        except:
            cc[i] = np.nan;
            xshift[i]=np.nan; yshift[i] = np.nan;
    
    
    # perform regression to predict frameshift based on eye shifts
    
    #set up models; eyeData (th,phi) = predictors; shift (x,y) = outputs
    xmodel = LinearRegression()
    ymodel = LinearRegression()
    
    eyeData = np.zeros((max_frames,2))
    eyeData[:,0] = dth[0:max_frames];
    eyeData[:,1] = dphi[0:max_frames];
    
    xshiftdata = xshift[0:max_frames];
    yshiftdata = yshift[0:max_frames];
    
    # only use good data - not nans, good correlation between frames, small eye movements(not sacccades which are not compensatory)
    usedata = ~np.isnan(eyeData[:,0]) & ~np.isnan(eyeData[:,1]) & (cc>0.95)  & (np.abs(eyeData[:,0])<2) & (np.abs(eyeData[:,1])<2) & (np.abs(xshiftdata)<5) & (np.abs(yshiftdata)<5)
    
    # fit xshift 
    xmodel.fit(eyeData[usedata,:],xshiftdata[usedata])
    xmap = xmodel.coef_;
    xrscore = xmodel.score(eyeData[usedata,:],xshiftdata[usedata])
    print(xmap, xrscore)

    # fit yshift
    ymodel.fit(eyeData[usedata,:],yshiftdata[usedata])
    ymap = ymodel.coef_;
    yrscore = ymodel.score(eyeData[usedata,:],yshiftdata[usedata])
    print(ymap,yrscore)
    
   # diagnostic plots
    
    fig = plt.figure(figsize = (8,6))
    plt.subplot(2,2,1)
    plt.plot(dth[0:max_frames],xshift[0:max_frames],'.');plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12); plt.ylim(-12,12); plt.xlabel('dtheta'); plt.ylabel('xshift')
    plt.title('xmap = ' + str(xmap))
    plt.subplot(2,2,2)
    plt.plot(dth[0:max_frames],yshift[0:max_frames],'.');plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12); plt.ylim(-12,12); plt.xlabel('dtheta'); plt.ylabel('yshift')
    plt.title('ymap = ' + str(ymap))
    plt.subplot(2,2,3)
    plt.plot(dphi[0:max_frames],xshift[0:max_frames],'.');plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12); plt.ylim(-12,12); plt.xlabel('dphi'); plt.ylabel('xshift')
    plt.subplot(2,2,4)
    plt.plot(dphi[0:max_frames],yshift[0:max_frames],'.');plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12); plt.ylim(-12,12); plt.xlabel('dphi'); plt.ylabel('yshift')
    plt.tight_layout()
    
    return(xmap,ymap,fig)

def plot_ind_contrast_funcs(n_units, goodcells, crange, resp):
    fig = plt.figure(figsize = (6,np.ceil(n_units/2)))
    for i, ind in enumerate(goodcells.index):
        plt.subplot(int(np.ceil(n_units/4)),4,i+1)
        plt.plot(crange[:-2],resp[i,:-2])
    # plt.ylim([0 , max(resp[i,1:-3])*1.2])
        # plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec')
        plt.ylim([0,np.nanmax(resp[i,1:-2])])
    plt.tight_layout()
    return fig

def plot_STA_single_lag(n_units, img_norm, goodcells, worldT, movInterp, ch_count):
    print('get timing')
    model_dt = 0.025;
    model_t = np.arange(0,np.max(worldT),model_dt)
    model_nsp = np.zeros((n_units,len(model_t)))
    
    # get spikes / rate
    print('get spikes')
    bins = np.append(model_t,model_t[-1]+model_dt)
    for i,ind in enumerate(goodcells.index):
        model_nsp[i,:],bins = np.histogram(goodcells.at[ind,'spikeT'],bins)
    
    print('set up video')
    nks = np.shape(img_norm[0,:,:]); nk = nks[0]*nks[1];    
    model_vid = np.zeros((len(model_t),nk))
    for i in range(len(model_t)):
        model_vid[i,:] = np.reshape(movInterp(model_t[i] + model_dt/2),nk)
        
    staAll = np.zeros((n_units,np.shape(img_norm)[1],np.shape(img_norm)[2]))
    model_vid[np.isnan(model_vid)]=0;
    lag = 2
    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
    for c, ind in enumerate(goodcells.index):
        sp = model_nsp[c,:].copy();
        sp = np.roll(sp,-lag)
        sta = model_vid.T@sp;
        sta = np.reshape(sta,nks)
        nsp = np.sum(sp);
        
        plt.subplot(int(np.ceil(n_units/4)),4,c+1)
        ch = int(goodcells.at[ind,'ch'])
        if ch_count == 64:
            shank = np.floor(ch/32); site = np.mod(ch,32)
        else:
            shank = 0; site = ch
        plt.title(f'ind={ind!s} nsp={nsp!s}\n ch={ch!s} shank={shank!s}\n site={site!s}')
        plt.axis('off')
        
        if nsp > 0:
            sta = sta/nsp
        else:
            sta = np.nan
        if pd.isna(sta) is True:
            plt.imshow(np.zeros([120,160]))
        else:
            plt.imshow((sta-np.mean(sta) ),vmin=-0.3,vmax=0.3,cmap = 'jet')
            staAll[c,:,:] = sta
    plt.tight_layout()
    return staAll, fig

def plot_STA_multi_lag(n_units, goodcells, worldT, movInterp):
    print('get timing')
    model_dt = 0.025;
    model_t = np.arange(0,np.max(worldT),model_dt)
    model_nsp = np.zeros((n_units,len(model_t)))
    
    # get spikes / rate
    print('get spikes')
    bins = np.append(model_t,model_t[-1]+model_dt)
    for i,ind in enumerate(goodcells.index):
        model_nsp[i,:],bins = np.histogram(goodcells.at[ind,'spikeT'],bins)
    
    print('set up video')
    nks = np.shape(movInterp(model_t[0])); nk = nks[0]*nks[1];    
    model_vid = np.zeros((len(model_t),nk))
    for i in range(len(model_t)):
        model_vid[i,:] = np.reshape(movInterp(model_t[i] + model_dt/2),nk)

    model_vid[np.isnan(model_vid)]=0;
    sta = 0; 
    lagRange = np.arange(-2,8,2)
    fig = plt.figure(figsize = (12,2*n_units))
    for c, ind in enumerate(goodcells.index):
        for  lagInd, lag in enumerate(lagRange):
            sp = model_nsp[c,:].copy();
            sp = np.roll(sp,-lag)
            sta = model_vid.T@sp;
            sta = np.reshape(sta,nks)
            nsp = np.sum(sp);
            plt.subplot(n_units,6,(c*6)+lagInd + 1)
            if nsp > 0:
                sta = sta/nsp
            else:
                sta = np.nan
            if pd.isna(sta) is True:
                plt.imshow(np.zeros([120,160]))
            else:
                plt.imshow((sta-np.mean(sta) ),vmin=-0.3,vmax=0.3,cmap = 'jet')
            # plt.title(str(c) + ' ' + str(np.round(lag*1000)) + 'msec')
            if c == 0:
                plt.title(str(np.round(lag*model_dt*1000)) + 'msec')
            plt.axis('off')
    plt.tight_layout()
    
    return fig

def plot_spike_triggered_variance(n_units, goodcells, t, movInterp, img_norm):
    stv_all = np.zeros((n_units,np.shape(img_norm)[1],np.shape(img_norm)[2]))
    sta = 0; lag = 0.125
    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
    for c, ind in enumerate(goodcells.index):
        r = goodcells.at[ind,'rate']
        sta = 0
        for i in range(5,t.size-10):
            sta = sta+r[i]*(movInterp(t[i]-lag))**2
        plt.subplot(int(np.ceil(n_units/4)),4,c+1)
        sta = sta/np.sum(r)
        plt.imshow(sta - np.mean(img_norm**2,axis=0),vmin=-1,vmax=1)
        stv_all[c,:,:] = sta - np.mean(img_norm**2,axis=0)
    plt.tight_layout()
    plt.axis('off')
    return stv_all, fig

def fit_glm_vid(model_vid,model_nsp,model_dt, use,nks):
    ### calculate GLM spatial RF
    ### just needs model_vid, model_nsp, use, nks
    ### so easy to make into a function!

    nT = np.shape(model_nsp)[1]
    x = model_vid.copy();
    nk  = nks[0]*nks[1] #image dimensions
    n_units = np.shape(model_nsp)[0]

    #subtract mean and renormalize - necessary? 
    mn_img = np.mean(x[use,:],axis=0)
    x = x-mn_img
    x = x/np.std(x[use,:],axis =0)
    x = np.append(x,np.ones((nT,1)), axis = 1) # append column of ones
    x = x[use,:]

    # set up prior matrix (regularizer)

    #L2 prior
    Imat = np.eye(nk);
    Imat = linalg.block_diag(Imat,np.zeros((1,1)))
    #smoothness prior
    consecutive = np.ones((nk,1))
    consecutive[nks[1]-1::nks[1]] = 0;
    diff = np.zeros((1,2)); diff[0,0] = -1; diff[0,1]= 1;
    Dxx = sparse.diags((consecutive @ diff).T, np.array([0, 1]), (nk-1,nk))
    Dxy = sparse.diags((np.ones((nk,1))@ diff).T, np.array([0, nks[1]]), (nk - nks[1], nk))
    Dx = Dxx.T @ Dxx + Dxy.T @ Dxy
    D  = linalg.block_diag(Dx.toarray(),np.zeros((1,1)))      
    #summed prior matrix
    #Cinv = D + Imat;
    Cinv = D + Imat


    lag_list = [ -4, -2, 0 , 2, 4]
    lambdas = 1024 * (2**np.arange(0,16))
    nlam = len(lambdas)

    sta_all = np.zeros((n_units, len(lag_list), nks[0],nks[1] ));
    cc_all = np.zeros((n_units,len(lag_list)))

    for celln in tqdm(range(n_units)):
        for lag_ind, lag in enumerate(lag_list):


            sps = np.roll(model_nsp[celln,:],-lag)
            sps = sps[use];
            nT = len(sps)

            #split training and test data
            test_frac = 0.3;
            ntest = int(nT*test_frac)
            x_train = x[ntest:,:] ; sps_train = sps[ntest:]
            x_test = x[:ntest,:]; sps_test = sps[:ntest]

            #calculate a few terms
            sta = x_train.T@sps_train/np.sum(sps_train)
            XXtr = x_train.T @ x_train;
            XYtr = x_train.T @sps_train;

            msetrain = np.zeros((nlam,1))
            msetest = np.zeros((nlam,1))
            w_ridge = np.zeros((nk+1,nlam))
            w = sta; # initial guess

            #plt.figure(figsize = (8,8))

            for l in range(len(lambdas)):  # loop over regularization strength

                # calculate MAP estimate               
                w = np.linalg.solve(XXtr + lambdas[l]*Cinv, XYtr)  # equivalent of \ (left divide) in matlab
                w_ridge[:,l] =w;

                #calculate test and training rms error
                msetrain[l] = np.mean((sps_train - x_train@w)**2)
                msetest[l] = np.mean((sps_test - x_test@w)**2)

    #             #plot MAP estimate for this lambda
    #             plt.subplot(4,4,l+1)
    #             crange = np.max(np.abs(w_ridge[:-1,l]))
    #             plt.imshow(np.reshape(w_ridge[:-1,l],nks),vmin = -crange,vmax = crange,cmap = 'jet'); #plt.colorbar()

            # select best cross-validated lambda for RF
            best_lambda = np.argmin(msetest)
            w = w_ridge[:,best_lambda]
            ridge_rf = w_ridge[:,best_lambda]
            sta_all[celln,lag_ind,:,:] = np.reshape(w[:-1],nks);



            #plotting!!!

            # training/test errors
    #         plt.figure(figsize = (4,4))
    #         plt.subplot(2,2,1)
    #         plt.plot(msetrain);  plt.xlabel('lambda'); plt.ylabel('training mse'); plt.title('unit' + str(celln) + ' lag'+str(lag))
    #         plt.subplot(2,2,3)
    #         plt.plot(msetest); plt.xlabel('lambda'); plt.ylabel('testing mse'); plt.title('var=' + str(np.var(sps_test)))


    #         #plot sta (for comparison)
    #         plt.subplot(2,2,2)
    #         crange = np.max(np.abs(sta[:-1]))
    #         plt.imshow(np.reshape(sta[:-1],nks),vmin = -crange, vmax= crange, cmap = 'jet')                                      
    #         plt.title('STA')

    #         #best cross-validated RF
    #         plt.subplot(2,2,4)
    #         crange = np.max(np.abs(w[:-1]))
    #         plt.imshow(np.reshape(w[:-1],nks),vmin = -crange, vmax= crange, cmap = 'jet')
    #         plt.title('GLM fit')       
    #         plt.tight_layout()


            # plot predicted vs actual firing rate


            # predicted firing rate
            sp_pred = x_test@ridge_rf

            # bin the firing rate to get smooth rate vs time
            bin_length = 80;
            sp_smooth = (np.convolve(sps_test,np.ones(bin_length),'same'))/(bin_length*model_dt)
            pred_smooth = (np.convolve(sp_pred,np.ones(bin_length),'same'))/(bin_length*model_dt)

            # a few diagnostics
            err = np.mean((sp_smooth-pred_smooth)**2)
            cc = np.corrcoef(sp_smooth, pred_smooth)
            cc_all[celln,lag_ind] = cc[0,1];


            #plot
    #         plt.figure()
    #         maxt = 3600;
    #         plt.plot(model_t[0:maxt],sp_smooth[0:maxt],label = 'actual')
    #         plt.plot(model_t[0:maxt],pred_smooth[0:maxt],label='predicted')
    #         plt.xlabel('secs'); plt.ylabel('sp/sec'); plt.legend(); plt.xlim([0,90])
    #         expvar =(np.var(sp_smooth) - err)/(np.var(sp_smooth) )
    #         plt.title('exp var = '+str(expvar) + 'correlation = ' + str(cc[0,1]))  


    fig = plt.figure(figsize = (12,2*n_units))
    for celln in tqdm(range(n_units)):
        for lag_ind, lag in enumerate(lag_list):
            crange = np.max(np.abs(sta_all[celln,:,:,:]))
            plt.subplot(n_units,6,(celln*6)+lag_ind + 1)  
            plt.imshow(sta_all[celln,lag_ind,:,:],vmin = -crange,vmax = crange, cmap = 'jet')
            plt.title('cc={:.2f}'.format (cc_all[celln,lag_ind]))
    return(sta_all,cc_all,fig)

def plot_saccade_locked(goodcells, upsacc,  downsacc, trange):
    #upsacc = upsacc[upsacc>5];     upsacc = upsacc[upsacc<np.max(t)-5]
    #downsacc = downsacc[downsacc>5]; downsacc = downsacc[downsacc<np.max(t)-5]
    n_units = len(goodcells)
    upsacc_avg = np.zeros((n_units,trange.size-1))
    downsacc_avg = np.zeros((n_units,trange.size-1))
    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
    for i, ind in enumerate(goodcells.index):
        #rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'])
        for s in np.array(upsacc):
            hist,edges = np.histogram(goodcells.at[ind,'spikeT']-s,trange)
            upsacc_avg[i,:] = upsacc_avg[i,:]+ hist/(upsacc.size*np.diff(trange))
        for s in np.array(downsacc):
            hist,edges = np.histogram(goodcells.at[ind,'spikeT']-s,trange)
            downsacc_avg[i,:]= downsacc_avg[i,:]+ hist/(downsacc.size*np.diff(trange))
        plt.subplot(np.ceil(n_units/4).astype('int'),4,i+1)
        plt.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[i,:])
        plt.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[i,:],'r')
        maxval = np.max(np.maximum(upsacc_avg[i,:],downsacc_avg[i,:]))
        plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2),'r')
        plt.xlim([-0.5,0.5])
        plt.ylim([0,maxval*1.2])
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

def plot_spike_rate_vs_var(use, var_range, goodcells, useT, t, var_name):
    n_units = len(goodcells)
    scatter = np.zeros((n_units,len(use)))
    tuning = np.zeros((n_units,len(var_range)-1))
    tuning_err = tuning.copy()
    var_cent = np.zeros(len(var_range)-1)
    for j in range(len(var_range)-1):
        var_cent[j] = 0.5*(var_range[j] + var_range[j+1])
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'],bounds_error=False)
        scatter[i,:] = rateInterp(useT)
        for j in range(len(var_range)-1):
            usePts =(use>=var_range[j]) & (use<var_range[j+1])
            tuning[i,j] = np.nanmean(scatter[i,usePts])
            tuning_err[i,j] = np.nanstd(scatter[i,usePts])/np.sqrt(np.count_nonzero(usePts))
    fig = plt.figure(figsize = (12,3*np.ceil(n_units/4)))
    for i, ind in enumerate(goodcells.index):
        plt.subplot(np.ceil(n_units/4),4,i+1)
        plt.errorbar(var_cent,tuning[i,:],yerr=tuning_err[i,:])
        try:
            plt.ylim(0,np.nanmax(tuning[i,:]*1.2))
        except ValueError:
            plt.ylim(0,1)
        plt.xlim([var_range[0], var_range[-1]]);  plt.title(ind)
    plt.xlabel(var_name); plt. ylabel('sp/sec')
    plt.tight_layout()
    return var_cent, tuning, tuning_err, fig

def plot_summary(n_units, goodcells, crange, resp, file_dict, staAll, trange, upsacc_avg, downsacc_avg, ori_tuning=None, drift_spont=None):
    samprate = 30000  # ephys sample rate
    fig = plt.figure(figsize = (12,np.ceil(n_units)*2))
    for i, ind in enumerate(goodcells.index): 
        # plot waveform
        plt.subplot(n_units,4,i*4 + 1)
        wv = goodcells.at[ind,'waveform']
        plt.plot(np.arange(len(wv))*1000/samprate,goodcells.at[ind,'waveform'])
        plt.xlabel('msec'); plt.title(str(ind) + ' ' + goodcells.at[ind,'KSLabel']  +  ' cont='+ str(goodcells.at[ind,'ContamPct']))
        
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
        plt.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[i,:])
        plt.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[i,:],'r')
        plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2),'r')
        plt.ylim([0, np.max(upsacc_avg[i,:])*1.8])
        plt.ylabel('sp/sec')
    plt.tight_layout()
    return fig