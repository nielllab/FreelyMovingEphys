"""
figures.py

camera and ephys figures

Oct. 19, 2020
"""

from netCDF4 import Dataset
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
import time
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl 
import wavio
mpl.rcParams['animation.ffmpeg_path'] = r'C:/Program Files/ffmpeg/bin/ffmpeg.exe'
from scipy.interpolate import interp1d
from numpy import nan

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# get figures given paths to the .nc files and ephys .json
def get_figures(config):

    pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(config['save_path'], (trial_name + '_figures.pdf')))

    # TOP1
    top_data = xr.open_dataset(config['top_file'])
    top_vid = np.uint8(top_data['TOP1_video'])
    topT = top_data['timestamps']

    ptNames = top_data['TOP1_pts']
    spinex = ptNames.sel(point_loc = 'spine_x')
    spiney = ptNames.sel(point_loc = 'spine_y')
    spinep = ptNames.sel(point_loc = 'spine_likelihood')

    box = np.ones(11)/11
    spinex = np.convolve(spinex, box, mode='same')
    spiney = np.convolve(spiney, box, mode='same')
    sp = np.sqrt(np.diff(spinex)**2 + np.diff(spiney)**2)*60/10
    sp[0:10]=0
    plt.figure()
    plt.plot(sp[0:1800])
    plt.ylabel('mouse speed (pxl/sec)'); plt.xlabel('frame')
    pdf.savefig()
    plt.close()

    # WORLD
    world_data = xr.open_dataset(config['world_file'])
    world_vid_raw = np.uint8(world_data['WORLD_video'])

    sz = world_vid_raw.shape
    downsamp = 0.25
    world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
    for f in range(sz[0]):
        world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
    worldT = world_data.timestamps

    fig, axs = plt.subplots(1,2,figsize=(8,3))
    axs[0].plot(np.diff(worldT)); axs[0].set_xlabel('frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('world cam')
    axs[1].hist(np.diff(worldT),100);axs[1].set_xlabel('deltaT')
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.imshow(np.mean(world_vid,axis=0)); plt.title('mean worldcam')
    pdf.savefig()
    plt.close()

    # ephys
    ephys_data = pd.read_json(config['ephys_file'])
    ephysT0 = ephys_data.iloc[0,12]

    goodcells = ephys_data.loc[ephys_data['group']=='good']
    units = goodcells.index.values

    # spike rasters
    fig, ax = plt.subplots(figsize=(20,8))
    ax.fontsize = 20
    for i,ind in enumerate(goodcells.index):
    plt.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25)
    plt.xlim(0, 10); plt.xlabel('secs',fontsize = 20); plt.ylabel('unit #',fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pdf.savefig()
    plt.close()

    # generate wave file
    spk =np.array(ephys_data.at[units[1],'spikeT'])

    datarate = 30000
    rate = 44100    # samples per second
    Tmax = 10           # sample duration (seconds)
    f = 440.0       # sound frequency (Hz)
    # Compute waveform samples
    t = np.linspace(0, Tmax, Tmax*datarate,endpoint=False)
    x = np.zeros(np.size(t))
    for spt in spk[spk<Tmax]:
        x[np.int64(spt*datarate) : np.int64(spt*datarate +30)] = 1
        x[np.int64(spt*datarate)+31 : np.int64(spt*datarate +60)] =- 1
    plt.figure()
    plt.plot(x)
    pdf.savefig()
    plt.close()

    # write the samples to a file
    wavio.write("spike.wav", x, datarate, sampwidth=1)

    # REYE
    eye_data = xr.open_dataset(config['eye_file'])
    eye_vid = np.uint8(eye_data['REYE_video'])
    plt.imshow(eye_vid[0,:,:]); plt.title('eye cam')
    eyeT = eye_data.timestamps

    fig, axs = plt.subplots(1,2,figsize=(8,3))
    axs[0].plot(np.diff(eyeT)); axs[0].set_xlabel('frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('eye cam')
    axs[1].hist(np.diff(eyeT),100)
    pdf.savefig()
    plt.close()

    eye_params = eye_data['REYE_ellipse_params']

    plt.figure()
    plt.plot(eye_params.sel(ellipse_params = 'theta')*180/3.1415,eye_params.sel(ellipse_params = 'phi')*180/3.1415,'.')
    pdf.savefig()
    plt.close()

    eyeT = eye_data.timestamps  - ephysT0
    if eyeT[0]<-600:
        eyeT = eyeT + 8*60*60 # 8hr offset for some data
    worldT = world_data.timestamps - ephysT0
    if worldT[0]<-600:
        worldT = worldT + 8*60*60
    topT = top_data.timestamps - ephysT0

    fig,axs = plt.subplots(4,1,figsize = (4,8))
    for i,val in enumerate(eye_params.ellipse_params[0:4]):
        axs[i].plot(eyeT,eye_params.sel(ellipse_params = val))
        axs[i].set_ylabel(val.values)
    pdf.savefig()
    plt.close()

    dEye = np.diff(eye_params.sel(ellipse_params = 'theta'))*180/3.14159

    # plot data figure
    fig = plt.figure(figsize = (6,8))
    gs = fig.add_gridspec(6,2)
    axEye = fig.add_subplot(gs[0:2,0])
    axWorld = fig.add_subplot(gs[0:2,1])
    axTheta = fig.add_subplot(gs[2,:])
    axdTheta = fig.add_subplot(gs[3,:])
    axR = fig.add_subplot(gs[4:6,:])
    #axRad = fig.add_subplot(gs[3,:])

    tr = [0,30] # 340-385
    fr = np.mean(tr) # time for frame
    eyeFr = np.abs(eyeT-fr).argmin(dim = "frame")
    worldFr = np.abs(worldT-fr).argmin(dim = "frame")

    axEye.cla(); axEye.axis('off'); 
    axEye.imshow(eye_vid[eyeFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    #axEye.plot(eye_params.sel(ellipse_params = 'X0')[fr]/2,eye_params.sel(ellipse_params = 'Y0')[fr]/2,'r.')
    axEye.set_xlim(40,200); axEye.set_ylim(0,120)

    axWorld.cla();  axWorld.axis('off'); 
    axWorld.imshow(world_vid[worldFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    
    #plot eye position
    axTheta.cla()
    axTheta.plot(eyeT,0.5*eye_params.sel(ellipse_params = 'theta')*180/3.14159)
    axTheta.set_xlim(tr[0],tr[1]); 
    axTheta.set_ylabel('theta - deg'); axTheta.set_ylim(-30,30)

    # plot eye velocity
    axdTheta.cla()
    axdTheta.plot(eyeT[0:-1],dEye*60/2); ax.set_ylabel('dtheta')
    #sacc = np.transpose(np.where(np.abs(dEye)>10))
    #axdTheta.plot(sacc,np.sign(dEye[sacc])*20,'.')
    axdTheta.set_xlim(tr[0],tr[1]); 
    axdTheta.set_ylim(-10*60,10*60); axdTheta.set_ylabel('eye vel - deg/sec')

    #axRad.cla()
    #axRad.plot(eye_params.sel(ellipse_params = 'longaxis')[frameRange])
    #axRad.set_xlim(0,frameRange[-1]-frameRange[0]); 
    #axRad.set_ylabel('radius'); axRad.set_xlabel('frame #'); axRad.set_ylim(0,40)

    # plot spikes
    axR.fontsize = 20
    for i,ind in enumerate(goodcells.index):
        axR.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25,'k',linewidth=0.25)
    axR.set_xlim(tr[0],tr[1]); axR.set_xlabel('secs'); axR.set_ylabel('unit #')
    axR.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    img_norm = (world_vid-np.mean(world_vid,axis=0))/np.std(world_vid,axis=0)
    contrast = np.empty(worldT.size)
    for i in range(worldT.size):
        contrast[i] = np.std(img_norm[i,:,:])
    plt.figure()
    plt.plot(worldT[0:1000],contrast[0:1000])
    plt.xlabel('time')
    plt.ylabel('contrast')
    pdf.savefig()
    plt.close()

    # plot data figure
    fig = plt.figure(figsize = (6,8))
    gs = fig.add_gridspec(7,3)
    axEye = fig.add_subplot(gs[0:2,1])
    axWorld = fig.add_subplot(gs[0:2,0])
    axTop = fig.add_subplot(gs[0:2,2])
    axTheta = fig.add_subplot(gs[3,:])
    axdTheta = fig.add_subplot(gs[4,:])
    axVid = fig.add_subplot(gs[2,:])
    axR = fig.add_subplot(gs[5:7,:])
    #axRad = fig.add_subplot(gs[3,:])

    tr = [0,30] # 340-385
    fr = np.mean(tr) # time for frame
    eyeFr = np.abs(eyeT-fr).argmin(dim = "frame")
    worldFr = np.abs(worldT-fr).argmin(dim = "frame")

    axEye.cla(); axEye.axis('off'); 
    axEye.imshow(eye_vid[eyeFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    #axEye.plot(eye_params.sel(ellipse_params = 'X0')[fr]/2,eye_params.sel(ellipse_params = 'Y0')[fr]/2,'r.')
    axEye.set_xlim(40,200); axEye.set_ylim(0,120)

    axWorld.cla();  axWorld.axis('off'); 
    axWorld.imshow(world_vid[worldFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")

    axTop.axis('off')
    axTop.imshow(top_vid[0,:,:],'gray',vmin=0,vmax=255,aspect = 'equal')
    
    #plot eye position
    axTheta.cla()
    axTheta.plot(eyeT,0.5*eye_params.sel(ellipse_params = 'theta')*180/3.14159)
    axTheta.set_xlim(tr[0],tr[1]); 
    axTheta.set_ylabel('theta - deg'); axTheta.set_ylim(-30,30)

    # grab bag plot -speed / eye vel

    #speed
    axdTheta.plot(topT[0:-1],sp); axdTheta.set_ylabel('speed cm/sec')
    axdTheta.set_ylim([0, 15])

    #eve velocity
    #axdTheta.plot(eyeT[0:-1],dEye*60/2); ax.set_ylabel('dtheta')
    #sacc = np.transpose(np.where(np.abs(dEye)>10))
    #axdTheta.plot(sacc,np.sign(dEye[sacc])*20,'.')
    #axdTheta.set_ylim(-10*60,10*60); axdTheta.set_ylabel('eye vel - deg/sec')

    axdTheta.set_xlim(tr[0],tr[1]); 

    #plot contrast or radius
    #axVid.plot(worldT,contrast/2.5)
    #axVid.set_ylim([0, 1.1]); axVid.set_ylabel('contrast')

    axVid.plot(eyeT,0.5*(eye_params.sel(ellipse_params = 'longaxis') +eye_params.sel(ellipse_params = 'shortaxis')) )
    axVid.set_xlabel('radius'); axVid.set_ylim([0, 50]); axVid.set_xlim(tr[0],tr[1])

    # plot spikes
    axR.fontsize = 20
    for i,ind in enumerate(goodcells.index):
        axR.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25,'k',linewidth=0.25)
    axR.set_xlim(tr[0],tr[1]); axR.set_xlabel('secs'); axR.set_ylabel('unit #')
    #axR.spines['right'].set_visible(False)
    #axR.spines['top'].set_visible(False)

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    eye_params.sel(ellipse_params = 'theta')

    eyeInterp = interp1d(eyeT,eye_vid,axis=0)
    worldInterp = interp1d(worldT,world_vid,axis=0)

    # make movie and sound

    # unit to plot/record
    this_unit = 1

    # set up figure
    fig = plt.figure(figsize = (6,8))
    gs = fig.add_gridspec(4,2)
    axEye = fig.add_subplot(gs[0,0])
    axWorld = fig.add_subplot(gs[0,1])
    axTheta = fig.add_subplot(gs[1,:])
    axdTheta = fig.add_subplot(gs[2,:])
    axR = fig.add_subplot(gs[3,:])
    # axRad = fig.add_subplot(gs[3,:])

    # timerange and center frame (only)
    tr = [0, 30]
    fr = np.mean(tr) # time for frame
    eyeFr = np.abs(eyeT-fr).argmin(dim = "frame")
    worldFr = np.abs(worldT-fr).argmin(dim = "frame")

    axEye.cla(); axEye.axis('off'); 
    axEye.imshow(eye_vid[eyeFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    #axEye.plot(eye_params.sel(ellipse_params = 'X0')[fr]/2,eye_params.sel(ellipse_params = 'Y0')[fr]/2,'r.')
    axEye.set_xlim(40,200); axEye.set_ylim(0,120)

    axWorld.cla();  axWorld.axis('off'); 
    axWorld.imshow(world_vid[worldFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    
    #plot eye position
    axTheta.cla()
    axTheta.plot(eyeT,eye_params.sel(ellipse_params = 'theta')*180/3.14159)
    axTheta.set_xlim(tr[0],tr[1]); 
    axTheta.set_ylabel('theta - deg'); axTheta.set_ylim(-45,45)

    # plot eye velocity
    axdTheta.cla()
    axdTheta.plot(eyeT[0:-1],dEye*60); ax.set_ylabel('dtheta')
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

    axR.set_xlim(tr[0],tr[1]); axR.set_ylim(-0.5 , 13); axR.set_xlabel('secs'); axR.set_ylabel('unit #')
    axR.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)

    plt.tight_layout()

    # now animate
    writer = FFMpegWriter(fps=30)
    with writer.saving(fig, "eye_world_spikes_092820_wn1_1.mp4", 100):
        for t in np.arange(tr[0],tr[1],1/30):
            
            # show eye and world frames
            axEye.cla(); axEye.axis('off'); 
            axEye.imshow(eyeInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            axEye.set_xlim(40,200); axEye.set_ylim(0,120)
            
            axWorld.cla(); axWorld.axis('off'); 
            axWorld.imshow(worldInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            
            #plot line for time, then remove
            ln = axR.vlines(t,0,30,'b')
            writer.grab_frame()
            ln.remove()
    
    # generate wave file
    sp = np.array(ephys_data.at[units[this_unit],'spikeT'])-tr[0]
    sp = sp[sp>0]
    datarate = 30000
    f = 440.0       # sound frequency (Hz)
    # Compute waveform samples
    tmax = tr[1]-tr[0]
    t = np.linspace(0, tr[1]-tr[0], (tr[1]-tr[0])*datarate,endpoint=False)
    x = np.zeros(np.size(t))
    for spt in sp[sp<tmax]:
        x[np.int64(spt*datarate) : np.int64(spt*datarate +30)] = 1
        x[np.int64(spt*datarate)+31 : np.int64(spt*datarate +60)] =- 1
    plt.plot(x)
    # Write the samples to a file
    wavio.write("wn_spike1.wav", x, datarate, sampwidth=1)

    # normalize world video
    img_norm = (world_vid-np.mean(world_vid,axis=0))/np.std(world_vid,axis=0)
    
    #calculate image contrast
    contrast = np.empty(worldT.size)
    for i in range(worldT.size):
        contrast[i] = np.std(img_norm[i,:,:])
    plt.plot(worldT[0:1000],contrast[0:1000])
    plt.xlabel('time')
    plt.ylabel('contrast')
    pdf.savefig()
    plt.close()

    #set up timebase for subsequent analysis
    dt = 0.025
    t = np.arange(0, np.max(worldT),dt)

    # interpolate and plot contrast
    newc =interp1d(worldT,contrast)
    contrast_interp = newc(t[0:-1])
    contrast_interp.shape
    plt.plot(contrast_interp[0:600])
    plt.title('interp world contrast')
    pdf.savefig()
    plt.close()

    # calculate firing rate at new timebase
    ephys_data['rate'] = nan
    ephys_data['rate'] = ephys_data['rate'].astype(object)
    for i,ind in enumerate(ephys_data.index):
        ephys_data.at[ind,'rate'],bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
    ephys_data['rate']= ephys_data['rate']/dt
    goodcells = ephys_data.loc[ephys_data['group']=='good']

    # plot firing rates
    for i, ind in enumerate(goodcells.index):
        plt.subplot(7,2,i+1)
        plt.plot(t[0:-1],goodcells.at[ind,'rate'])
    pdf.savefig()
    plt.close()

    plt.figure(figsize = (12,4))
    for i, ind in enumerate(goodcells.index):
        plt.subplot(7,2,i+1)
        plt.plot(contrast_interp,goodcells.at[ind,'rate'],'.')
    plt.xlabel('contrast')
    plt.ylabel('rate')
    pdf.savefig()
    plt.close()   

    # calculate contrast - response functions
    # mean firing rate in timebins correponding to contrast ranges
    resp = np.empty((14,20))
    crange = np.arange(0,2,0.1)
    for i,ind in enumerate(goodcells.index):
        for c,cont in enumerate(crange):
            resp[i,c] = np.mean(goodcells.at[ind,'rate'][(contrast_interp>cont) & (contrast_interp<(cont+0.1))])
    plt.plot(crange,np.transpose(resp))
    #plt.ylim(0,10)
    plt.xlabel('contrast')
    plt.ylabel('sp/sec')
    pdf.savefig()
    plt.close()

    fig = plt.figure(figsize = (12,8))
    for i, ind in enumerate(goodcells.index):
        plt.subplot(3,5,i+1)
        plt.plot(crange[0:-3],resp[i,0:-3])
        plt.ylim([0 , max(resp[i,1:-3])*1.2])
        plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec')
    pdf.savefig()
    plt.close()

    movInterp = interp1d(worldT,img_norm,axis=0)

    # calculate spike-triggered average
    sta = 0
    lag = 0.05
    plt.figure(figsize = (12,8))
    for c, ind in enumerate(goodcells.index):
        r = goodcells.at[ind,'rate']
        for i in range(100,t.size-100):
            sta = sta+r[i]*(movInterp(t[i]-lag))
        plt.subplot(3,5,c+1)
        sta = sta/np.sum(r)
        #sta[abs(sta)<0.1]=0
        plt.imshow((sta - np.mean(sta)),vmin=-0.45,vmax=0.45,cmap = 'jet')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # calculate spike-triggered variance
    sta = 0
    lag = 0.035
    plt.figure(figsize = (12,8))
    for c, ind in enumerate(goodcells.index):
        r = goodcells.at[ind,'rate']
        for i in range(5,t.size-10):
            sta = sta+r[i]*(movInterp(t[i]-lag))**2
        plt.subplot(3,5,c+1)
        sta = sta/np.sum(r)
        plt.imshow(sta)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    fig = plt.figure(figsize = (12,8))
    trange = np.arange(-1,1.1,0.1)
    sthresh = 5;
    upsacc = eyeT[np.append(dEye,0)>sthresh]
    upsacc = upsacc[upsacc>2]
    downsacc= eyeT[np.append(dEye,0)<-sthresh]
    downsacc = downsacc[downsacc>2]
    upsacc_avg = np.zeros((units.size,trange.size))
    downsacc_avg = np.zeros((units.size,trange.size))
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'])
        for s in upsacc:
            upsacc_avg[i,:] = upsacc_avg[i,:]+ rateInterp(np.array(s)+trange)/upsacc.size
        for s in downsacc:
            downsacc_avg[i,:]= downsacc_avg[i,:]+ rateInterp(np.array(s)+trange)/upsacc.size
        plt.subplot(3,5,i+1)
        plt.plot(trange,upsacc_avg[i,:])
        #plt.plot(trange,downsacc_avg[i,:],'r')
        plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2),'r')
        plt.ylim([0, np.max(upsacc_avg[i,:])*1.2])
        plt.ylabel('sp/sec')
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    pdf.close()

