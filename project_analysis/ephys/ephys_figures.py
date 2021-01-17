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
def plot_cam_time(camT, camname)
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
    fig, ax = plt.figure()
    plt.plot(np.rad2deg(eye_params.sel(ellipse_params='theta')),np.rad2deg(eye_params.sel(ellipse_params='phi')),'.')
    plt.xlabel('theta'); plt.ylabel('phi'); plt.title('eye position accross recording')
    return fig

# optical mouse speed
def plot_optmouse_spd(spd_tstamps, spd):
    fig, ax = plt.figure()
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
    fig, ax = plt.figure()
    plt.subplot(1,2,1)
    plt.plot(th[(35*60):(40*60)]); plt.title('theta')
    plt.subplot(1,2,2)
    plt.plot(th_switch[(35*60):(40*60)]); plt.title('theta switch')
    return fig

# plot the four main paramters of the eye tracking
def plot_eye_params(eye_params, eyeT):
    fig, axs = plt.subplots(4,1)
    for i,val in enumerate(eye_params.ellipse_params[0:4]):
        axs[i].plot(eyeT,eye_params.sel(ellipse_params = val))
        axs[i].set_ylabel(val.values)
    return fig

def make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, this_unit, accT=None, gz=None, speedT=None, spd=None):
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

    vidfile = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+this_unit+'.mp4'))
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

def make_sound(file_dict, ephys_data, this_unit):
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
    audfile = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+this_unit+'.wav'))
    wavio.write(audfile, x, datarate, sampwidth=1)

    return audfile
