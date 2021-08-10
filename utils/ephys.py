"""
ephys.py
"""
import gc, json, os, cv2, platform, subprocess, traceback
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
else:
    mpl.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
from matplotlib.backends.backend_pdf import PdfPages
from numpy import nan
from scipy.interpolate import interp1d
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from datetime import datetime
import scipy.interpolate
from scipy.signal import butter, freqz, lfilter, sosfiltfilt, medfilt
from tqdm import tqdm
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.signal as signal
import wavio
from matplotlib.animation import FFMpegWriter
from matplotlib.backends.backend_pdf import PdfPages
from numpy import nan
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import scipy.signal
from scipy.io import loadmat
from scipy.ndimage import shift as imshift
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from utils.aux_funcs import nanxcorr
from utils.paths import find, list_subdirs
from utils.time import open_time

def format_spikes(merge_file, samprate):
    """
    spike preprocessing, which splits one .mat ephys file into individual .json
    files in each recording directory
    INPUTS
        merge_file: .mat file
        samprate: ephys sample rate
    OUTPUTS
        None
    """
    # open 
    merge_info = loadmat(merge_file)
    fileList = merge_info['fileList']
    pathList = merge_info['pathList']
    nSamps = merge_info['nSamps']

    # load phy2 output data
    phy_path = os.path.split(merge_file)
    allSpikeT = np.load(os.path.join(phy_path[0],'spike_times.npy'))
    clust = np.load(os.path.join(phy_path[0],'spike_clusters.npy'))
    templates = np.load(os.path.join(phy_path[0],'templates.npy'))

    # ephys_data_master holds information that is same for all recordings (i.e. cluster information + waveform)
    ephys_data_master = pd.read_csv(os.path.join(phy_path[0],'cluster_info.tsv'),sep = '\t',index_col=0)

    # insert waveforms
    ephys_data_master['waveform'] = np.nan
    ephys_data_master['waveform'] = ephys_data_master['waveform'].astype(object)
    for _, ind in enumerate(ephys_data_master.index):
        ephys_data_master.at[ind,'waveform'] = templates[ind,21:,ephys_data_master.at[ind,'ch']]

    # create boundaries between recordings (in terms of timesamples)
    boundaries = np.concatenate((np.array([0]),np.cumsum(nSamps)))

    # loop over each recording and create/save ephys_data for each one
    for s in range(np.size(nSamps)):

        # select spikes in this timerange
        use = (allSpikeT >= boundaries[s]) & (allSpikeT<boundaries[s+1])
        theseSpikes = allSpikeT[use]
        theseClust = clust[use[:,0]]

        # place spikes into ephys data structure
        ephys_data = ephys_data_master.copy()
        ephys_data['spikeT'] = np.NaN
        ephys_data['spikeT'] = ephys_data['spikeT'].astype(object)
        for c in np.unique(clust):
            ephys_data.at[c,'spikeT'] =(theseSpikes[theseClust==c].flatten() - boundaries[s])/samprate
        
        # get timestamp from csv for this recording
        fname = fileList[0,s][0].copy()
        fname = fname[0:-4] + '_BonsaiBoardTS.csv'
        ephys_time_path = os.path.join(pathList[0,s][0],fname)
        ephys_data['t0'] = open_time(ephys_time_path)[0]
        
        # write ephys data into json file
        fname = fileList[0,s][0].copy()
        fname = fname[0:-10] + '_ephys_merge.json'
        ephys_json_path = os.path.join(pathList[0,s][0],fname)
        ephys_data.to_json(ephys_json_path)

def read_ephys_bin(binary_path, probe_name, do_remap=True, mapping_json=None):
    """
    read in ephys binary file and apply remapping using the name of the probe,
    where the binary dimensions and remaping vector are read in from relative
    path within the FreelyMovingEphys directory (FreelyMovingEphys/matlab/channel_maps.json)
    INPUTS
        binary_path: path to binary file
        probe_name: name of probe, which should be a key in the dict stored in the .json of probe remapping vectors
        do_remap: bool, whether or not to remap the drive
        mapping_json: path to a .json in which each key is a probe name and each value is the 1-indexed sequence of channels
    OUTPUTS
        ephys: ephys DataFrame
    """
    # get channel number
    if '16' in probe_name:
        ch_num = 16
    elif '64' in probe_name:
        ch_num = 64
    elif '128' in probe_name:
        ch_num = 128
    if mapping_json is not None:
        # open file of default mappings
        with open(mapping_json, 'r') as fp:
            mappings = json.load(fp)
        # get the mapping for the probe name used in the current recording
        ch_remap = mappings[probe_name]
    # set up data types to read binary file into
    dtypes = np.dtype([("ch"+str(i),np.uint16) for i in range(0,ch_num)])
    # read in binary file
    ephys = pd.DataFrame(np.fromfile(binary_path, dtypes, -1, ''))
    # remap with known order of channels
    if do_remap is True:
        ephys = ephys.iloc[:,[i-1 for i in list(ch_remap)]]
    return ephys

def butter_bandpass(data, lowcut=1, highcut=300, fs=30000, order=5):
    """
    apply bandpass filter to ephys lfp applied along axis=0
    axis=0 should be the time dimension for any data passed in
    INPUTS
        data: 2d array of multiple channels of ephys data as a numpy array or pandas dataframe
        lowcut: low end of cut off for frequency
        highcut: high end of cut off for frequency
        fs: sample rate
        order: order of filter
    OUTPUTS
        filtered data in the same type as input data
    """
    nyq = 0.5 * fs # Nyquist frequency
    low = lowcut / nyq # low cutoff
    high = highcut / nyq # high cutoff
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, data, axis=0)

def find_files(rec_path, rec_name, free_move, cell, stim_type, mp4, probe_name, drop_slow_frames):
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
        file_dict: dict of the paths to important files and options with which to run ephys analysis
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
    if platform.system() == 'Linux':
        mapping_json = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/probes/channel_maps.json'
    else:
        mapping_json = '/'.join(os.path.abspath(__file__).split('\\')[:-2]) + '/probes/channel_maps.json'
    # shorten gratings stim, since 'grat' is the str used in ephys analysis
    # this can be eliminated if either name passed in or usage in run_ephys_analysis is changed
    if stim_type == 'gratings':
        stim_type = 'grat'
    # assemble dict
    if free_move is True:
        dict_out = {'cell':cell,'top':top_file,'eye':eye_file,'world':world_file,'ephys':ephys_file,
        'ephys_bin':ephys_bin_file,'speed':None,'imu':imu_file,'save':rec_path,'name':rec_name,
        'stim_type':stim_type,'mp4':mp4,'probe_name':probe_name,'mapping_json':mapping_json, 'drop_slow_frames':drop_slow_frames}
    elif free_move is False:
        dict_out = {'cell':cell,'eye':eye_file,'world':world_file,'ephys':ephys_file,'ephys_bin':ephys_bin_file,
        'speed':speed_file,'imu':None,'save':rec_path,'name':rec_name,'stim_type':stim_type,
        'mp4':mp4,'probe_name':probe_name,'mapping_json':mapping_json, 'drop_slow_frames':drop_slow_frames}
    return dict_out

def plot_psth(goodcells, onsets, lower, upper, dt):
    """
    calculate and plot psth relative to timepoints
    histogram of times when neurons fire, used currently for gratings stimulus
    INPUTS
        goodcells: ephys dataframe
        onsets: array of times when identical stimulus starts
        lower: lower-bound of bin
        upper: upper-bound of bin
        dt: bin dt
    OUTPUTS:
        fig: figure
        psth_all: psth for all units
    """
    # number of units in recording
    n_units = len(goodcells)
    # create bins
    bins = np.arange(lower, upper+dt, dt)
    # setup figure
    fig = plt.figure(figsize=(10,np.ceil(n_units/2)))
    # empty array into which psth will be saved
    psth_all = np.zeros((n_units,len(bins)-1))
    # iterate through units
    for i, ind in enumerate(goodcells.index):
        plt.subplot(int(np.ceil(n_units/4)),4,i+1)
        # empty array for psth of this unit
        psth = np.zeros(len(bins)-1)
        # iterate through onset moments i.e. each time the identical stimulus begins
        for t in onsets:
            # get a histogram of spike times in each of the stimulus bins
            hist, edges = np.histogram(goodcells.at[ind,'spikeT']-t, bins)
            # make this cumulative
            psth = psth + hist
        # normalize spikes in bins to the number of times the stim had an onset
        psth = psth/len(onsets)
        # then normalize to length of time for each bin
        psth = psth/dt
        # plot histogram as a line
        plt.plot(bins[0:-1]+dt/2, psth)
        plt.ylim(0,np.nanmax(psth)*1.2)
        # add psth from this unit to array of all units
        psth_all[i,:] = psth
    plt.xlabel('time'); plt.ylabel('sp/sec')
    plt.tight_layout()
    plt.close()
    return fig, psth_all

def plot_spike_raster(goodcells):
    """
    plot spike raster so that superficial channels are at the top of the panel
    INPUTS
        goodcells: ephys dataframe
    OUTPUTS
        fig: figure
    """
    fig, ax = plt.subplots()
    ax.fontsize = 20
    n_units = len(goodcells)
    # iterate through units
    for i, ind in enumerate(goodcells.index):
        # array of spike times
        sp = np.array(goodcells.at[ind,'spikeT'])
        # make vertical line for each time the unit fires
        plt.vlines(sp[sp<10],i-0.25,i+0.25)
        # plot only ten seconds
        plt.xlim(0, 10)
        # turn off ticks
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.xlabel('secs',fontsize = 20)
    plt.ylabel('unit number',fontsize=20)
    plt.ylim([n_units,0])
    return fig

def plot_eye_pos(eye_params):
    """
    plot eye parameters, theta and phi, over the recording
    INPUTS
        eye_params: xarray of eye parameters
    OUTPUTS
        fig: figure
    """
    # get out theta and phi
    th = np.rad2deg(eye_params.sel(ellipse_params='theta').values)[0:-1:10]
    phi = np.rad2deg(eye_params.sel(ellipse_params='phi').values)[0:-1:10]
    # get the fraction of good points in the recording
    good_pts = np.sum(~np.isnan(th))/len(th)
    fig = plt.figure()
    plt.plot(th, phi, '.')
    plt.xlabel('theta'); plt.ylabel('phi')
    plt.title('eye position (fraction good = '+str(np.round(good_pts,3))+')')
    return fig

def plot_param_switch_check(eye_params):
    """
    flip the order of frames from [1,2,3,4] to [2,1,4,3,]
    if deinterlacing was done correctly, reversing the order of frames will
    make the trace of theta bumpy, but if frames were put into the video file
    in the wrong order when interlaced frames were split apart, th_switch will be
    smoother than th itself
    INPUTS
        eye_params: xarray of eye parameters
    OUTPUTS
        fig: figure
        th_switch: theta with switched frames
    """
    # get theta and phi out
    th = eye_params.sel(ellipse_params='theta')
    # zero-center values by subtracting mean off
    th = np.rad2deg(th - np.nanmean(th))
    # now flip the order of frames in an every-other fashion
    th_switch = np.zeros(np.shape(th))
    th_switch[0:-1:2] = np.array(th[1::2])
    th_switch[1::2] = np.array(th[0:-1:2])
    # make the plot of theta
    # plot will be of 5sec starting 35sec into the video
    start = 35*60; stop = 40*60
    fig, ax = plt.subplots(121)
    plt.subplot(1,2,1)
    plt.plot(th[start:stop])
    plt.title('theta')
    plt.subplot(1,2,2)
    plt.plot(th_switch[start:stop])
    plt.title('theta switch')
    return fig, th_switch

def plot_acc_eyetime_alignment(eyeT, t1, offset, ccmax):
    """
    plot alignemnt between eye timestamps and imu timestamps
    INPUTS
        eyeT: timestamps from eye
        t1:
        offset:
        ccmax:
    OUTPUTS
        fig: figure
    """
    fig = plt.subplot(1,2,1)
    plt.plot(eyeT[t1*60],offset)
    plt.xlabel('secs'); plt.ylabel('offset (secs)')
    plt.subplot(1,2,2)
    plt.plot(eyeT[t1*60],ccmax)
    plt.xlabel('secs'); plt.ylabel('max cc')
    return fig

def plot_regression_timing_fit(dataT, offset, offset0, drift_rate):
    """
    INPUTS
        dataT:
        offset:
        offset0:
        drift_rate:
    OUTPUTS
        fig: figure
    """
    fig = plt.figure()
    plt.plot(dataT, offset,'.')
    plt.plot(dataT, offset0 + dataT*drift_rate)
    plt.xlabel('secs'); plt.ylabel('offset - secs')
    plt.title('offset0 = '+str(np.round(offset0,3))+' drift_rate = '+str(np.round(drift_rate,3)))
    return fig

def plot_saccade_and_fixate(eyeT, dEye, gInterp, th):
    """
    plot dEye and dHead on top of each other
    """
    fig = plt.figure()
    plt.subplot(1,2,1)
    # plot dEye and gyro interpolated to match timing of eye
    plt.plot(eyeT[0:-1],dEye, label = 'dEye')
    plt.plot(eyeT, gInterp(eyeT), label = 'dHead')
    plt.xlim(37,39); plt.ylim(-10,10)
    plt.legend(); plt.ylabel('deg'); plt.xlabel('secs')
    plt.subplot(1,2,2)
    plt.plot(eyeT[0:-1],np.nancumsum(gInterp(eyeT[0:-1])), label = 'head')
    plt.plot(eyeT[0:-1],np.nancumsum(gInterp(eyeT[0:-1])+dEye),label ='gaze')
    plt.plot(eyeT[1:],th[0:-1],label ='eye')
    plt.xlim(35,40); plt.ylim(-30,30); plt.legend(); plt.ylabel('deg'); plt.xlabel('secs')
    plt.tight_layout()
    return fig

def eye_shift_estimation(th, phi, eyeT, world_vid, worldT, max_frames=3600):
    """
    do a simple shift of the worldcam using eye parameters
    aim is to approximate the visual scene
    INPUTS
        th: theta as an array
        phi: phi as an array
        eyeT: eye timestamps
        world_vid: worldcam video as an array
        worldT: worldcam timestamps
        max_frames: number of frames to use in the estimation
    OUTPUTS
        fig: figure
        xmap: worldcam x-correction factor
        ymap: worldcam y-correction factor
    """
    # get eye displacement for each worldcam frame
    th_interp = interp1d(eyeT, th, bounds_error=False)
    phi_interp = interp1d(eyeT, phi, bounds_error=False)
    dth = np.diff(th_interp(worldT))
    dphi = np.diff(phi_interp(worldT))
    # calculate x-y shift for each worldcam frame  
    number_of_iterations = 5000
    termination_eps = 1e-4
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    warp_mode = cv2.MOTION_TRANSLATION
    cc = np.zeros(max_frames)
    xshift = np.zeros(max_frames)
    yshift = np.zeros(max_frames)
    warp_all = np.zeros((6,max_frames))
    # get shift between adjacent frames
    for i in tqdm(range(max_frames)):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try: 
            (cc[i], warp_matrix) = cv2.findTransformECC(world_vid[i,:,:], world_vid[i+1,:,:], warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
            xshift[i] = warp_matrix[0,2]
            yshift[i] = warp_matrix[1,2]
        except:
            cc[i] = np.nan
            xshift[i]=np.nan
            yshift[i] = np.nan
    # perform regression to predict frameshift based on eye shifts
    # set up models
    xmodel = LinearRegression()
    ymodel = LinearRegression()
    # eye data as predictors
    eyeData = np.zeros((max_frames,2))
    eyeData[:,0] = dth[0:max_frames]
    eyeData[:,1] = dphi[0:max_frames]
    # shift in x and y as outputs
    xshiftdata = xshift[0:max_frames]
    yshiftdata = yshift[0:max_frames]
    # only use good data
    # not nans, good correlation between frames, small eye movements (no sacccades, only compensatory movements)
    usedata = ~np.isnan(eyeData[:,0]) & ~np.isnan(eyeData[:,1]) & (cc>0.95)  & (np.abs(eyeData[:,0])<2) & (np.abs(eyeData[:,1])<2) & (np.abs(xshiftdata)<5) & (np.abs(yshiftdata)<5)
    # fit xshift
    xmodel.fit(eyeData[usedata,:],xshiftdata[usedata])
    xmap = xmodel.coef_
    xrscore = xmodel.score(eyeData[usedata,:],xshiftdata[usedata])
    # fit yshift
    ymodel.fit(eyeData[usedata,:],yshiftdata[usedata])
    ymap = ymodel.coef_
    yrscore = ymodel.score(eyeData[usedata,:],yshiftdata[usedata])
   # diagnostic plots
    fig = plt.figure(figsize = (8,6))
    plt.subplot(2,2,1)
    plt.plot(dth[0:max_frames],xshift[0:max_frames],'.')
    plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12)
    plt.ylim(-12,12); plt.xlabel('dtheta'); plt.ylabel('xshift')
    plt.title('xmap = '+str(xmap))
    plt.subplot(2,2,2)
    plt.plot(dth[0:max_frames],yshift[0:max_frames],'.')
    plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12)
    plt.ylim(-12,12); plt.xlabel('dtheta'); plt.ylabel('yshift')
    plt.title('ymap = '+str(ymap))
    plt.subplot(2,2,3)
    plt.plot(dphi[0:max_frames],xshift[0:max_frames],'.')
    plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12)
    plt.ylim(-12,12); plt.xlabel('dphi'); plt.ylabel('xshift')
    plt.subplot(2,2,4)
    plt.plot(dphi[0:max_frames],yshift[0:max_frames],'.')
    plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12)
    plt.ylim(-12,12); plt.xlabel('dphi'); plt.ylabel('yshift')
    plt.tight_layout()
    return fig, xmap, ymap

def plot_STA(goodcells, img_norm, worldT, movInterp, ch_count, lag=2, show_title=True):
    """
    plot spike-triggered average for either a single lag or a range of lags
    INPUTS
        goodcells: dataframe of ephys data
        img_norm: normalized worldcam video
        worldT: worldcam timestamps
        movInterp: interpolator for worldcam movie
        ch_count: number of probe channels
        lag: time lag, should be np.arange(-2,8,2) for range of lags, or 2 for single lag
        show_title: bool, whether or not to show title above each panel
    OUTPUTS
        staAll: STA receptive field of each unit
        fig: figure
    """
    n_units = len(goodcells)
    # model setup
    model_dt = 0.025
    model_t = np.arange(0, np.max(worldT), model_dt)
    model_nsp = np.zeros((n_units, len(model_t)))
    # get binned spike rate
    bins = np.append(model_t, model_t[-1]+model_dt)
    for i, ind in enumerate(goodcells.index):
        model_nsp[i,:], bins = np.histogram(goodcells.at[ind,'spikeT'], bins)
    # settting up video
    nks = np.shape(img_norm[0,:,:])
    nk = nks[0]*nks[1]
    model_vid = np.zeros((len(model_t),nk))
    for i in range(len(model_t)):
        model_vid[i,:] = np.reshape(movInterp(model_t[i]+model_dt/2), nk)
    # spike-triggered average
    staAll = np.zeros((n_units, np.shape(img_norm)[1], np.shape(img_norm)[2]))
    model_vid[np.isnan(model_vid)] = 0
    if type(lag) == int:
        fig = plt.subplots(int(np.ceil(n_units/10)),10,figsize=(20,np.int(np.ceil(n_units/3))),dpi=50)
        for c, ind in enumerate(goodcells.index):
            sp = model_nsp[c,:].copy()
            sp = np.roll(sp, -lag)
            sta = model_vid.T @ sp
            sta = np.reshape(sta, nks)
            nsp = np.sum(sp)
            plt.subplot(int(np.ceil(n_units/10)),10,c+1)
            ch = int(goodcells.at[ind,'ch'])
            if ch_count == 64 or ch_count == 128:
                shank = np.floor(ch/32); site = np.mod(ch,32)
            else:
                shank = 0; site = ch
            if show_title:
                plt.title(f'ind={ind!s} nsp={nsp!s}\n ch={ch!s} shank={shank!s}\n site={site!s}',fontsize=5)
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
    else:
        lagRange = lag
        fig = plt.subplots(n_units,5,figsize=(6, np.int(np.ceil(n_units/2))),dpi=300)
        for c, ind in enumerate(goodcells.index):
            for lagInd, lag in enumerate(lagRange):
                sp = model_nsp[c,:].copy()
                sp = np.roll(sp,-lag)
                sta = model_vid.T@sp
                sta = np.reshape(sta,nks)
                nsp = np.sum(sp)
                plt.subplot(n_units,5,(c*5)+lagInd + 1)
                if nsp > 0:
                    sta = sta/nsp
                else:
                    sta = np.nan
                if pd.isna(sta) is True:
                    plt.imshow(np.zeros([120,160]))
                else:
                    plt.imshow((sta-np.mean(sta)),vmin=-0.3,vmax=0.3,cmap = 'jet')
                if c == 0:
                    plt.title(str(np.round(lag*model_dt*1000)) + 'msec',fontsize=5)
                plt.axis('off')
            plt.tight_layout()
        return fig

def plot_STV(goodcells, t, movInterp, img_norm):
    """
    plot spike-triggererd varaince
    INPUTS
        goodcells: ephys dataframe
        t: timebase from worldT
        movInterp: interpolator for worldcam movie
        img_norm: normalized worldcam video
    OUTPUTS
        stvAll: spike triggered variance for all units
        fig: figure
    """
    n_units = len(goodcells)
    stvAll = np.zeros((n_units,np.shape(img_norm)[1],np.shape(img_norm)[2]))
    sta = 0
    lag = 0.125
    fig = plt.subplots(int(np.ceil(n_units/10)),10,figsize=(20,np.int(np.ceil(n_units/3))),dpi=50)
    for c, ind in enumerate(goodcells.index):
        r = goodcells.at[ind,'rate']
        sta = 0
        for i in range(5, t.size-10):
            sta = sta+r[i]*(movInterp(t[i]-lag))**2
        plt.subplot(int(np.ceil(n_units/10)), 10, c+1)
        sta = sta/np.sum(r)
        plt.imshow(sta - np.mean(img_norm**2,axis=0), vmin=-1, vmax=1)
        stvAll[c,:,:] = sta - np.mean(img_norm**2, axis=0)
        plt.axis('off')
    plt.tight_layout()
    return stvAll, fig

def plot_overview(goodcells, crange, resp, file_dict, staAll, trange, upsacc_avg, downsacc_avg,
                 ori_tuning=None, drift_spont=None, grating_ori=None, sf_cat=None, grating_rate=None,
                 spont_rate=None):
    """
    overview figure of ephys analysis
    INPUTS
        goodcells: ephys dataframe
        crange: range of contrast bins used
        resp: contrast reponse for each bin in crange
        file_dict: dict of files and options (comes from function find_files)
        staAll: STA for all units
        trange: range of time values used in head/eye movement plots
        upsacc_avg: average eye movement reponse
        downsacc_avg: average eye movement reponse
        ori_tuning: orientation reponse
        drift_spont: grating orientation spont
        grating_ori: grating response
        sf_cat: spatial frequency categories
        grating_rate: sp/sec at each sf
        spont_rate: grating spontanious rate
    OUTPUTS
        fig: figure
    """
    n_units = len(goodcells)
    samprate = 30000  # ephys sample rate
    fig = plt.figure(figsize=(5,np.int(np.ceil(n_units/3))),dpi=50)
    for i, ind in enumerate(goodcells.index): 
        # plot waveform
        plt.subplot(n_units,4,i*4 + 1)
        wv = goodcells.at[ind,'waveform']
        plt.plot(np.arange(len(wv))*1000/samprate,goodcells.at[ind,'waveform'])
        plt.xlabel('msec')
        plt.title(str(ind)+' '+goodcells.at[ind,'KSLabel']+' cont='+str(goodcells.at[ind,'ContamPct']))
        # plot CRF
        if grating_ori is not None:
            # for gratings stim, plot orientation tuning curve
            plt.subplot(n_units,4,i*4 + 2)
            plt.scatter(grating_ori,grating_rate[i,:],c=sf_cat)
            plt.plot(3*np.ones(len(spont_rate[i,:])),spont_rate[i,:],'r.')
        if file_dict['stim_type'] == 'dark_arena':
            # dark arena will have no values for contrast response function
            # skip this panel for now
            plt.subplot(n_units,4,i*4 + 2)
            plt.axis('off')
        else:
            # for light fm and all hf besides gratings, plot CRF
            plt.subplot(n_units,4,i*4 + 2)
            plt.plot(crange[2:-1],resp[i,2:-1])
            plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec')
            try:
                plt.ylim([0,np.nanmax(resp[i,2:-1])])
            except ValueError:
                plt.ylim(0,1)
        # plot STA or tuning curve
        plt.subplot(n_units,4,i*4 + 3)
        if ori_tuning is not None:
            plt.plot(np.arange(8)*45, ori_tuning[i,:,0], label = 'low sf')
            plt.plot(np.arange(8)*45, ori_tuning[i,:,1], label = 'mid sf')
            plt.plot(np.arange(8)*45, ori_tuning[i,:,2], label = 'hi sf')
            plt.plot([0,315],[drift_spont[i],drift_spont[i]],'r:', label = 'spont')
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

def plot_spike_rate_vs_var(use, var_range, goodcells, useT, t, var_label):
    """
    plot spike rate vs a given variable (e.g. pupil radius, worldcam contrast, etc.)
    INPUTS
        use: varaible to plot (can be filtered e.g. only active times)
        var_range: range of bins to calculate response over
        goodcells: ephys dataframe
        useT: timestamps that match the vairable (use)
        t: timebase
        var_label: label for last panels xlabel
    OUTPUTS
        var_cent: x axis bins
        tuning: tuning curve across bins
        tuning_err: stderror of variable at each bin
        fig: figure
    """
    n_units = len(goodcells)
    scatter = np.zeros((n_units,len(use)))
    tuning = np.zeros((n_units,len(var_range)-1))
    tuning_err = tuning.copy()
    var_cent = np.zeros(len(var_range)-1)
    for j in range(len(var_range)-1):
        var_cent[j] = 0.5*(var_range[j] + var_range[j+1])
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1], goodcells.at[ind,'rate'], bounds_error=False)
        scatter[i,:] = rateInterp(useT)
        for j in range(len(var_range)-1):
            usePts = (use>=var_range[j]) & (use<var_range[j+1])
            tuning[i,j] = np.nanmean(scatter[i, usePts])
            tuning_err[i,j] = np.nanstd(scatter[i, usePts]) / np.sqrt(np.count_nonzero(usePts))
    fig = plt.subplots(np.ceil(n_units/7).astype('int'),7,figsize=(35,np.int(np.ceil(n_units/3))),dpi=50)
    for i, ind in enumerate(goodcells.index):
        plt.subplot(np.ceil(n_units/10),10,i+1)
        plt.errorbar(var_cent,tuning[i,:],yerr=tuning_err[i,:])
        try:
            plt.ylim(0,np.nanmax(tuning[i,:]*1.2))
        except ValueError:
            plt.ylim(0,1)
        plt.xlim([var_range[0], var_range[-1]]); plt.title(ind,fontsize=5)
        plt.xlabel(var_label,fontsize=5); plt.ylabel('sp/sec',fontsize=5)
        plt.xticks(fontsize=5); plt.yticks(fontsize=5)
    plt.tight_layout()
    return var_cent, tuning, tuning_err, fig

def plot_saccade_locked(goodcells, upsacc, downsacc, trange):
    """
    plot average eye movements in window
    upsacc and downsacc can be filtered ahead of time so only include times when e.g. dHead crosses threshold
    INPUTS
        goodcells: ephys dataframe
        upsacc: eye timestamps when there is an eye movement in the positive direction (left)
        downsacc: eye timestamps when there is an eye movement in the negative direction (right)
        trange: time window to plot
    OUTPUTS
        upsacc_avg: trace of average saccade to the left
        downsacc_avg: trace of average saccade to the right
        fig: figure
    """
    n_units = len(goodcells)
    upsacc_avg = np.zeros((n_units,trange.size-1))
    downsacc_avg = np.zeros((n_units,trange.size-1))
    fig = plt.subplots(np.ceil(n_units/7).astype('int'),7,figsize=(35,np.int(np.ceil(n_units/3))),dpi=50)
    for i, ind in enumerate(goodcells.index):
        for s in np.array(upsacc):
            hist, edges = np.histogram(goodcells.at[ind,'spikeT']-s, trange)
            upsacc_avg[i,:] = upsacc_avg[i,:] + hist / (upsacc.size*np.diff(trange))
        for s in np.array(downsacc):
            hist,edges = np.histogram(goodcells.at[ind,'spikeT']-s,trange)
            downsacc_avg[i,:] = downsacc_avg[i,:]+ hist/(downsacc.size*np.diff(trange))
        plt.subplot(np.ceil(n_units/7).astype('int'),7,i+1)
        plt.plot(0.5*(trange[0:-1] + trange[1:]), upsacc_avg[i,:])
        plt.plot(0.5*(trange[0:-1] + trange[1:]), downsacc_avg[i,:],'r')
        maxval = np.max(np.maximum(upsacc_avg[i,:], downsacc_avg[i,:]))
        plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2), 'r')
        plt.xlim([-0.5, 0.5])
        plt.ylim([0,maxval*1.2])
        plt.ylabel('sp/sec')
    plt.tight_layout()
    return upsacc_avg, downsacc_avg, fig

def fit_glm_vid(model_vid, model_nsp, model_dt, use, nks):
    """
    calculate GLM spatial receptive field
    INPUTS
        model_vid: video as array
        model_nsp: binned number of spikes
        model_dt: dt
        use: frames when animal is active
        nks: dimensions of video
    OUTPUTS
        sta_all: receptive fields for each unit
        cc_all: cross correlation for each unit
        fig: figure
    """
    nT = np.shape(model_nsp)[1]
    x = model_vid.copy()
    # image dimensions
    nk  = nks[0] * nks[1]
    n_units = np.shape(model_nsp)[0]
    # subtract mean and renormalize -- necessary? 
    mn_img = np.mean(x[use,:],axis=0)
    x = x-mn_img
    x = x/np.std(x[use,:],axis =0)
    x = np.append(x,np.ones((nT,1)), axis = 1) # append column of ones
    x = x[use,:]
    # set up prior matrix (regularizer)
    # L2 prior
    Imat = np.eye(nk)
    Imat = linalg.block_diag(Imat,np.zeros((1,1)))
    # smoothness prior
    consecutive = np.ones((nk, 1))
    consecutive[nks[1]-1::nks[1]] = 0
    diff = np.zeros((1,2))
    diff[0,0] = -1
    diff[0,1]= 1
    Dxx = sparse.diags((consecutive @ diff).T, np.array([0, 1]), (nk-1,nk))
    Dxy = sparse.diags((np.ones((nk,1))@ diff).T, np.array([0, nks[1]]), (nk-nks[1], nk))
    Dx = Dxx.T @ Dxx + Dxy.T @ Dxy
    D  = linalg.block_diag(Dx.toarray(),np.zeros((1,1)))      
    # summed prior matrix
    Cinv = D + Imat
    lag_list = [ -4, -2, 0 , 2, 4]
    lambdas = 1024 * (2**np.arange(0,16))
    nlam = len(lambdas)
    # set up empty arrays for receptive field and cross correlation
    sta_all = np.zeros((n_units, len(lag_list), nks[0], nks[1]))
    cc_all = np.zeros((n_units,len(lag_list)))
    # iterate through units
    for celln in tqdm(range(n_units)):
        # iterate through timing lags
        for lag_ind, lag in enumerate(lag_list):
            sps = np.roll(model_nsp[celln,:],-lag)
            sps = sps[use]
            nT = len(sps)
            #split training and test data
            test_frac = 0.3
            ntest = int(nT*test_frac)
            x_train = x[ntest:,:] ; sps_train = sps[ntest:]
            x_test = x[:ntest,:]; sps_test = sps[:ntest]
            #calculate a few terms
            sta = x_train.T@sps_train/np.sum(sps_train)
            XXtr = x_train.T @ x_train
            XYtr = x_train.T @sps_train
            msetrain = np.zeros((nlam,1))
            msetest = np.zeros((nlam,1))
            w_ridge = np.zeros((nk+1,nlam))
            # initial guess
            w = sta
            # loop over regularization strength
            for l in range(len(lambdas)):  
                # calculate MAP estimate               
                w = np.linalg.solve(XXtr + lambdas[l]*Cinv, XYtr) # equivalent of \ (left divide) in matlab
                w_ridge[:,l] = w
                # calculate test and training rms error
                msetrain[l] = np.mean((sps_train - x_train@w)**2)
                msetest[l] = np.mean((sps_test - x_test@w)**2)
            # select best cross-validated lambda for RF
            best_lambda = np.argmin(msetest)
            w = w_ridge[:,best_lambda]
            ridge_rf = w_ridge[:,best_lambda]
            sta_all[celln,lag_ind,:,:] = np.reshape(w[:-1],nks)
            # plot predicted vs actual firing rate
            # predicted firing rate
            sp_pred = x_test@ridge_rf
            # bin the firing rate to get smooth rate vs time
            bin_length = 80
            sp_smooth = (np.convolve(sps_test, np.ones(bin_length), 'same')) / (bin_length * model_dt)
            pred_smooth = (np.convolve(sp_pred, np.ones(bin_length), 'same')) / (bin_length * model_dt)
            # a few diagnostics
            err = np.mean((sp_smooth-pred_smooth)**2)
            cc = np.corrcoef(sp_smooth, pred_smooth)
            cc_all[celln,lag_ind] = cc[0,1]
    # figure of receptive fields
    fig = plt.figure(figsize=(10,np.int(np.ceil(n_units/3))),dpi=50)
    for celln in tqdm(range(n_units)):
        for lag_ind, lag in enumerate(lag_list):
            crange = np.max(np.abs(sta_all[celln,:,:,:]))
            plt.subplot(n_units,6,(celln*6)+lag_ind + 1)  
            plt.imshow(sta_all[celln, lag_ind, :, :], vmin=-crange, vmax=crange, cmap='jet')
            plt.title('cc={:.2f}'.format (cc_all[celln,lag_ind]),fontsize=5)
    return sta_all, cc_all, fig

def make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units,
               this_unit, eyeInterp, worldInterp, accT=None, gz=None, speedT=None, spd=None):
    """
    make a video (without sound)
    INPUTS
        file_dict: dict of files and options from function file_files
        eyeT: timestamps for eyecam
        worldT: timestamps for worldcam
        eye_vid: eyecam video as array
        worldvid: worldcam viedo as array
        contrast: contrast over time
        eye_params: xarray of eye parameters (e.g. theta, phi, X0, etc.)
        dEye: eye velocity over time
        goodcells: ephys dataframe
        units: indexes of all units
        this_unit: unit number to highlight
        eyeInterp: interpolator for eye video
        worldInterp: interpolator for world video
        accT: imu timestamps, if this is a freely moving recording (not plotted if it's not provided)
        gz: gyro z-axis (not plotted if it's not provided)
        speedT: ball optical mouse timestamps (used in place of accT if headfixed)
        spd: ball optical mouse speed (used in place of gz if headfixed)
    OUTPUTS
        vidfile: filepath to the generated video, which is saved out by the function
    """
    # set up figure
    fig = plt.figure(figsize = (8,12))
    gs = fig.add_gridspec(12,4)
    axEye = fig.add_subplot(gs[0:2,0:2])
    axWorld = fig.add_subplot(gs[0:2,2:4])
    axRad = fig.add_subplot(gs[2,:])
    axTheta = fig.add_subplot(gs[3,:])
    axdTheta = fig.add_subplot(gs[4,:])
    axGyro = fig.add_subplot(gs[5,:])
    axContrast = fig.add_subplot(gs[6,:])
    axR = fig.add_subplot(gs[7:12,:])
    # timerange and center frame (only)
    tr = [15, 30]
    fr = np.mean(tr) # time for frame
    eyeFr = np.abs(eyeT-fr).argmin(dim="frame")
    worldFr = np.abs(worldT-fr).argmin(dim="frame")
    # panel for eyecam
    axEye.cla()
    axEye.axis('off')
    axEye.imshow(eye_vid[eyeFr,:,:], 'gray', vmin=0, vmax=255, aspect="equal")
    # panel for worldcam
    axWorld.cla()
    axWorld.axis('off'); 
    axWorld.imshow(world_vid[worldFr,:,:], 'gray', vmin=0, vmax=255, aspect="equal")
    # panel for worldcam contrast
    axContrast.plot(worldT,contrast)
    axContrast.set_xlim(tr[0],tr[1]); axContrast.set_ylim(0,2)
    axContrast.set_ylabel('image contrast')
    # panel for pupil radius
    axRad.cla()
    axRad.plot(eyeT,eye_params.sel(ellipse_params = 'longaxis'))
    axRad.set_xlim(tr[0],tr[1]); 
    axRad.set_ylabel('pupil radius'); axRad.set_xlabel('frame #'); axRad.set_ylim(0,40)
    # panel for horizontal eye position
    axTheta.cla()
    axTheta.plot(eyeT,(eye_params.sel(ellipse_params = 'theta')-np.nanmean(eye_params.sel(ellipse_params = 'theta')))*180/3.14159)
    axTheta.set_xlim(tr[0],tr[1]); 
    axTheta.set_ylabel('theta (deg)'); axTheta.set_ylim(-30,30)
    # panel for eye velocity
    axdTheta.cla()
    axdTheta.plot(eyeT[0:-1],dEye*60); axdTheta.set_ylabel('dtheta')
    axdTheta.set_xlim(tr[0],tr[1]); 
    axdTheta.set_ylim(-900,900); axdTheta.set_ylabel('eye vel (deg/sec)')
    # panel for movement speed
    if file_dict['imu'] is not None:
        # if freely moving, plot gyro z
        axGyro.plot(accT,gz)
        axGyro.set_xlim(tr[0],tr[1]); axGyro.set_ylim(0,5)
        axGyro.set_ylabel('gyro V')
    if file_dict['speed'] is not None:
        # if headfixed, plot balls speed
        axGyro.plot(speedT,spd)
        axGyro.set_xlim(tr[0],tr[1]); axGyro.set_ylim(0,20)
        axGyro.set_ylabel('speed cm/sec')     
    # plot spike raster
    axR.fontsize = 20; n_units = len(goodcells)
    for i,ind in enumerate(goodcells.index):
        axR.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25,'k',linewidth=0.5)
    axR.vlines(goodcells.at[units[this_unit],'spikeT'],this_unit-0.25,this_unit+0.25,'b',linewidth=0.5)
    axR.set_xlim(tr[0],tr[1]); axR.set_ylim(-0.5 , n_units); axR.set_xlabel('secs'); axR.set_ylabel('unit #')
    axR.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)
    plt.tight_layout()
    # path to save video at
    vidfile = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'.mp4'))
    # animate
    writer = FFMpegWriter(fps=30, extra_args=['-vf','scale=800:-2'])
    with writer.saving(fig, vidfile, 100):
        for t in np.arange(tr[0],tr[1],1/30):
            # show eye and world frames
            axEye.cla(); axEye.axis('off'); 
            axEye.imshow(eyeInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            axWorld.cla(); axWorld.axis('off'); 
            axWorld.imshow(worldInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            # plot line for time, then remove
            ln = axR.vlines(t,-0.5,n_units,'b')
            writer.grab_frame()
            ln.remove()
    return vidfile

def make_sound(file_dict, ephys_data, units, this_unit):
    """
    make sound to accompany video
    INPUTS
        file_dict: dict of file names and options for ephys analysis
        ephys_data: ephys data for all units as dataframe
        units: index of all units
        this_unit: unit to highlight for sound
    OUTPUTS
        audfile: filepath to .wav file
    """
    # timerange
    tr = [15, 30]
    # generate wav file
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
    # write the samples to a file
    audfile = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'.wav'))
    wavio.write(audfile, x, datarate, sampwidth=1)
    return audfile

def make_summary_panels(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params,
			dEye, goodcells, units, this_unit, eyeInterp, worldInterp, top_vid,
			topT, topInterp, th, phi, top_speed, accT=None, gz=None, speedT=None, spd=None):
    # set up figure
    fig = plt.figure(figsize = (10,16))
    gs = fig.add_gridspec(12,6)
    axEye = fig.add_subplot(gs[0:2,0:2])
    axWorld = fig.add_subplot(gs[0:2,2:4])
    axTopdown = fig.add_subplot(gs[0:2,4:6])
    axRad = fig.add_subplot(gs[2,:])
    axTh = fig.add_subplot(gs[3,:])
    axGyro = fig.add_subplot(gs[4,:])
    axR = fig.add_subplot(gs[5:12,:])
    
    #timerange and center frame (only)
    tr = [7, 7+15]
    fr = np.mean(tr) # time for frame
    eyeFr = np.abs(eyeT-fr).argmin(dim = "frame")
    worldFr = np.abs(worldT-fr).argmin(dim = "frame")
    topFr = np.abs(topT-fr).argmin(dim = "frame")

    axEye.cla(); axEye.axis('off')
    axEye.imshow(eye_vid[eyeFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")

    axWorld.cla();  axWorld.axis('off'); 
    axWorld.imshow(world_vid[worldFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")

    axTopdown.cla();  axTopdown.axis('off'); 
    axTopdown.imshow(top_vid[topFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    
    axTh.cla()
    axTh.plot(eyeT,th)
    axTh.set_xlim(tr[0],tr[1]); 
    axTh.set_ylabel('theta (deg)'); axTh.set_ylim(-50,0)

    axRad.cla()
    axRad.plot(eyeT,eye_params.sel(ellipse_params='longaxis'))
    axRad.set_xlim(tr[0],tr[1])
    axRad.set_ylabel('pupil radius')
    
    # plot gyro
    axGyro.plot(accT,gz)
    axGyro.set_xlim(tr[0],tr[1]); axGyro.set_ylim(-500,500)
    axGyro.set_ylabel('gyro z (deg/s)')

    # plot spikes
    axR.fontsize = 20
    probe = file_dict['probe_name']
    if '64' in probe:
        sh_num = 2
    elif '128' in probe:
        sh_num = 4
    elif '16' in probe:
        sh_num = 16
    even_raster = np.arange(0,len(goodcells.index),sh_num)
    for i,ind in enumerate(goodcells.index):
        i = (even_raster+(i%32))[int(np.floor(i/32))]
        axR.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25,'k',linewidth=0.5) # all units
    axR.vlines(goodcells.at[units[this_unit],'spikeT'],this_unit-0.25,this_unit+0.25,'k',linewidth=0.5) # this unit
    
    n_units = len(goodcells)
    axR.set_ylim(n_units,-.5)
    axR.set_xlim(tr[0],tr[1]); axR.set_xlabel('secs'); axR.set_ylabel('unit')
    axR.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)
    plt.tight_layout()
    return fig

def run_ephys_analysis(file_dict):
    """
    ephys analysis bringing together eyecam, worldcam, ephys data, imu data, and running ball optical mouse data
    runs on one recording at a time
    saves out an .h5 file for the rec structured as a dict of 
    h5 file is  best read in with pandas, or if pooling data across recordings, and then across sessions, with load_ephys func in /project_analysis/ephys/ephys_utils.py
    INPUTS
        file_dict: dictionary saved out from func find_files
    OUTPUTS
        None
    """
    # set up recording properties
    if file_dict['speed'] is None:
        free_move = True; has_imu = True; has_mouse = False
    else:
        free_move = False; has_imu = False; has_mouse = True
    # delete the existing h5 file, so that a new one can be written
    if os.path.isfile(os.path.join(file_dict['save'], (file_dict['name']+'_ephys_props.h5'))):
        os.remove(os.path.join(file_dict['save'], (file_dict['name']+'_ephys_props.h5')))
    # open three pdfs
    print('opening pdfs')
    overview_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_overview_analysis_figures.pdf')))
    detail_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_detailed_analysis_figures.pdf')))
    diagnostic_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_diagnostic_analysis_figures.pdf')))
    print('opening and resizing worldcam data')
    # open worldcam
    world_data = xr.open_dataset(file_dict['world'])
    world_vid_raw = np.uint8(world_data['WORLD_video'])
    # resize worldcam
    sz = world_vid_raw.shape # raw video size
    # if size is larger than the target 60x80, resize by 0.5
    if sz[1]>160:
        downsamp = 0.5
        world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
        for f in range(sz[0]):
            world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
    else:
        # if the worldcam has already been resized when the nc file was written in preprocessing, don't resize
        world_vid = world_vid_raw.copy()
    # world timestamps
    worldT = world_data.timestamps.copy()
    # plot worldcam timing
    fig, axs = plt.subplots(1,2)
    axs[0].plot(np.diff(worldT)[0:-1:10]); axs[0].set_xlabel('every 10th frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('worldcam timing')
    axs[1].hist(np.diff(worldT),100);axs[1].set_xlabel('deltaT')
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
        # most points don't track well enough for this to be done with other parts of animal (e.g. head points)
        topx = top_data.TOP1_pts.sel(point_loc='tailbase_x').values; topy = top_data.TOP1_pts.sel(point_loc='tailbase_y').values
        topdX = np.diff(topx); topdY = np.diff(topy)
        top_speed = np.sqrt(topdX**2, topdY**2) # speed of tailbase in topdown camera
        topT = top_data.timestamps.copy() # read in time timestamps
        top_vid = np.uint8(top_data['TOP1_video']) # read in top video
        # clear from memory
        del top_data
        gc.collect()
    # load IMU data
    if file_dict['imu'] is not None:
        print('opening imu data')
        imu_data = xr.open_dataset(file_dict['imu'])
        accT = imu_data.IMU_data.sample # imu timestamps
        acc_chans = imu_data.IMU_data # imu dample data
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
        plt.plot(gz_deg[0:100*60])
        plt.title('gyro z (deg)')
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
    # ephys data for this individual recording
    ephys_data = pd.read_json(file_dict['ephys'])
    # sort units by shank and site order
    ephys_data = ephys_data.sort_values(by='ch', axis=0, ascending=True)
    ephys_data = ephys_data.reset_index()
    ephys_data = ephys_data.drop('index', axis=1)
    # spike times
    ephys_data['spikeTraw'] = ephys_data['spikeT']
    print('getting good cells')
    # select good cells from phy2
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    units = goodcells.index.values
    # get number of good units
    n_units = len(goodcells)
    # plot spike raster
    spikeraster_fig = plot_spike_raster(goodcells)
    detail_pdf.savefig()
    plt.close()
    print('opening eyecam data')
    # load eye data
    eye_data = xr.open_dataset(file_dict['eye'])
    eye_vid = np.uint8(eye_data['REYE_video'])
    eyeT = eye_data.timestamps.copy()
    # plot eye timestamps
    fig, axs = plt.subplots(1,2)
    axs[0].plot(np.diff(eyeT)[0:-1:10]); axs[0].set_xlabel('every 10th frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('eyecam timing')
    axs[1].hist(np.diff(eyeT),100);axs[1].set_xlabel('deltaT')
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
    # plot optical mouse speeds
    if file_dict['speed'] is not None:
        fig = plt.figure()
        plt.plot(spd_tstamps,spd)
        plt.xlabel('sec'); plt.ylabel('running speed cm/sec')
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
    # make space in memory
    del eye_data
    gc.collect()
    if file_dict['drop_slow_frames'] is True:
        # in the case that the recording has long time lags, drop data in a window +/- 3 frames around these slow frames
        isfast = np.diff(eyeT)<=0.03
        isslow = sorted(list(set(chain.from_iterable([list(range(int(i)-3,int(i)+4)) for i in np.where(isfast==False)[0]]))))
        th[isslow] = np.nan
        phi[isslow] = np.nan
    # check that deinterlacing worked correctly
    # plot theta and theta_switch
    # want theta_switch to be jagged, theta to be smooth
    theta_switch_fig, th_switch = plot_param_switch_check(eye_params)
    diagnostic_pdf.savefig()
    plt.close()
    # plot eye variables
    fig, axs = plt.subplots(4,1)
    for i,val in enumerate(eye_params.ellipse_params[0:4]):
        axs[i].plot(eyeT[0:-1:10],eye_params.sel(ellipse_params = val)[0:-1:10])
        axs[i].set_ylabel(val.values)
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()
    # calculate eye veloctiy
    dEye = np.diff(th)
    # check accelerometer / eye temporal alignment
    if file_dict['imu'] is not None:
        print('checking accelerometer / eye temporal alignment')
        # plot eye velocity against head movements
        plt.figure
        plt.plot(eyeT[0:-1],-dEye,label='-dEye')
        plt.plot(accTraw,gz_deg,label='gz')
        plt.legend()
        plt.xlim(0,10); plt.xlabel('secs'); plt.ylabel('gyro (deg)')
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
        fig, xmap, ymap = eye_shift_estimation(th, phi, eyeT, world_vid,worldT,60*60)
        xcorrection = xmap.copy()
        ycorrection = ymap.copy()
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
    # std of worldcam image
    fig = plt.figure()
    plt.imshow(std_im)
    plt.colorbar(); plt.title('std img')
    diagnostic_pdf.savefig()
    plt.close()
    # make movie and sound
    this_unit = file_dict['cell']
    # set up interpolators for eye and world videos
    eyeInterp = interp1d(eyeT, eye_vid, axis=0, bounds_error=False)
    worldInterp = interp1d(worldT, world_vid_raw, axis=0, bounds_error=False)
    if free_move:
        topInterp = interp1d(topT, top_vid, axis=0,bounds_error=False)
    if file_dict['imu'] is not None:
        fig = make_summary_panels(file_dict, eyeT, worldT, eye_vid, world_vid_raw, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, top_vid, topT, topInterp, th, phi, top_speed, accT=accT, gz=gz)
        detail_pdf.savefig()
        plt.close()
    if file_dict['mp4']:
        if file_dict['imu'] is not None:
            print('making video figure')
            vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid_raw, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, accT=accT, gz=gz)
        elif file_dict['speed'] is not None:
            print('making video figure')
            vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid_raw, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, speedT=speedT, spd=spd)
        print('making audio figure')
        audfile = make_sound(file_dict, ephys_data, units, this_unit)
        print('merging videos with sound')
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
    plt.imshow(np.mean(world_vid, axis=0), vmin=0, vmax=255)
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
    try:
        plt.figure()
        plt.plot(eyeT,Rnorm)
        #plt.xlim([0,60])
        plt.xlabel('secs')
        plt.ylabel('normalized pupil R')
        diagnostic_pdf.savefig()
        plt.close()
    except:
        pass

    if not free_move:
        # don't run for freely moving, at least for now, because recordings can be too long to fit ephys binary into memory
        # was only a problem for a 128ch recording
        # but hf recordings should be sufficient length to get good estimate
        print('starting continuous LFP laminar depth estimation')
        print('loading ephys binary file')
        # read in ephys binary
        lfp_ephys = read_ephys_bin(file_dict['ephys_bin'], file_dict['probe_name'], do_remap=True, mapping_json=file_dict['mapping_json'])
        print('applying bandpass filter')
        # subtract mean in time dim and apply bandpass filter
        ephys_center_sub = lfp_ephys - np.mean(lfp_ephys,0)
        filt_ephys = butter_bandpass(ephys_center_sub, lowcut=600, highcut=6000, fs=30000, order=6)
        print('getting lfp power profile across channels')
        # get lfp power profile for each channel
        ch_num = np.size(filt_ephys,1)
        lfp_power_profiles = np.zeros([ch_num])
        for ch in range(ch_num):
            lfp_power_profiles[ch] = np.sqrt(np.mean(filt_ephys[:,ch]**2)) # multiunit LFP power profile
        # median filter
        print('applying median filter')
        lfp_power_profiles_filt = medfilt(lfp_power_profiles)
        if file_dict['probe_name'] == 'DB_P64-8':
            ch_spacing = 25/2
        else:
            ch_spacing = 25
        print('making figures')
        if ch_num == 64:
            norm_profile_sh0 = lfp_power_profiles_filt[:32]/np.max(lfp_power_profiles_filt[:32])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            norm_profile_sh1 = lfp_power_profiles_filt[32:64]/np.max(lfp_power_profiles_filt[32:64])
            layer5_cent_sh1 = np.argmax(norm_profile_sh1)
            lfp_power_profiles = [norm_profile_sh0, norm_profile_sh1]
            lfp_layer5_centers = [layer5_cent_sh0, layer5_cent_sh1]
            plt.subplots(1,2)
            plt.tight_layout()
            plt.subplot(1,2,1)
            plt.plot(norm_profile_sh0,range(0,32))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            plt.subplot(1,2,2)
            plt.plot(norm_profile_sh1,range(0,32))
            plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
            plt.title('shank1')
            detail_pdf.savefig(); plt.close()
        elif ch_num == 16:
            norm_profile_sh0 = lfp_power_profiles_filt[:16]/np.max(lfp_power_profiles_filt[:16])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            lfp_power_profiles = [norm_profile_sh0]
            lfp_layer5_centers = [layer5_cent_sh0]
            plt.figure()
            plt.tight_layout()
            plt.plot(norm_profile_sh0,range(0,16))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([17,-1]); plt.yticks(ticks=list(range(-1,17)),labels=(ch_spacing*np.arange(18)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            detail_pdf.savefig(); plt.close()
        elif ch_num == 128:
            norm_profile_sh0 = lfp_power_profiles_filt[:32]/np.max(lfp_power_profiles_filt[:32])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            norm_profile_sh1 = lfp_power_profiles_filt[32:64]/np.max(lfp_power_profiles_filt[32:64])
            layer5_cent_sh1 = np.argmax(norm_profile_sh1)
            norm_profile_sh2 = lfp_power_profiles_filt[64:96]/np.max(lfp_power_profiles_filt[64:96])
            layer5_cent_sh2 = np.argmax(norm_profile_sh2)
            norm_profile_sh3 = lfp_power_profiles_filt[96:128]/np.max(lfp_power_profiles_filt[96:128])
            layer5_cent_sh3 = np.argmax(norm_profile_sh3)
            lfp_power_profiles = [norm_profile_sh0, norm_profile_sh1, norm_profile_sh2, norm_profile_sh3]
            lfp_layer5_centers = [layer5_cent_sh0, layer5_cent_sh1, layer5_cent_sh2, layer5_cent_sh3]
            plt.subplots(1,4)
            plt.tight_layout()
            plt.subplot(1,4,1)
            plt.plot(norm_profile_sh0,range(0,32))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            plt.subplot(1,4,2)
            plt.plot(norm_profile_sh1,range(0,32))
            plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
            plt.title('shank1')
            plt.subplot(1,4,3)
            plt.plot(norm_profile_sh2,range(0,32))
            plt.plot(norm_profile_sh2[layer5_cent_sh2]+0.01,layer5_cent_sh2,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh2*ch_spacing)))
            plt.title('shank2')
            plt.subplot(1,4,4)
            plt.plot(norm_profile_sh3,range(0,32))
            plt.plot(norm_profile_sh3[layer5_cent_sh3]+0.01,layer5_cent_sh3,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh3*ch_spacing)))
            plt.title('shank3')
            detail_pdf.savefig(); plt.close()

    if file_dict['stim_type'] == 'revchecker':
        print('running revchecker analysis')
        print('loading ephys binary file and applying filters')
        # read in the binary file of ephys recording
        lfp_ephys = read_ephys_bin(file_dict['ephys_bin'], file_dict['probe_name'], do_remap=True, mapping_json=file_dict['mapping_json'])
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
        if '64' in file_dict['probe_name']:
            num_channels = 64
            colors = plt.cm.jet(np.linspace(0,1,32))
        elif '128' in file_dict['probe_name']:
            num_channels = 128
            colors = plt.cm.jet(np.linspace(0,1,32))
        elif '16' in file_dict['probe_name']:
            num_channels = 16
            colors = plt.cm.jet(np.linspace(0,1,16))
        # plot traces for shanks
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
        elif num_channels == 16:
            plt.figure()
            for ch_num in np.arange(0,16):
                plt.plot(rev_resp_mean[ch_num], color=colors[ch_num], linewidth=1)
                plt.axvline(x=(0.1*samprate))
                plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
            detail_pdf.savefig(); plt.close()
        elif num_channels == 128:
            plt.subplots(1,4 ,figsize=(40,6))
            for ch_num in np.arange(0,128):
                if ch_num < 32:
                    plt.subplot(1,4,1)
                    plt.plot(rev_resp_mean[ch_num], color=colors[ch_num], linewidth=1)
                    plt.title('ch1:32'); plt.axvline(x=(0.1*samprate))
                    plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                elif 32 <= ch_num < 64:
                    plt.subplot(1,4,2)
                    plt.plot(rev_resp_mean[ch_num], color=colors[ch_num-32], linewidth=1)
                    plt.title('ch33:64'); plt.axvline(x=(0.1*samprate))
                    plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                elif 64 <= ch_num < 96:
                    plt.subplot(1,4,3)
                    plt.plot(rev_resp_mean[ch_num], color=colors[ch_num-64], linewidth=1)
                    plt.title('ch33:64'); plt.axvline(x=(0.1*samprate))
                    plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                elif 96 <= ch_num < 128:
                    plt.subplot(1,4,4)
                    plt.plot(rev_resp_mean[ch_num], color=colors[ch_num-96], linewidth=1)
                    plt.title('ch33:64'); plt.axvline(x=(0.1*samprate))
                    plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
            detail_pdf.savefig(); plt.close()
        if num_channels == 64:
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
        if num_channels == 128:
            # channels arranged in columns
            fig, axes = plt.subplots(int(np.size(rev_resp_mean,0)/4),4, figsize=(7,20),sharey=True)
            ch_num = 0
            for ax in axes.T.flatten():
                ax.plot(rev_resp_mean[ch_num], linewidth=1)
                ax.axvline(x=(0.1*samprate), linewidth=1)
                ax.axis('off')
                ax.set_title(ch_num)
                ch_num = ch_num + 1
            detail_pdf.savefig(); plt.close()
        if num_channels == 16:
            # channels arranged in columns
            fig, axes = plt.subplots(int(np.size(rev_resp_mean,0)),1, figsize=(7,20),sharey=True)
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
        elif num_channels == 128:
            shank0_layer4cent = np.argmin(np.min(rev_resp_mean[0:32,int(samprate*0.1):int(samprate*0.3)], axis=1))
            shank1_layer4cent = np.argmin(np.min(rev_resp_mean[32:64,int(samprate*0.1):int(samprate*0.3)], axis=1))
            shank2_layer4cent = np.argmin(np.min(rev_resp_mean[64:96,int(samprate*0.1):int(samprate*0.3)], axis=1))
            shank3_layer4cent = np.argmin(np.min(rev_resp_mean[96:128,int(samprate*0.1):int(samprate*0.3)], axis=1))
            shank0_ch_positions = list(range(32)) - shank0_layer4cent; shank1_ch_positions = list(range(32)) - shank1_layer4cent
            shank2_ch_positions = list(range(32)) - shank2_layer4cent; shank3_ch_positions = list(range(32)) - shank3_layer4cent
            lfp_depth = [shank0_ch_positions, shank1_ch_positions, shank2_ch_positions, shank3_ch_positions]
            layer4_out = [shank0_layer4cent, shank1_layer4cent, shank2_layer4cent, shank3_layer4cent]

    if file_dict['stim_type'] == 'grat':
        print('getting grating flow')
        nf = np.size(img_norm, 0) - 1
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
            # ax.cla()
            # ax.imshow(frm,vmin = 0, vmax = 255)
            u = flow_norm[f,:,:,0]; v = -flow_norm[f,:,:,1]  # negative to fix sign for y axis in images
            sx = cv2.Sobel(frm,cv2.CV_64F,1,0,ksize=11)
            sy = -cv2.Sobel(frm,cv2.CV_64F,0,1,ksize=11)# negative to fix sign for y axis in images
            sx[std_im<20]=0; sy[std_im<20]=0; # get rid of values outside of monitor
            sy[sx<0] = -sy[sx<0]  # make vectors point in positive x direction (so opposite sides of grating don't cancel)
            sx[sx<0] = -sx[sx<0]
            # sy[np.abs(sx)<500000] = np.abs(sy[np.abs(sx)<500000]) # deals with horitzontal cases - flips them right side up
            sy[np.abs(sx/sy)<0.15] = np.abs(sy[np.abs(sx/sy)<0.15])
            # ax.quiver(x[::nx,::nx],y[::nx,::nx],sx[::nx,::nx],sy[::nx,::nx], scale = 100000 )
            # u_mn[f]= np.mean(u); v_mn[f]= np.mean(v); sx_mn[f] = np.mean(sx); sy_mn[f] = np.mean(sy)
            u_mn[f]= np.mean(u[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg]); v_mn[f]= np.mean(v[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg]); 
            sx_mn[f] = np.mean(sx[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg]); sy_mn[f] = np.mean(sy[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg])
        scr_contrast = np.empty(worldT.size)
        for i in range(worldT.size):
            scr_contrast[i] = np.nanmean(np.abs(img_norm[i,ycent-25:ycent+25,xcent-40:xcent+40]))
        scr_contrast = signal.medfilt(scr_contrast,11)
        stimOn = np.double(scr_contrast>0.5)
        stim_start = np.array(worldT[np.where(np.diff(stimOn)>0)])
        psth_fig, grating_psth = plot_psth(goodcells, stim_start, -0.5, 1.5, 0.1)
        plt.title('grating psth')
        detail_pdf.savefig(); plt.close()
        stim_end = np.array(worldT[np.where(np.diff(stimOn)<0)])
        stim_end = stim_end[stim_end>stim_start[0]]
        stim_start = stim_start[stim_start<stim_end[-1]]
        grating_th = np.zeros(len(stim_start))
        grating_mag = np.zeros(len(stim_start))
        grating_dir = np.zeros(len(stim_start))
        dI = np.zeros(len(stim_start))
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
            dI[i] = np.mean(np.diff(img_norm[tpts,ycent,xcent])**2)  # rate of change of image give temporal frequency
        grating_ori = grating_th.copy()
        grating_ori[grating_dir<0] = grating_ori[grating_dir<0] + np.pi
        grating_ori = grating_ori - np.min(grating_ori)
        np.unique(grating_ori)
        grating_tf = np.zeros(len(stim_start))
        grating_tf[dI>0.5] = 1;  # 0 = low sf; 1 = hi sf
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
        ori_tuning_tf = np.zeros((len(goodcells),8,3,2))
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
                    ori_tuning[c,ori,sf] = np.mean(grating_rate[c,(ori_cat==ori) & (sf_cat==sf)])
                    for tf in range(2):
                        ori_tuning_tf[c,ori,sf,tf] = np.mean(grating_rate[c,(ori_cat==ori) & (sf_cat ==sf) & (grating_tf==tf)])
            drift_spont[c] = np.mean(spont_rate[c,:])
            plt.subplot(n_units,4,4*c+1)
            plt.scatter(grating_ori,grating_rate[c,:],c=sf_cat)
            plt.plot(3*np.ones(len(spont_rate[c,:])),spont_rate[c,:],'r.')
            plt.subplot(n_units,4,4*c+2)
            plt.plot(ori_tuning[c,:,0],label = 'low sf'); plt.plot(ori_tuning[c,:,1],label = 'mid sf');plt.plot(ori_tuning[c,:,2],label = 'hi sf')
            plt.plot([0,7],[drift_spont[c],drift_spont[c]],'r:', label = 'spont')
            try:
                plt.ylim(0,np.nanmax(ori_tuning_tf[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1) 
            plt.subplot(n_units,4,4*c+3)
            plt.plot(ori_tuning_tf[c,:,0,0],label = 'low sf'); plt.plot(ori_tuning_tf[c,:,1,0],label = 'mid sf');plt.plot(ori_tuning_tf[c,:,2,0],label = 'hi sf')
            plt.plot([0,7],[drift_spont[c],drift_spont[c]],'r:', label = 'spont')
            try:
                plt.ylim(0,np.nanmax(ori_tuning_tf[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.subplot(n_units,4,4*c+4)
            plt.plot(ori_tuning_tf[c,:,0,1],label = 'low sf'); plt.plot(ori_tuning_tf[c,:,1,1],label = 'mid sf');plt.plot(ori_tuning_tf[c,:,2,1],label = 'hi sf')
            plt.plot([0,7],[drift_spont[c],drift_spont[c]],'r:', label = 'spont')
            try:
                plt.ylim(0,np.nanmax(ori_tuning_tf[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
        plt.legend()
        detail_pdf.savefig()
        plt.close()
        del eyeInterp, worldInterp
        gc.collect()
    # create interpolator for movie data so we can evaluate at same timebins are firing rate
    # img_norm[img_norm<-2] = -2
    sz = np.shape(img_norm); downsamp = 0.5
    img_norm_sm = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)))
    for f in range(sz[0]):
        img_norm_sm[f,:,:] = cv2.resize(img_norm[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
    movInterp = interp1d(worldT, img_norm_sm, axis=0, bounds_error=False)
    # get channel number
    if '16' in file_dict['probe_name']:
        ch_num = 16
    elif '64' in file_dict['probe_name']:
        ch_num = 64
    elif '128' in file_dict['probe_name']:
        ch_num = 128
    print('getting STA for single lag')
    # calculate spike-triggered average
    staAll, STA_single_lag_fig = plot_STA(goodcells, img_norm_sm, worldT, movInterp, ch_num, lag=2)
    detail_pdf.savefig()
    plt.close()
    print('getting STA for range in lags')
    # calculate spike-triggered average
    fig = plot_STA(goodcells, img_norm_sm, worldT, movInterp, ch_num, lag=np.arange(-2,8,2))
    detail_pdf.savefig()
    plt.close()
    print('getting STV')
    # calculate spike-triggered variance
    st_var, fig = plot_STV(goodcells, t, movInterp, img_norm_sm)
    detail_pdf.savefig()
    plt.close()

    if (free_move is True) | (file_dict['stim_type'] == 'white_noise'):
        print('doing GLM receptive field estimate')
        # simplified setup for GLM
        # these are general parameters (spike rates, eye position)
        n_units = len(goodcells)
        print('get timing')
        model_dt = 0.025
        model_t = np.arange(0,np.max(worldT),model_dt)
        model_nsp = np.zeros((n_units,len(model_t)))
        # get spikes / rate
        print('get spikes')
        bins = np.append(model_t,model_t[-1]+model_dt)
        for i,ind in enumerate(goodcells.index):
            model_nsp[i,:],bins = np.histogram(goodcells.at[ind,'spikeT'],bins)
        # get eye position
        print('get eye')
        thInterp = interp1d(eyeT,th, bounds_error = False)
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
        # get video ready for GLM
        downsamp = 0.25
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
            smallvid = smallvid[5:-5,5:-5]
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

    print('plotting head and eye movements')
    # calculate saccade-locked psth
    spike_corr = 1 # correction factor for ephys timing drift
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
        
        plt.figure()
        plt.plot(dEye[0:-1:10],dhead(eyeT[0:-1:10]),'.')
        plt.xlabel('dEye'); plt.ylabel('dHead')
        plt.xlim((-10,10)); plt.ylim((-10,10))
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
        upsacc = eyeT[(np.append(dEye,0)>sthresh) & (np.append(dgz,0)>sthresh)]
        downsacc = eyeT[(np.append(dEye,0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
        upsacc_avg_gaze_shift_dEye, downsacc_avg_gaze_shift_dEye, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('gaze shift dEye');  detail_pdf.savefig() ;  plt.close()
        # plot compensatory eye movements    
        sthresh = 3
        upsacc = eyeT[(np.append(dEye,0)>sthresh) & (np.append(dgz,0)<1)]
        downsacc = eyeT[(np.append(dEye,0)<-sthresh) & (np.append(dgz,0)>-1)]
        upsacc_avg_comp_dEye, downsacc_avg_comp_dEye, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('comp dEye'); detail_pdf.savefig() ;  plt.close()
        # plot gaze shifting head movements
        sthresh = 3
        upsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)>sthresh) & (np.append(dgz,0)>sthresh)]
        downsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
        upsacc_avg_gaze_shift_dHead, downsacc_avg_gaze_shift_dHead, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('gaze shift dhead') ; detail_pdf.savefig() ;  plt.close()
        # plot compensatory eye movements    
        sthresh = 3
        upsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)>sthresh) & (np.append(dgz,0)<1)]
        downsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)<-sthresh) & (np.append(dgz,0)>-1)]
        upsacc_avg_comp_dHead, downsacc_avg_comp_dHead, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('comp dhead') ; detail_pdf.savefig() ;  plt.close()

    # normalize and plot eye radius
    eyeR = eye_params.sel(ellipse_params='longaxis').copy()
    Rnorm = (eyeR-np.mean(eyeR)) / np.std(eyeR)
    plt.figure()
    plt.plot(eyeT,Rnorm)
    plt.xlabel('secs')
    plt.ylabel('normalized pupil R')
    diagnostic_pdf.savefig()
    plt.close()

    print('plotting spike rate vs pupil radius and position')
    # plot rate vs pupil
    R_range = np.linspace(10,50,10)
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

    eyePhi = eye_params.sel(ellipse_params='phi').copy()
    phiNorm = (eyePhi-np.mean(eyePhi)) / np.std(eyePhi)

    print('plotting spike rate vs theta/phi')
    # plot rate vs theta
    th_range = np.linspace(-30,30,10)
    spike_rate_vs_theta_cent, spike_rate_vs_theta_tuning, spike_rate_vs_theta_err, spike_rate_vs_theta_fig = plot_spike_rate_vs_var(th, th_range, goodcells, eyeT, t, 'eye theta')
    detail_pdf.savefig()
    plt.close()
    phi_range = np.linspace(-30,30,10)
    spike_rate_vs_phi_cent, spike_rate_vs_phi_tuning, spike_rate_vs_phi_err, spike_rate_vs_phi_fig = plot_spike_rate_vs_var(phi, phi_range, goodcells, eyeT, t, 'eye phi')
    detail_pdf.savefig()
    plt.close()
    
    if free_move is True:
        print('plotting spike rate vs gyro and speed')
        # get active times only
        active_interp = interp1d(model_t, model_active, bounds_error=False)
        active_accT = active_interp(accT.values)
        use = np.where(active_accT > 40)
        # spike rate vs gyro x
        gx_range = np.linspace(-5,5,10)
        active_gx = ((gx-np.mean(gx))*7.5)[use]
        spike_rate_vs_gx_cent, spike_rate_vs_gx_tuning, spike_rate_vs_gx_err, spike_rate_vs_gx_fig = plot_spike_rate_vs_var(active_gx, gx_range, goodcells, accT[use], t, 'gyro x')
        detail_pdf.savefig()
        plt.close()
        # spike rate vs gyro y
        gy_range = np.linspace(-5,5,10)
        active_gy = ((gy-np.mean(gy))*7.5)[use]
        spike_rate_vs_gy_cent, spike_rate_vs_gy_tuning, spike_rate_vs_gy_err, spike_rate_vs_gy_fig = plot_spike_rate_vs_var(active_gy, gy_range, goodcells, accT[use], t, 'gyro y')
        detail_pdf.savefig()
        plt.close()
        # spike rate vs gyro z
        gz_range = np.linspace(-7,7,10)
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
        roll_range = np.linspace(-30,30,10)
        spike_rate_vs_roll_cent, spike_rate_vs_roll_tuning, spike_rate_vs_roll_err, spike_rate_vs_roll_fig = plot_spike_rate_vs_var(groll[use], roll_range, goodcells, accT[use], t, 'roll')
        detail_pdf.savefig()
        plt.close()
        # pitch vs spike rate
        pitch_range = np.linspace(-30,30,10)
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

    print('making overview plot')
    if file_dict['stim_type'] == 'grat':
        summary_fig = plot_overview(goodcells, crange, resp, file_dict, staAll, trange, upsacc_avg, downsacc_avg, ori_tuning=ori_tuning, drift_spont=drift_spont, grating_ori=grating_ori, sf_cat=sf_cat, grating_rate=grating_rate, spont_rate=spont_rate)
    else:
        summary_fig = plot_overview(goodcells, crange, resp, file_dict, staAll, trange, upsacc_avg, downsacc_avg)
    overview_pdf.savefig()
    plt.close()

    print('making summary plot')
    hist_dt = 1
    hist_t = np.arange(0, np.max(worldT),hist_dt)
    plt.subplots(n_units+3,1,figsize=(8,int(np.ceil(n_units/3))))
    plt.tight_layout()
    # either gyro or optical mouse reading
    plt.subplot(n_units+3,1,1)
    if has_imu:
        plt.plot(accT,gz)
        plt.xlim(0, np.max(worldT)); plt.ylabel('gz'); plt.title('gyro')
    elif has_mouse:
        plt.plot(speedT,spd)
        plt.xlim(0, np.max(worldT)); plt.ylabel('cm/sec'); plt.title('mouse speed')  
    # pupil diameter
    plt.subplot(n_units+3,1,2)
    plt.plot(eyeT,eye_params.sel(ellipse_params = 'longaxis'))
    plt.xlim(0, np.max(worldT)); plt.ylabel('rad (pix)'); plt.title('pupil diameter')
    # worldcam contrast
    plt.subplot(n_units+3,1,3)
    plt.plot(worldT,contrast)
    plt.xlim(0, np.max(worldT)); plt.ylabel('contrast a.u.'); plt.title('contrast')
    # raster
    for i,ind in enumerate(goodcells.index):
        rate,bins = np.histogram(ephys_data.at[ind,'spikeT'],hist_t)
        plt.subplot(n_units+3,1,i+4)
        plt.plot(bins[0:-1],rate)
        plt.xlim(bins[0],bins[-1]); plt.ylabel('unit ' + str(ind))
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
                                        'ori_tuning_tf',
                                        'ori_tuning',
                                        'drift_spont',
                                        'spont_rate',
                                        'grating_rate',
                                        'sf_cat',
                                        'trange',
                                        'theta',
                                        'phi',
                                        'spike_rate_vs_spd_cent',
                                        'spike_rate_vs_spd_tuning',
                                        'spike_rate_vs_spd_err',
                                        'spike_rate_vs_phi_cent',
                                        'spike_rate_vs_phi_tuning',
                                        'spike_rate_vs_phi_err',
                                        'lfp_power_profiles',
                                        'lfp_layer5_centers']]
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
                                    ori_tuning_tf[unit_num],
                                    drift_spont[unit_num],
                                    spont_rate[unit_num],
                                    grating_rate[unit_num],
                                    sf_cat[unit_num],
                                    trange,
                                    th,
                                    phi,
                                    spike_rate_vs_spd_cent,
                                    spike_rate_vs_spd_tuning[unit_num],
                                    spike_rate_vs_spd_err[unit_num],
                                    spike_rate_vs_phi_cent,
                                    spike_rate_vs_phi_tuning[unit_num],
                                    spike_rate_vs_phi_err[unit_num],
                                    lfp_power_profiles,
                                    lfp_layer5_centers]),dtype=object).T
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
                                        'layer4center',
                                        'lfp_power_profiles',
                                        'lfp_layer5_centers']]
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
                                    layer4_out,
                                    lfp_power_profiles,
                                    lfp_layer5_centers]),dtype=object).T
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
                                        'spike_rate_vs_phi_err',
                                        'lfp_power_profiles',
                                        'lfp_layer5_centers']]
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
                                    spike_rate_vs_phi_err[unit_num],
                                    lfp_power_profiles,
                                    lfp_layer5_centers]),dtype=object).T
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
                                        'spike_rate_vs_phi_err',
                                        'lfp_power_profiles',
                                        'lfp_layer5_centers']]
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
                                    spike_rate_vs_phi_err[unit_num],
                                    lfp_power_profiles,
                                    lfp_layer5_centers]),dtype=object).T
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

def load_ephys(csv_filepath):
    """
    using a .csv file of metadata identical to the one used to run batch analysis, pool experiments marked for inclusion and orgainze properties
    saved out from ephys analysis into .h5 files as columns and each unit as an index
    also reads in the .json of calibration properties saved out from fm recording eyecam analysis so that filtering can be done based on how well the eye tracking worked
    INPUTS
        csv_filepath: path to csv file used for batch analysis
    OUTPUTS
        all_data: DataFrame of all units marked for pooled analysis, with each index representing a unit across all recordings of a session
    """
    # open the csv file of metadata and pull out all of the desired data paths
    csv = pd.read_csv(csv_filepath)
    for_data_pool = csv[csv['load_for_data_pool'] == any(['TRUE' or True or 'True'])]
    goodsessions = []
    probenames_for_goodsessions = []
    layer5_depth_for_goodsessions = []
    # get all of the best freely moving recordings of a session into a dictionary
    goodlightrecs = dict(zip(list([j+'_'+i for i in [i.split('\\')[-1] for i in for_data_pool['animal_dirpath']] for j in [datetime.strptime(i,'%m/%d/%y').strftime('%m%d%y') for i in list(for_data_pool['experiment_date'])]]),[i if i !='' else 'fm1' for i in for_data_pool['best_light_fm']]))
    gooddarkrecs = dict(zip(list([j+'_'+i for i in [i.split('\\')[-1] for i in for_data_pool['animal_dirpath']] for j in [datetime.strptime(i,'%m/%d/%y').strftime('%m%d%y') for i in list(for_data_pool['experiment_date'])]]),[i if i !='' else None for i in for_data_pool['best_dark_fm']]))
    # change paths to work with linux
    if platform.system() == 'Linux':
        for ind, row in for_data_pool.iterrows():
            drive = [row['drive'] if row['drive'] == 'nlab-nas' else row['drive'].capitalize()][0]
            for_data_pool.loc[ind,'animal_dirpath'] = os.path.expanduser('~/'+('/'.join([row['computer'].title(), drive] + list(filter(None, row['animal_dirpath'].replace('\\','/').split('/')))[2:])))
    for ind, row in for_data_pool.iterrows():
        goodsessions.append(row['animal_dirpath'])
        probenames_for_goodsessions.append(row['probe_name'])
        layer5_depth_for_goodsessions.append(row['overwrite_layer5center'])
    # get the .h5 files from each day
    # this will be a list of lists, where each list inside of the main list has all the data of a single session
    sessions = [find('*_ephys_props.h5',session) for session in goodsessions]
    # read the data in and append them into one shared df
    all_data = pd.DataFrame([])
    ind = 0
    for session in sessions:
        session_data = pd.DataFrame([])
        for recording in session:
            rec_data = pd.read_hdf(recording)
            # get name of the current recording (i.e. 'fm' or 'hf1_wn')
            rec_type = '_'.join(([col for col in rec_data.columns.values if 'trange' in col][0]).split('_')[:-1])
            # rename spike time columns so that data is retained for each of the seperate trials
            rec_data = rec_data.rename(columns={'spikeT':rec_type+'_spikeT', 'spikeTraw':rec_type+'_spikeTraw','rate':rec_type+'_rate','n_spikes':rec_type+'_n_spikes'})
            # add a column for which fm recording should be prefered
            for key,val in goodlightrecs.items():
                if key in rec_data['session'].iloc[0]:
                    rec_data['best_light_fm'] = val
            for key,val in gooddarkrecs.items():
                if key in rec_data['session'].iloc[0]:
                    rec_data['best_dark_fm'] = val
            # get column names
            column_names = list(session_data.columns.values) + list(rec_data.columns.values)
            # new columns for same unit within a session
            session_data = pd.concat([session_data, rec_data],axis=1,ignore_index=True)
            # add the list of column names from all sessions plus the current recording
            session_data.columns = column_names
            # remove duplicate columns (i.e. shared metadata)
            session_data = session_data.loc[:,~session_data.columns.duplicated()]
        # add probe name as new col
        animal = goodsessions[ind]
        ellipse_json_path = find('*fm_eyecameracalc_props.json', animal)
        if ellipse_json_path != []:
            with open(ellipse_json_path[0]) as f:
                ellipse_fit_params = json.load(f)
            session_data['best_ellipse_fit_m'] = ellipse_fit_params['regression_m']
            session_data['best_ellipse_fit_r'] = ellipse_fit_params['regression_r']
        else:
            print(ellipse_json_path)
        # add probe name
        session_data['probe_name'] = probenames_for_goodsessions[ind]
        # replace LFP power profile estimate of laminar depth with value entered into spreadsheet
        try:
            manual_depth_entry = layer5_depth_for_goodsessions[ind]
            num_auto_depth_entries = len(session_data['hf1_wn_lfp_layer5_centers'].iloc[-1])
            if type(manual_depth_entry) != np.nan and manual_depth_entry != '?' and manual_depth_entry != '' and manual_depth_entry != 'FALSE':
                session_data['hf1_wn_lfp_layer5_centers'] = list(np.ones(num_auto_depth_entries).astype(int)*int(manual_depth_entry))
        except Exception as e:
            print('error with overwriting depth for ', rec_data['session'])
            print(e)
        ind += 1
        # new rows for units from different mice or sessions
        all_data = pd.concat([all_data, session_data], axis=0)
    fm2_light = [c for c in all_data.columns.values if 'fm2_light' in c]
    fm1_dark = [c for c in all_data.columns.values if 'fm1_dark' in c]
    dark_dict = dict(zip(fm1_dark, [i.replace('fm1_dark', 'fm_dark') for i in fm1_dark]))
    light_dict = dict(zip(fm2_light, [i.replace('fm2_light_', 'fm1_') for i in fm2_light]))
    all_data = all_data.rename(dark_dict, axis=1).rename(light_dict, axis=1)
    # drop data without session name
    for ind, row in all_data.iterrows():
        if type(row['session']) != str:
            all_data = all_data.drop(ind, axis=0)
    # combine columns where one property of the unit is spread across multiple columns because of renaming scheme
    for col in list(all_data.loc[:,all_data.columns.duplicated()].columns.values):
        all_data[col] = all_data[col].iloc[:,0].combine_first(all_data[col].iloc[:,1])
    # and drop the duplicates that have only partial data (all the data will now be in another column)
    all_data = all_data.loc[:,~all_data.columns.duplicated()]
    return all_data

def session_ephys_analysis(config):
    """
    run ephys analysis on a full session, finding all recordings in that session and organizing analysis options
    INPUTS
        config: options dict
    OUTPUTS
        None
    """
    # get options out
    data_path = config['animal_dir']
    unit = config['ephys_analysis']['unit_to_highlight']
    probe_name = config['ephys_analysis']['probe_type']
    # get subdirectories (i.e. name of each recording for this session)
    dirnames = list_subdirs(data_path)
    recording_names = sorted([i for i in dirnames if 'hf' in i or 'fm' in i])
    if config['ephys_analysis']['recording_list'] != []:
        recording_names = [i for i in recording_names if i in config['ephys_analysis']['recording_list']]
    # iterate through each recording's name
    for recording_name in recording_names:
        try:
            print('starting ephys analysis for',recording_name,'in path',data_path)
            if 'fm' in recording_name:
                fm = True
            elif 'fm' not in recording_name:
                fm = False
            this_unit = int(unit)
            if fm == True and 'light' in recording_name:
                stim_type = 'light_arena'
            elif fm == True and 'dark' in recording_name:
                stim_type = 'dark_arena'
            elif fm == True and 'light' not in recording_name and 'dark' not in recording_name:
                stim_type = 'light'
            elif 'wn' in recording_name:
                stim_type = 'white_noise'
            elif 'grat' in recording_name:
                stim_type = 'gratings'
            elif 'noise' in recording_name:
                stim_type = 'sparse_noise'
            elif 'revchecker' in recording_name:
                stim_type = 'revchecker'
            recording_path = os.path.join(data_path, recording_name)
            norm_recording_path = os.path.normpath(recording_path).replace('\\', '/')
            full_recording_name = '_'.join(norm_recording_path.split('/')[-3:-1])+'_control_Rig2_'+os.path.split(norm_recording_path)[1].split('/')[-1]
            mp4 = config['ephys_analysis']['write_videos']
            drop_slow_frames = config['parameters']['drop_slow_frames']
            file_dict = find_files(recording_path, full_recording_name, fm, this_unit, stim_type, mp4, probe_name, drop_slow_frames)
            run_ephys_analysis(file_dict)
        except:
            print(traceback.format_exc())