

import os
import json
from glob import glob
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import cv2

import scipy.signal
import scipy.interpolate

import fmEphys


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

    channel_map_path = os.path.join(os.path.split(__file__)[0], 'probes.json')

    # open channel map file
    with open(channel_map_path, 'r') as fp:
        all_maps = json.load(fp)
        
    ch_map = all_maps[probe_name]['map']
    ch_num = all_maps[probe_name]['nCh']

    # set up data types to read binary file into
    dtypes = np.dtype([("ch"+str(i),np.uint16) for i in range(0,ch_num)])
    # read in binary file
    ephys = pd.DataFrame(np.fromfile(binary_path, dtypes, -1, ''))
    # remap with known order of channels
    if do_remap is True:
        ephys = ephys.iloc[:,[i-1 for i in list(ch_map)]]
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
    sos = scipy.signal.butter(order, [low, high], btype='bandpass', output='sos')
    return scipy.signal.sosfiltfilt(sos, data, axis=0)

def open_time(path, dlc_len=None, force_shift=False):
    """ Read in the timestamps for a camera and adjust to deinterlaced video length if needed
    Parameters:
    path (str): path to a timestamp .csv file
    dlc_len (int): number of frames in the DLC data (used to decide if interpolation is needed, but this can be left as None to ignore)
    force_shift (bool): whether or not to interpolate timestamps without checking
    
    Returns:
    time_out (np.array): timestamps as numpy
    """
    # read in the timestamps if they've come directly from cameras
    read_time = pd.read_csv(path, encoding='utf-8', engine='c', header=None).squeeze()
    if read_time[0] == 0: # in case header == 0, which is true of some files, drop that header which will have been read in as the first entry  
        read_time = read_time[1:]
    time_in = []
    fmt = '%H:%M:%S.%f'
    if read_time.dtype!=np.float64:
        for current_time in read_time:
            currentT = str(current_time).strip()
            try:
                t = datetime.strptime(currentT,fmt)
            except ValueError as v:
                ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                if ulr:
                    currentT = currentT[:-ulr]
            try:
                time_in.append((datetime.strptime(currentT, '%H:%M:%S.%f') - datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')).total_seconds())
            except ValueError:
                time_in.append(np.nan)
        time_in = np.array(time_in)
    else:
        time_in = read_time.values

    # auto check if vids were deinterlaced
    if dlc_len is not None:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing if needed
        timestep = np.nanmedian(np.diff(time_in, axis=0))
        if dlc_len > len(time_in):
            time_out = np.zeros(np.size(time_in, 0)*2)
            # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
            time_out[::2] = time_in - 0.25 * timestep
            time_out[1::2] = time_in + 0.25 * timestep
        elif dlc_len == len(time_in):
            time_out = time_in
        elif dlc_len < len(time_in):
            time_out = time_in
    elif dlc_len is None:
        time_out = time_in

    # force the times to be shifted if the user is sure it should be done
    if force_shift is True:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing
        timestep = np.nanmedian(np.diff(time_in, axis=0))
        time_out = np.zeros(np.size(time_in, 0)*2)
        # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
        time_out[::2] = time_in - 0.25 * timestep
        time_out[1::2] = time_in + 0.25 * timestep

    return time_out

def plot_prelim_STA(spikeT, img_norm, worldT, movInterp, ch_count, lag=2):
    n_units = len(spikeT)
    # model setup
    model_dt = 0.025
    model_t = np.arange(0, np.nanmax(worldT), model_dt)
    model_nsp = np.zeros((n_units, len(model_t)))
    # get binned spike rate
    bins = np.append(model_t, model_t[-1]+model_dt)
    for i in range(n_units):
        model_nsp[i,:], bins = np.histogram(spikeT[i], bins)
    # settting up video
    nks = np.shape(img_norm[0,:,:])
    nk = nks[0]*nks[1]
    model_vid = np.zeros((len(model_t),nk))
    for i in range(len(model_t)):
        model_vid[i,:] = np.reshape(movInterp(model_t[i]+model_dt/2), nk)
    # spike-triggered average
    staAll = np.zeros((n_units, np.shape(img_norm)[1], np.shape(img_norm)[2]))
    model_vid[np.isnan(model_vid)] = 0
    fig = plt.figure(figsize=(20,np.ceil(n_units/2)))
    for c in range(n_units):
        sp = model_nsp[c,:].copy()
        sp = np.roll(sp, -lag)
        sta = model_vid.T @ sp
        sta = np.reshape(sta, nks)
        nsp = np.sum(sp)
        plt.subplot(int(np.ceil(n_units/10)),10,c+1)
        if nsp > 0:
            sta = sta/nsp
            # flip matrix so that physical top is at the top (worldcam comes in upsidedown)
            sta = np.fliplr(np.flipud(sta))
        else:
            sta = np.nan
        if pd.isna(sta) is True:
            plt.imshow(np.zeros([120,160]))
        else:
            starange = np.max(np.abs(sta))*1.1
            plt.imshow((sta-np.mean(sta)), vmin=-starange, vmax=starange, cmap='jet')
            staAll[c,:,:] = sta
    plt.tight_layout()
    return staAll, fig

def prelimRF_raw(whitenoise_directory, probe):
    print('finding files')
    world_file = glob(os.path.join(whitenoise_directory, '*WORLD.avi'))[0]
    world_time_file = glob(os.path.join(whitenoise_directory, '*WORLD_BonsaiTS.csv'))[0]
    ephys_file = glob(os.path.join(whitenoise_directory, '*Ephys.bin'))[0]
    ephys_time_file = glob(os.path.join(whitenoise_directory, '*Ephys_BonsaiBoardTS.csv'))[0]
    print('loading and filtering ephys binary')
    pdf = PdfPages(os.path.join(whitenoise_directory, 'prelim_raw_whitenoise.pdf'))
    lfp_ephys = read_ephys_bin(ephys_file, probe, do_remap=False)
    ephys_center_sub = lfp_ephys - np.mean(lfp_ephys,0)
    filt_ephys = butter_bandpass(ephys_center_sub, lowcut=800, highcut=8000, fs=30000, order=6)
    t0 = open_time(ephys_time_file)[0]
    worldT = open_time(world_time_file)
    worldT = worldT - t0
    print('loading worldcam video')
    vidread = cv2.VideoCapture(world_file)

    f_count = int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))
    if f_count > np.size(worldT):
        f_count = int(np.size(worldT))

    world_vid = np.empty([f_count,
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.25),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*0.25)], dtype=np.uint8)
    # iterate through each frame
    for frame_num in range(0,f_count):
        # read the frame in and make sure it is read in correctly
        ret, frame = vidread.read()
        if not ret:
            break
        # convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # downsample the frame by an amount specified in the config file
        sframe = cv2.resize(frame, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        # add the downsampled frame to all_frames as int8
        world_vid[frame_num,:,:] = sframe.astype(np.int8)
    print('setting up worldcam and ephys')
    offset0 = 0.1
    drift_rate = -0.1/1000
    num_samp = np.size(filt_ephys,0)
    samp_freq = 30000
    ephys_time = np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq) - t0
    cam_gamma = 2
    world_norm = (world_vid/255)**cam_gamma
    std_im = np.std(world_norm,axis=0)
    std_im[std_im<10/255] = 10/255
    img_norm = (world_norm-np.mean(world_norm,axis=0))/std_im
    img_norm = img_norm * (std_im>20/255)
    img_norm[img_norm<-2] = -2
    movInterp = scipy.interpolate.interp1d(worldT, img_norm, axis=0, bounds_error=False)
    plt.subplots(np.size(filt_ephys,1),1,figsize=(5,int(np.ceil(np.size(filt_ephys,1)/2))))
    print('getting receptive fields and plotting')
    all_spikeT = []
    for ch in tqdm(range(np.size(filt_ephys,1))):
        spike_thresh = -350
        spike_inds = list(np.where(filt_ephys[:,ch] < spike_thresh)[0])
        spikeT = ephys_time[spike_inds]
        all_spikeT.append(spikeT - (offset0 + spikeT * drift_rate))
    all_STA, fig = plot_prelim_STA(all_spikeT, img_norm, worldT, movInterp, np.size(filt_ephys,1))
    pdf.savefig(); plt.close()
    pdf.close()
    print('done')

if __name__ == '__main__':
    prelimRF_raw()
    