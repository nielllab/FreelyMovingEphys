"""
prelim_raw_whitenoise.py
"""
from glob import glob
import os, cv2
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
from utils.ephys import read_ephys_bin, butter_bandpass
from utils.time import open_time

def plot_prelim_STA(spikeT, img_norm, worldT, movInterp, ch_count, lag=2):
    n_units = len(spikeT)
    # model setup
    model_dt = 0.025
    model_t = np.arange(0, np.max(worldT), model_dt)
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

def main(whitenoise_directory, probe):
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
    world_timestamps = open_time(world_time_file)
    world_timestamps = world_timestamps - t0
    print('loading worldcam video')
    vidread = cv2.VideoCapture(world_file)
    world_vid = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.25),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*0.25)], dtype=np.uint8)
    # iterate through each frame
    for frame_num in range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))):
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
    worldT = open_time(world_time_file) - t0
    movInterp = interp1d(worldT, img_norm, axis=0, bounds_error=False)
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