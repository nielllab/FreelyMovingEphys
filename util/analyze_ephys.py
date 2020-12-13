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

def find_files(rec_path, rec_name, free_move):
    # get the files names in the provided path
    eye_file = os.path.join(rec_path, rec_name + '_Reye.nc')
    world_file = os.path.join(rec_path, rec_name + '_world.nc')
    ephys_file = os.path.join(rec_path, rec_name + '_ephys_merge.json')
    imu_file = os.path.join(rec_path, rec_name + '_imu.nc')
    speed_file = os.path.join(rec_path, rec_name + '_speed.nc')

    if free_move is True:
        dict_out = {'eye':eye_file,'world':world_file,'ephys':ephys_file,'imu':imu_file,'save':rec_path,'name':rec_name}
    elif free_move is False:
        dict_out = {'eye':eye_file,'world':world_file,'ephys':ephys_file,'speed':speed_file,'save':rec_path,'name':rec_name}

    return dict_out

def figures(file_dict):

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

    # read ephys data
    ephys_data = pd.read_json(ephys_file)

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

    overview_pdf.close(); detail_pdf.close(); diagnostic_pdf.close()

