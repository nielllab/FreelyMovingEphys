"""
ephys_utils.py

utilities for using ephys analysis outputs
"""
import pandas as pd
import numpy as np
import json
import os
from scipy.signal import sosfiltfilt
import cv2
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy.interpolate import interp1d
from tqdm import tqdm

from util.paths import find

def load_ephys(csv_filepath):
    # open the csv file of metadata and pull out all of the desired data paths
    csv = pd.read_csv(csv_filepath)
    for_data_pool = csv.loc[csv['load_for_data_pool'] == True]
    goodsessions = []
    # get all of the best freely moving recordings of a session into a dictionary
    goodfmrecs = dict(zip(list(for_data_pool['Experiment date']+'_'+for_data_pool['Animal name']),['fm1' if np.isnan(i) else i for i in for_data_pool['best_fm_rec']]))
    # get all of the session data locations into a list
    for ind, row in for_data_pool.iterrows():
        goodsessions.append(row['data_location'])
    # get the .h5 files from each day
    # this will be a list of lists, where each list inside of the main list has all the data of a single session
    sessions = [find('*_ephys_props.h5',session) for session in goodsessions]
    # read the data in and append them into one shared df
    all_data = pd.DataFrame([])
    for session in sessions:
        session_data = pd.DataFrame([])
        for recording in session:
            rec_data = pd.read_hdf(recording)
            # get name of the current recording (i.e. 'fm' or 'hf1_wn')
            rec_type = '_'.join(([col for col in rec_data.columns.values if 'trange' in col][0]).split('_')[:-1])
            # rename spike time columns so that data is retained for each of the seperate trials
            rec_data = rec_data.rename(columns={'spikeT':rec_type+'_spikeT', 'spikeTraw':rec_type+'_spikeTraw','rate':rec_type+'_rate','n_spikes':rec_type+'_n_spikes'})
            # add a column for which fm recording should be prefered
            for key,val in goodfmrecs.items():
                if key in rec_data['session']:
                    rec_data['best_fm_rec'] = val
            # add a column for the 'r' and 'm' of ellipse fit
            try:
                ellipse_json_path = find('*'+rec_data['best_fm_rec']+'*fm_eyecameracalc_props.json',session)
                with open(ellipse_json_path, 'r') as fp:
                    ellipse_fit_params = json.load(fp)
                rec_data['best_ellipse_fit_m'] = ellipse_fit_params['regression_m']
                rec_data['best_ellipse_fit_r'] = ellipse_fit_params['regression_r']
            except:
                rec_data['best_ellipse_fit_m'] = np.nan
                rec_data['best_ellipse_fit_r'] = np.nan
            # get column names
            column_names = list(session_data.columns.values) + list(rec_data.columns.values)
            # new columns for same unit within a session
            session_data = pd.concat([session_data, rec_data],axis=1,ignore_index=True)
            # add the list of column names from all sessions plus the current recording
            session_data.columns = column_names
            # remove duplicate columns (i.e. shared metadata)
            session_data = session_data.loc[:,~session_data.columns.duplicated()]
        # new rows for units from different mice or sessions
        all_data = pd.concat([all_data,session_data],axis=0)
    return all_data

def read_ephys_bin(binpath, ch_num, do_remap=True):
    if ch_num!=16 and ch_num!=64:
        print('not 16 or 64 ch probe -- check arg ch_num')
        return None
    if ch_num == 16:
        ch_remap = [15,18,10,23,11,22,12,21,9,24,13,20,14,19,16,17] - 8
        dtypes = np.dtype([("ch"+str(i),np.uint16) for i in range(0,16)])
    elif ch_num == 64:
        ch_remap = [32,62,33,63,34,60,36,61,37,58,38,59,40,56,41,57,42,54,44,55,
                    45,52,46,53,47,50,43,51,39,48,35,49,0,30,1,31,2,28,3,26,4,27,
                    5,24,6,22,7,23,8,20,9,18,10,19,11,16,12,17,13,21,14,25,15,29]
        dtypes = np.dtype([("ch"+str(i),np.uint16) for i in range(0,64)])
    # read in binary file
    ephys = pd.DataFrame(np.fromfile(binpath, dtypes, -1, ''))
    # remap with known order of channels
    if do_remap is True:
        ephys = ephys.iloc[:,list(ch_remap)]
    return ephys

def butter_bandpass(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, data, axis=0)