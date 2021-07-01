"""
ephys_utils.py
"""
import pandas as pd
import numpy as np
import json, platform
import os
from scipy.signal import sosfiltfilt
import cv2
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy.interpolate import interp1d
import platform
from tqdm import tqdm
from datetime import datetime

from util.paths import find
from project_analysis.ephys.population_utils import make_population_summary, make_session_summary, make_unit_summary

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
    # get all of the best freely moving recordings of a session into a dictionary
    goodfmrecs = dict(zip(list(for_data_pool['experiment_date']+'_'+for_data_pool['animal_name']),['fm1' if np.isnan(i) else i for i in for_data_pool['best_fm_rec']]))
    # change paths to work with linux
    if platform.system() == 'Linux':
        for ind, row in for_data_pool.iterrows():
            drive = [row['drive'] if row['drive'] == 'nlab-nas' else row['drive'].capitalize()][0]
            for_data_pool.loc[ind,'animal_dirpath'] = os.path.expanduser('~/'+('/'.join([row['computer'].title(), drive] + list(filter(None, row['animal_dirpath'].replace('\\','/').split('/')))[2:])))
    for ind, row in for_data_pool.iterrows():
        goodsessions.append(row['animal_dirpath'])
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

def read_ephys_bin(binpath, probe_name, do_remap=True):
    """
    read in ephys binary file and apply remapping using the name of the probe
    where the binary dimensions and remaping vector are read in from relative
    path within the FreelyMovingEphys directory (FreelyMovingEphys/matlab/channel_maps.json)
    INPUTS
        binpath: path to binary file
        probe_name: name of probe, which should be a key in the dict stored in the .json of probe remapping vectors
        do_remap: bool, whether or not to remap the drive
    OUTPUTS
        ephys: ephys data as a DataFrame
    """
    # get channel number
    ch_num = int([16 if '16' in probe_name else 64][0])
    # find the file of default mappings
    if platform.system() == 'Linux':
        mapping_json = '/'.join(os.path.abspath(__file__).split('/')[:-3]) + '/matlab/channel_maps.json'
    else:
        mapping_json = '/'.join(os.path.abspath(__file__).split('\\')[:-3]) + '/matlab/channel_maps.json'
    # open file of default mappings
    with open(mapping_json, 'r') as fp:
        mappings = json.load(fp)
    # get the mapping for the probe name used in the current recording
    ch_remap = mappings[probe_name]
    # set up data types to read binary file into
    dtypes = np.dtype([("ch"+str(i),np.uint16) for i in range(0,ch_num)])
    # read in binary file
    ephys = pd.DataFrame(np.fromfile(binpath, dtypes, -1, ''))
    # remap with known order of channels
    if do_remap is True:
        ephys = ephys.iloc[:,[i-1 for i in list(ch_remap)]]
    return ephys

def butter_bandpass(data, lowcut=1, highcut=300, fs=30000, order=5):
    """
    apply bandpass filter to ephys lfp applied along axis=0
    INPUTS
        data: 2d array of multiple channels of ephys data as a numpy arrya or pandas dataframe
        lowcut: low end of cut off for frequency
        highcut: high end of cut off for frequency
        fs: sample rate
        order: order of filter
    OUTPUTS
        filtered data in the same type as input data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, data, axis=0)

def population_analysis(config):
    print('pooling ephys data')
    df = load_ephys(config['population']['metadata_csv_path'])
    # clean up h5 file
    cols = df.columns.values
    shcols = [c for c in cols if 'gratingssh' in c]
    for c in shcols:
        new_col = str(c.replace('gratingssh', 'gratings'))
        df = df.rename(columns={str(c): new_col})
    badcols = []
    for c in cols:
        if any(s in c for s in ['fm2','hf5','hf6','hf7','hf8']):
            badcols.append(c)
    df = df.drop(labels=badcols, axis=1)
    df = df.groupby(lambda x:x, axis=1); df = df.agg(np.nansum) # combine identical column names
    print('saving pooled ephys data to '+config['population']['save_path'])
    h5path = os.path.join(config['population']['save_path'],'pooled_ephys_'+datetime.today().strftime('%m%d%y')+'.h5')
    if os.path.isfile(h5path):
        os.remove(h5path)
    df.to_hdf(h5path, 'w')
    print('writing session summary')
    make_session_summary(df, config['population']['save_path'])
    print('writing unit summary')
    unit_df = make_unit_summary(df, config['population']['save_path'])
    # print('starting unit population analysis')
    # make_population_summary(unit_df, config['population']['save_path'])