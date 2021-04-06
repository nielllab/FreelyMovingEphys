"""
ephys_utils.py

utilities for using ephys analysis outputs
"""
import pandas as pd
import numpy as np
import json
import os
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

# read in many .npy ephys files ending in '*ephys_props.npy' by searching one or more date subdirectories
# format the data into an xarray dataset of all units, with metadata stored as attributes
# then, user can search entire dataset including any number of mice, dates, and units, and make comparisons
# using xr.filter_by_attrs function
def ephys_to_dataset(path, dates):
    # path should be something like T:/freely_moving_ephys/ephys_recordings/
    # dates should be a list of dates to include, as str, in the format 010121 for Jan 1st 2021

    # search for .npy ephys files in subdirectories of date directory
    # then, organize the returned filepaths into a list
    ephys_filepaths = []
    for day in dates:
        day_npys = find('*ephys_props.npy',os.path.join(path, day))
        for i in day_npys:
            ephys_filepaths.append(i)

    # read in the npys, get metadata, and append into dataset
    processed_unit_count = 0
    for filepath in ephys_filepaths:
        ephys = np.load(filepath, allow_pickle=True) # open npy files
        keys = ephys.item().keys() # get all the names of unit/cell entries
        for key in keys:
            # prepare some metadata from the unit key
            split_key = key.split('_')
            date = split_key[1]; mouse = split_key[0]; exp = split_key[2]; rig = split_key[3]; unit = split_key[-1]
            for x in [date, mouse, exp, rig, unit]:
                split_key.remove(x) # stim can have variable num of '_', so it's best to remove everything else and add
                                    # the remaining items in the list to the 'stim' attribute
            stim = '_'.join(split_key)
            unit_data = ephys.item().get(key) # get the data for the current key
            unit_keys = list(unit_data.keys()) # we'll need the keys for each property (e.g. downsacc_avg, etc.)
            # put into an xarray with labeled coordinates and dims
            unit_xr = xr.DataArray([unit_data.get(i) for i in unit_keys], dims=['ephys_params'], coords=[('ephys_params', unit_keys)])
            # add metadata
            unit_xr.attrs['date'] = date; unit_xr.attrs['mouse'] = mouse; unit_xr.attrs['exp'] = exp; unit_xr.attrs['rig'] = rig; unit_xr.attrs['unit'] = unit
            unit_xr.attrs['stim'] = stim; unit_xr.name = key # also important to name so that each datavariable can be indexed once merged into dataset
            # and append each unit into one big dataset
            if processed_unit_count == 0:
                all_units_xr = unit_xr.copy()
            else:
                all_units_xr = xr.merge([all_units_xr, unit_xr])
            processed_unit_count = processed_unit_count + 1

    return all_units_xr

# read in many .json ephys files of spike data, etc. and save them into a dictionary
# each entry in dictionary will have a key for the name of the recording
def ephys_to_dataframe(path,dates,conditions):
    # path and dates should be in the same format as func ephys_to_dataset
    ephys_filepaths = []
    for day in dates:
        day_jsons = find('*ephys_merge.json',os.path.join(path, day)) # get the .json for all recordings
        for rec in day_jsons:
            ephys_filepaths.append(rec) # append into list
    # build a dictionary with the filename as the key and the pandas df as a value
    spike_data = {os.path.split(filepath)[1]: pd.read_json(filepath) for filepath in ephys_filepaths}
    
    # iterate through dictionary to add a column, 'doi', of whether or not it was a doi recording
   
    for key,data in spike_data.items():
        data['date'] = key.split('_')[0]
        data['mouse'] = key.split('_')[1]
        data['rec'] = key.split('_')[4]
        if any(i in key.split('_')[4] for i in ['fm1','hf1','hf2','hf3','hf4']):
            data['doi'] = 'none'
        elif any(i in key.split('_')[4] for i in ['fm2','hf5','hf6','hf7','hf8']) and key.split('_')[0] in conditions.get('dates_doi'):
            data['doi'] = 'doi'
        elif any(i in key.split('_')[4] for i in ['fm2','hf5','hf6','hf7','hf8']) and key.split('_')[0] in conditions.get('dates_saline'):
            data['doi'] = 'saline'

        if any(i in key.split('_')[4] for i in ['fm1','hf1','hf2','hf3','hf4']) and key.split('_')[0] in conditions.get('dates_predoi'):
            data['pre/post'] = 'pre'
        elif any(i in key.split('_')[4] for i in ['fm1','hf1','hf2','hf3','hf4']) and key.split('_')[0] in conditions.get('dates_postdoi'):
            data['pre/post'] = 'post'
        elif any(i in key.split('_')[4] for i in ['fm2','hf5','hf6','hf7','hf8']):
            data['pre/post'] = 'none'

    all_data = pd.concat([data for key,data in spike_data.items()], keys=[key for key,data in spike_data.items()])

    return all_data