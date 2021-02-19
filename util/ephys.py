"""
ephys.py

organize and combine ephys data

Dec. 02, 2020
"""
# package imports
import pandas as pd
import numpy as np
import xarray as xr
import os
from scipy.io import loadmat
# module imports
from util.time import open_time
from util.paths import find

# format ephys data read in from Phy2 output files
# returns DataFrame of ephys data
# also returns first timestamp from Bonsai timestamps
def format_spikes(spike_times_path, spike_clusters_path, cluster_group_path, ephys_time_path, templates_path, cluster_info_path, config):
    allSpikeT = np.load(spike_times_path)
    allSpikeT = allSpikeT/config['ephys_sample_rate']  # should be a lookup table with timestamps
    duration = np.max(allSpikeT)
    ephys_data = pd.read_csv(cluster_info_path,sep = '\t',index_col=0)
    clust = np.load(spike_clusters_path)
    ephys_data['spikeT'] = np.nan
    ephys_data['spikeT'] = ephys_data['spikeT'].astype(object)
    # get spiketimes for each cluster
    for c in np.unique(clust):
        ephys_data.at[c,'spikeT'] =allSpikeT[clust==c].flatten()
    # get waveform templates
    templates = np.load(templates_path)
    ephys_data['waveform'] = np.nan
    ephys_data['waveform'] = ephys_data['spikeT'].astype(object)
    for i, ind in enumerate(ephys_data.index):
        ephys_data.at[ind,'waveform'] = templates[ind,21:,ephys_data.at[ind,'ch']]
    # first timepoint
    ephys_data['t0'] = open_time(ephys_time_path)[0]

    return ephys_data

# format spikes as seperate json files
# merge file should be .mat file, config should be preprocessing .json
def format_spikes_multi(merge_file, config):
    # open 
    merge_info = loadmat(merge_file)
    fileList = merge_info['fileList']
    pathList = merge_info['pathList']
    nSamps = merge_info['nSamps']

    #load phy output data
    phy_path = os.path.split(merge_file)
    allSpikeT = np.load(os.path.join(phy_path[0],'spike_times.npy'))
    clust = np.load(os.path.join(phy_path[0],'spike_clusters.npy'))
    templates = np.load(os.path.join(phy_path[0],'templates.npy'))

    # ephys_data_master holds information that is same for all recordings (i.e. cluster information + waveform)
    ephys_data_master = pd.read_csv(os.path.join(phy_path[0],'cluster_info.tsv'),sep = '\t',index_col=0)

    # insert waveforms
    ephys_data_master['waveform'] = np.nan
    ephys_data_master['waveform'] = ephys_data_master['waveform'].astype(object) # does this need to be an object? spikeT is an object because they are all diff length, but I don't think these need to be -cmn
    for i, ind in enumerate(ephys_data_master.index):
        ephys_data_master.at[ind,'waveform'] = templates[ind,21:,ephys_data_master.at[ind,'ch']]

    # create boundaries between recordings (in terms of timesamples)
    boundaries = np.concatenate((np.array([0]),np.cumsum(nSamps)))

    # loop over each recording and create/save ephys_data for each one
    for s in range(np.size(nSamps)):

        # select spikes in this timerange
        use = (allSpikeT >= boundaries[s]) & (allSpikeT<boundaries[s+1] )
        theseSpikes = allSpikeT[use]
        theseClust = clust[use[:,0]]

        # place spikes into ephys data structure
        ephys_data = ephys_data_master.copy()
        ephys_data['spikeT'] = np.NaN
        ephys_data['spikeT'] = ephys_data['spikeT'].astype(object)
        for c in np.unique(clust):
            ephys_data.at[c,'spikeT'] =(theseSpikes[theseClust==c].flatten() - boundaries[s])/config['ephys_sample_rate'] 
        
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

        if any(i in key.split('_')[0] for i in conditions.get('dates_predoi')):
            data['pre/post'] = 'pre'
        elif any(i in key.split('_')[0] for i in conditions.get('dates_postdoi')):
            data['pre/post'] = 'post'

    all_data = pd.concat([data for key,data in spike_data.items()], keys=[key for key,data in spike_data.items()])

    return all_data