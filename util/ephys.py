"""
ephys.py
"""
import pandas as pd
import numpy as np
import xarray as xr
import os
from scipy.io import loadmat

from util.time import open_time
from util.paths import find

def format_spikes(spike_times_path, spike_clusters_path, cluster_group_path, ephys_time_path, templates_path, cluster_info_path, config):
    """
    format ephys data read in from phy2 output files
    only runs on one recording at a time, assuming individual analysis -- this is no longer used
    inputs are a series of paths to files output by phy2, that names of which match the names of inputs
    ephys_time_path should be the timestamp csv
    returns dataframe of ephys data
    """
    # open spike time data
    allSpikeT = np.load(spike_times_path)
    allSpikeT = allSpikeT/config['parameters']['ephys']['ephys_sample_rate']  # should be a lookup table with timestamps
    # length of the recording
    duration = np.max(allSpikeT)
    # read in cluster info
    ephys_data = pd.read_csv(cluster_info_path,sep = '\t',index_col=0)
    clust = np.load(spike_clusters_path)
    # enter the ephys data into the dataframe as an object (i.e. one list of many values as a single cell in the dataframe)
    ephys_data['spikeT'] = np.nan
    ephys_data['spikeT'] = ephys_data['spikeT'].astype(object)
    # get spiketimes for each cluster
    for c in np.unique(clust):
        ephys_data.at[c,'spikeT'] = allSpikeT[clust==c].flatten()
    # get waveform templates
    templates = np.load(templates_path)
    # load the waveform as a list in a cell also
    ephys_data['waveform'] = np.nan
    ephys_data['waveform'] = ephys_data['spikeT'].astype(object)
    for i, ind in enumerate(ephys_data.index):
        ephys_data.at[ind,'waveform'] = templates[ind,21:,ephys_data.at[ind,'ch']]
    # first timepoint
    ephys_data['t0'] = open_time(ephys_time_path)[0]

    return ephys_data

def format_spikes_multi(merge_file, config):
    """
    format spikes as seperate json files from one shared .mat file
    INPUTS
        merge_file: .mat file, config should be preprocessing .json
        config: options dict
    OUTPUTS
        None
    """
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
        use = (allSpikeT >= boundaries[s]) & (allSpikeT<boundaries[s+1])
        theseSpikes = allSpikeT[use]
        theseClust = clust[use[:,0]]

        # place spikes into ephys data structure
        ephys_data = ephys_data_master.copy()
        ephys_data['spikeT'] = np.NaN
        ephys_data['spikeT'] = ephys_data['spikeT'].astype(object)
        for c in np.unique(clust):
            ephys_data.at[c,'spikeT'] =(theseSpikes[theseClust==c].flatten() - boundaries[s])/config['parameters']['ephys']['ephys_sample_rate']
        
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
