"""
ephys.py

organize and combine ephys data

Oct. 06, 2020
"""

# package imports
import pandas as pd
import numpy as np
import xarray as xr
import os

# module imports
from util.read_data import open_time

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
    t0 = open_time(ephys_time_path)[0]

    return ephys_data, t0

# combine all ephys data of a trial into a nested dictionary where each key contains the data and metadata of a single unit
def ephys_to_dict(ephys_data, t0, trial_name):
    cluster_dict_list = []
    cluster_name_list = []
    for cluster_id in ephys_data.index:
        cluster = ephys_data.loc[cluster_id]
        cluster_dict = dict(cluster)
        cluster_dict['t0'] = t0
        cluster_name = 'unit_' + trial_name + '_cluster' + str(cluster_id)
        cluster_dict_list.append(cluster_dict)
        cluster_name_list.append(cluster_name)
    all_clusters = dict(zip(cluster_name_list, cluster_dict_list))
    
    return all_clusters