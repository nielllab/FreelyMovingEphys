"""
ephys.py

organize and combine ephys data

Sept. 24, 2020
"""

# package imports
import pandas as pd
import numpy as np
import xarray as xr
import os

# module imports
from util.read_data import open_time

# format ephys data read in from Phy2 output files and return them as an xarray
def format_spikes(spike_times_path, spike_clusters_path, cluster_group_path, ephys_time_path, config):
    # load in the spike data
    spike_times = np.load(spike_times_path)
    spike_clusters = np.load(spike_clusters_path)
    # open the cluster key
    with open(cluster_group_path) as cg:
        cluster_group = pd.read_csv(cg, delimiter="\t", quotechar='"')
    # open the timestamp file
    time = open_time(time_path)
    # combine spike times and clusters into a dataframe, then merge the key so that there will be a column identifying
    # each spike as eithter 'good' 'mua' or 'noise'
    spikes = pd.DataFrame(np.vstack([np.squeeze(spike_times), spike_clusters]).T, columns=['spike_time','cluster_id'])
    all_ephys = pd.DataFrame.merge(spikes_pd, cluster_group, on='cluster_id')
    # only keep the spikes that were labeled in Phy2 as being 'good'
    all_ephys_good = all_ephys[all_ephys['group'].str.contains('good')]
    # put it all into an xarray and throw time in as a dimension
    ephys_data = xr.DataArray(all_ephys_good, dims=['spike_num', 'ephys_params'])
    ephys_data.expand_dims({'timestamps':time})

    return ephys_data