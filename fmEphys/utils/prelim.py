"""
fmEphys/utils/prelim.py

Written by DMM, 2021
"""


import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io

import fmEphys as fme


class PrelimRF(fme.Ephys):
    """ Preliminary receptive field mapping.
    """

    def __init__(self, binary_path, probe):

        head, _ = os.path.split(binary_path)
        self.recording_path = head
        self.recording_name = fme.auto_recording_name(head)

        self.probe = probe
        self.num_channels = next(int(num) for num in ['128','64','16'] if num in self.probe)
        self.n_cells = self.num_channels

        self.generic_config = {
            'animal_directory': self.recording_path
        }

        self.ephys_samprate = 30000
        self.ephys_offset = 0.1
        self.ephys_drift_rate = -0.1/1000
        self.model_dt = 0.025
        self.spike_thresh = -350

        # this will be applied to the worldcam twice
        # once when avi is packed into a np array, and again when it is put into new bins for spike times
        self.vid_dwnsmpl = 0.25


    def prelim_video_setup(self):

        cam_gamma = 2
        world_norm = (self.world_vid / 255) ** cam_gamma
        std_im = np.std(world_norm, 0)
        std_im[std_im<10/255] = 10 / 255
        self.img_norm = (world_norm - np.mean(world_norm, axis=0)) / std_im
        self.img_norm = self.img_norm * (std_im > 20 / 255)
        self.small_world_vid[self.img_norm < -2] = -2


    def minimal_process(self):

        self.detail_pdf = PdfPages(os.path.join(self.recording_name, 'prelim_raw_whitenoise.pdf'))

        wc = fme.Worldcam(self.generic_config, self.recording_name, self.recording_path, 'WORLD')
        self.worldT = wc.read_timestamp_file()
        self.world_vid = wc.pack_video_frames(usexr=False, dwsmpl=0.25)
        
        self.ephys_bin_path = glob(os.path.join(self.recording_path, '*Ephys.bin'))[0]
        ephys_time_file = glob(os.path.join(self.recording_path, '*Ephys_BonsaiBoardTS.csv'))[0]

        lfp_ephys = self.read_binary_file(do_remap=True)
        ephys_center_sub = lfp_ephys - np.mean(lfp_ephys, 0)
        filt_ephys = self.butter_bandpass(ephys_center_sub, lowcut=800, highcut=8000, fs=30000, order=6)

        ephysT = self.read_timestamp_file()
        t0 = ephysT[0]

        self.worldT = self.worldT - t0

        num_samp = np.size(filt_ephys, 0)
        new_ephysT = np.array(t0 + np.linspace(0, num_samp-1, num_samp) / self.ephys_samprate) - t0
        
        self.model_t = np.arange(0, np.nanmax(self.worldT), self.model_dt)

        self.prelim_video_setup()
        self.worldcam_at_new_timebase(dwnsmpl=0.25)

        all_spikeT = []
        for ch in tqdm(range(np.size(filt_ephys,1))):
            spike_inds = list(np.where(filt_ephys[:,ch] < self.spike_thresh)[0])
            spikeT = new_ephysT[spike_inds]
            all_spikeT.append(spikeT - (self.ephys_offset + spikeT * self.ephys_drift_rate))

        self.model_nsp = np.zeros((self.n_cells, len(self.model_t)))
        bins = np.append(self.model_t, self.model_t[-1]+self.model_dt)
        for i in range(self.n_cells):
            self.model_nsp[i,:], _ = np.histogram(all_spikeT[i], bins)

        self.calc_sta(do_rotation=True, using_spike_sorted=False)

        self.detail_pdf.close()



class RawEphys(fme.BaseInput):
    """ Read raw ephys data from Phy2 and
    format for use in fmEphys.
    """


    def __init__(self, merge_file):
        self.merge_file = merge_file
        self.ephys_samprate = 30000


    def format_spikes(self):
        # open 
        merge_info = scipy.io.loadmat(self.merge_file)
        fileList = merge_info['fileList']
        pathList = merge_info['pathList']
        nSamps = merge_info['nSamps']

        # load phy2 output data
        phy_path = os.path.split(self.merge_file)
        allSpikeT = np.load(os.path.join(phy_path[0],'spike_times.npy'))
        clust = np.load(os.path.join(phy_path[0],'spike_clusters.npy'))
        templates = np.load(os.path.join(phy_path[0],'templates.npy'))

        # ephys_data_master holds information that is same for all recordings (i.e. cluster information + waveform)
        ephys_data_master = pd.read_csv(os.path.join(phy_path[0],'cluster_info.tsv'),sep = '\t',index_col=0)

        # insert waveforms
        ephys_data_master['waveform'] = np.nan
        ephys_data_master['waveform'] = ephys_data_master['waveform'].astype(object)
        for _, ind in enumerate(ephys_data_master.index):
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
                ephys_data.at[c,'spikeT'] =(theseSpikes[theseClust==c].flatten() - boundaries[s])/self.ephys_samprate
            
            # get timestamp from csv for this recording
            fname = fileList[0,s][0].copy()
            fname = fname[0:-4] + '_BonsaiBoardTS.csv'
            self.timestamp_path = os.path.join(pathList[0,s][0],fname)
            ephysT = self.read_timestamp_file()
            ephys_data['t0'] = ephysT[0]
            
            # write ephys data into json file
            fname = fileList[0,s][0].copy()
            fname = fname[0:-10] + '_ephys_merge.json'
            ephys_json_path = os.path.join(pathList[0,s][0],fname)
            ephys_data.to_json(ephys_json_path)

            print('Saved {}'.format(ephys_json_path))

