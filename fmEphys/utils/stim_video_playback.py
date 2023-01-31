
import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import xarray as xr

import scipy.interpolate

import fmEphys

class WorldcamPlayback(fmEphys.Ephys):
    def __init__(self, cfg, recording_name, recording_path):
        fmEphys.Ephys.__init__(self, cfg, recording_name, recording_path)
        self.fm = False
        self.stim = 'wc'
        self.playback_eventT_json = os.path.join(fmEphys.up_dir(__file__, 1), 'worldcam_playback_saccade_frames.json')

    def calc_pseudosaccade_times(self):

        ttl_path = fmEphys.find('{}_preprocessed_TTL_data.h5'.format(self.recording_name), self.recording_path, MR=True)
        ttl_data = fmEphys.read_h5(ttl_path)

        self.ttl_time = ttl_data['time'] - self.ephysT0

        self.ttl_time = self.ttl_time - (self.default_ephys_offset + self.ttl_time * self.default_ephys_drift_rate)

        stim_start_times = scipy.interpolate.interp1d(self.ttl_time, ttl_data['stim_TTL'], bounds_error=False)(self.worldT.values)

        self.stim_starts = np.argwhere(np.diff(stim_start_times)>1).flatten()
        stim_frame_counts = np.diff(self.stim_starts)


    def calc_pseudosaccade_PSTHs(self):

        with open(self.playback_eventT_json, 'r') as fp:
            worldcam_playback_saccade_frames = json.load(fp)
        
        ### PSTHs
        playback_PSTHs = np.empty([len(self.cells.index.values), 4, 2001])*np.nan

        for i, ind in tqdm(enumerate(self.cells.index.values)):
            _sps = self.cells.loc[ind,'spikeT'].copy()
            # print(len(_sps))
            for stype, sname in enumerate(['gazeL','gazeR','compL','compR']):
                # drop the final presentation; stimulus will be turned off before it completes
                _eventFrame = np.concatenate([worldcam_playback_saccade_frames[sname] + f for f in self.stim_starts[:-1]])
                _eventT = self.ttl_time[_eventFrame]
                # print(len(_eventT))
                playback_PSTHs[i, stype, :] = self.calc_kde_PSTH(_sps, _eventT)
        
    def diagnostic_figs(self):

        pdf_name = '{}_WorldcamPlayback_stimulus_analysis.pdf'.format(self.recording_name)
        pdf = PdfPages(os.path.join(self.recording_path, pdf_name))

        with open(self.playback_eventT_json, 'r') as fp:
            worldcam_playback_saccade_frames = json.load(fp)

        ### FIG 1
        fig, axs = plt.subplots(4,7, figsize=(15,9), dpi=350)

        axs = axs.flatten()

        template_ind = worldcam_playback_saccade_frames['gazeL'][5] + self.stim_starts[0]
        template_frame = self.img_norm[template_ind,:,:].copy()
        axs[0].imshow(template_frame, cmap='gray')

        for r, rFrame in enumerate(self.stim_starts[:-1]):
            
            centerFrame = worldcam_playback_saccade_frames['gazeL'][5] + rFrame
            
            fr_diff = self.img_norm[centerFrame, :, :] - template_frame
            _colorbar_info = axs[r+1].imshow(fr_diff, cmap='seismic', vmin=-5, vmax=5)
                
            axs[r+1].set_title('r{} ({:.2}/{:.2})'.format(r, np.nanmin(fr_diff), np.nanmax(fr_diff)))
                
        for r in range(7*4):
                axs[r].axis('off')

        fig.tight_layout()
        pdf.savefig()
        fig.close()


        ### FIG 2

        ### plot some example frames
        fig, axs = plt.subplots(21, 7, figsize=(6,16), dpi=300)

        for r, rFrame in enumerate(self.stim_starts[:-1]): # [:-1]
            
            centerFrame = worldcam_playback_saccade_frames['gazeL'][5] + rFrame
            
            for f, fFrame in enumerate(np.arange(centerFrame-3, centerFrame+4)):
                
                if f == 3:
                    axs[r,f].imshow(self.img_norm[centerFrame,:,:], cmap='gray')
                else:
                    axs[r,f].imshow(self.img_norm[fFrame, :, :] - self.img_norm[centerFrame,:,:], cmap='seismic', vmin=-10, vmax=10)
                
                if r==0:
                    axs[r,f].set_title((f-3))
                
                
        for r in range(21):
            for f in range(7):
                axs[r,f].axis('off')

        fig.tight_layout()
        pdf.savefig()
        fig.close()


        pdf.close()

    def analyze(self):

        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        # self.overview_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_overview_analysis_figures.pdf')))
        self.detail_pdf = PdfPages(os.path.join(self.recording_path, '{}_detailed_analysis_figures.pdf'.format(self.recording_name)))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, '_diagnostic_analysis_figures.pdf'.format(self.recording_name)))
        
        self.base_ephys_analysis()

        self.calc_pseudosaccade_times()

        self.detail_pdf.close()
        self.diagnostic_pdf.close()