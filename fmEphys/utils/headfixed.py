"""
FreelyMovingEphys/src/headfixed.py
"""
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import scipy.interpolate
import scipy.signal

import cv2

import sklearn.cluster

import fmEphys

class HeadFixedWhiteNoise(fmEphys.Ephys):
    def __init__(self, cfg, recording_name, recording_path):
        fmEphys.Ephys.__init__(self, cfg, recording_name, recording_path)
        self.fm = False
        self.stim = 'wn'

    def save_as_df(self):
        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        unit_data = pd.DataFrame([])
        for unit_num, ind in enumerate(self.cells.index):
            cols = [
                'contrast',
                'contrast_tuning_bins',
                'contrast_tuning',
                'contrast_tuning_err',
                'spike_triggered_average',
                'spike_triggered_variance',
                'saccade_rightT',
                'saccade_leftT',
                'saccade_rightPSTH',
                'saccade_leftPSTH',
                'pupilradius_tuning_bins',
                'pupilradius_tuning',
                'pupilradius_tuning_err',
                'theta_tuning_bins',
                'theta_tuning',
                'theta_tuning_err',
                'phi_tuning_bins',
                'phi_tuning',
                'phi_tuning_err',
                'eyeT',
                'dEye_dpf',
                'dEye',
                'theta',
                'phi',
                'ballspeed',
                'ballspeed_tuning_bins',
                'ballspeed_tuning',
                'ballspeed_tuning_err',
                'lfp_power',
                'layer5cent_from_lfp',
                'worldT'
            ]
            unit_df = pd.DataFrame(pd.Series([
                self.contrast,
                self.contrast_tuning_bins,
                self.contrast_tuning[unit_num],
                self.contrast_tuning_err[unit_num],
                self.sta[unit_num],
                self.stv[unit_num],
                self.all_eyeR,
                self.all_eyeL,
                self.rightsacc_avg[unit_num],
                self.leftsacc_avg[unit_num],
                self.pupilradius_tuning_bins,
                self.pupilradius_tuning[unit_num],
                self.pupilradius_tuning_err[unit_num],
                self.theta_tuning_bins,
                self.theta_tuning[unit_num],
                self.theta_tuning_err[unit_num],
                self.phi_tuning_bins,
                self.phi_tuning[unit_num],
                self.phi_tuning_err[unit_num],
                self.eyeT,
                self.dEye,
                self.dEye_dps,
                self.theta,
                self.phi,
                self.ball_speed,
                self.ballspeed_tuning_bins,
                self.ballspeed_tuning[unit_num],
                self.ballspeed_tuning_err[unit_num],
                self.lfp_power_profiles,
                self.lfp_layer5_centers,
                self.worldT.values
            ]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]

            unit_data = pd.concat([unit_data, unit_df], axis=0)

        data_out = pd.concat([self.cells, unit_data], axis=1)

        # Add a prefix to all column names
        data_out = data_out.add_prefix('Wn_')

        data_out['session'] = self.session_name

        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def save_as_dict(self):

        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        stim = 'Wn'
        save_dict = {
            'CRF': self.contrast_tuning_bins,
            'CRF_err': self.contrast_tuning,
            'CRF_bins':self.contrast_tuning_err,
            'STA': self.sta,
            'STV': self.stv,
            'saccade_rightT': self.all_eyeR,
            'saccade_leftT': self.all_eyeL,
            'saccade_rightPSTH': self.rightsacc_avg,
            'saccade_leftPSTH': self.leftsacc_avg,
            'PupilRadius_tuning': self.pupilradius_tuning,
            'PupilRadius_bins': self.pupilradius_tuning_bins,
            'PupilRadius_err': self.pupilradius_tuning_err,
            'Theta_tuning': self.theta_tuning,
            'Theta_bins': self.theta_tuning_bins,
            'Theta_err': self.theta_tuning_err,
            'Phi_tuning': self.phi_tuning,
            'Phi_bins': self.phi_tuning_bins,
            'Phi_err': self.phi_tuning_err,
            'eyeT': self.eyeT,
            'dEye': self.dEye_dps,
            'dEye_dpf': self.dEye,
            'theta': self.theta,
            'phi': self.phi,
            'ballspeed': self.ball_speed, 
            'ballspeed_tuning_bins': self.ballspeed_tuning_bins,
            'ballspeed_tuning': self.ballspeed_tuning,
            'ballspeed_tuning_err': self.ballspeed_tuning_err,
            'lfp_power_profiles': self.lfp_power_profiles,
            'laminar_position_layer5': self.lfp_layer5_centers,
            'session': self.session_name,
            'worldT': self.worldT.values,
            'stim': stim
        }

        for key, val in save_dict.items():
            if type(val)==xr.DataArray:
                save_dict[key] = val.values
        
        # Merge Phy2 cell data in
        save_cells = self.cells.to_dict(orient='list')
        save_dict = {**save_cells, **save_dict}

        # Add 'FmLt' to the start of each key
        for old_key in save_dict.keys():
            save_dict['{}_{}'.format(stim,old_key)] = save_dict.pop(old_key)

        savepath = os.path.join(self.recording_path,
                                '{}_ephys_props.h5'.format(self.recording_name))

        fmEphys.write_h5(savepath, save_dict)

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        # self.overview_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_overview_analysis_figures.pdf')))
        self.detail_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_diagnostic_analysis_figures.pdf')))

        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        print('closing pdfs')
        self.detail_pdf.close()
        self.diagnostic_pdf.close()

        print('saving ephys file')
        self.save_as_df()

class HeadFixedReversingCheckboard(fmEphys.Ephys):
    def __init__(self, cfg, recording_name, recording_path):
        fmEphys.Ephys.__init__(self, cfg, recording_name, recording_path)

        self.fm = False
        self.stim = 'rc'

        self.revchecker_window_start = 0.1 # sec
        self.revchecker_window_end = 0.5 # sec

    def revchecker_laminar_depth(self):
        # read in the binary file of ephys recording
        lfp_ephys = self.read_binary_file()
        # subtract off average for each channel, then apply bandpass filter
        ephys_center_sub = lfp_ephys - np.mean(lfp_ephys, 0)
        filt_ephys = self.butter_bandpass(ephys_center_sub, order=6)
        # k means clustering into two clusters
        # will seperate out the two checkerboard patterns
        # diff of labels will give each transition between checkerboard patterns (i.e. each reversal)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        num_frames = np.size(self.world_vid, 0); vid_width = np.size(self.world_vid, 1); vid_height = np.size(self.world_vid, 2)
        kmeans_input = self.world_vid.reshape(num_frames, vid_width*vid_height)
        _, labels, _ = cv2.kmeans(kmeans_input.astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        label_diff = np.diff(np.ndarray.flatten(labels))
        revind = list(abs(label_diff)) # need abs because reversing back will be -1
        # plot time between reversals, which should be centered around 1sec
        plt.figure()
        plt.title('time between checkerboard reversals')
        plt.hist(np.diff(self.worldT[np.where(revind)]), bins=100)
        plt.xlabel('sec'); plt.xlim([0.9,1.1])
        self.diagnostic_pdf.savefig(); plt.close()
        
        # get response of each channel centered around time of checkerboard reversal
        all_resp = np.zeros([np.size(filt_ephys, 1),
                             np.sum(revind),
                             len(list(set(np.arange(1-self.revchecker_window_start, 1+self.revchecker_window_end, 1/self.ephys_samprate))))])
        true_rev_index = 0
        for rev_index, rev_label in tqdm(enumerate(revind)):
            if rev_label == True and self.worldT[rev_index] > 1:
                for ch_num in range(np.size(filt_ephys, 1)):
                    # index of ephys data to start window with, aligned to set time before checkerboard will reverse
                    bin_start = int((self.worldT[rev_index]-self.revchecker_window_start)*self.ephys_samprate)
                    # index of ephys data to end window with, aligned to time after checkerboard reversed
                    bin_end = int((self.worldT[rev_index]+self.revchecker_window_end)*self.ephys_samprate)
                    # index into the filtered ephys data and store each trace for this channel of the probe
                    if bin_end < np.size(filt_ephys, 0): # make sure it's a possible range
                        all_resp[ch_num, true_rev_index] = filt_ephys[bin_start:bin_end, ch_num]
                true_rev_index = true_rev_index + 1
        # mean of responses within each channel
        rev_resp_mean = np.mean(all_resp, 1)

        # plot traces for shanks
        if self.num_channels==64:
            colors = plt.cm.jet(np.linspace(0,1,32))
            plt.subplots(1, 2, figsize=(8,6))
            for ch_num in np.arange(0,64):
                if ch_num<=31:
                    plt.subplot(1,2,1)
                    plt.plot(rev_resp_mean[ch_num], color=colors[ch_num], linewidth=1)
                    plt.title('shank0'); plt.axvline(x=(0.1*self.ephys_samprate))
                    plt.xticks(np.arange(0,18000,18000/5), np.arange(0,600,600/5))
                if ch_num>31:
                    plt.subplot(1,2,2)
                    plt.plot(rev_resp_mean[ch_num], color=colors[ch_num-32], linewidth=1)
                    plt.title('shank1'); plt.axvline(x=(0.1*self.ephys_samprate))
                    plt.xticks(np.arange(0,18000,18000/5), np.arange(0,600,600/5))
            plt.tight_layout()
            self.detail_pdf.savefig(); plt.close()
            num_cols_to_plot = 2

        elif self.num_channels==16:
            colors = plt.cm.jet(np.linspace(0,1,16))
            plt.figure()
            for ch_num in np.arange(0,16):
                plt.plot(rev_resp_mean[ch_num], color=colors[ch_num], linewidth=1)
                plt.axvline(x=(0.1*self.ephys_samprate))
                plt.xticks(np.arange(0,18000,18000/5), np.arange(0,600,600/5))
            self.detail_pdf.savefig(); plt.close()
            num_cols_to_plot = 1
        
        elif self.num_channels==128:
            colors = plt.cm.jet(np.linspace(0,1,32))
            plt.subplots(1,4 ,figsize=(40,6))
            for ch_num in np.arange(0,128):
                if ch_num < 32:
                    plt.subplot(1,4,1)
                    plt.plot(rev_resp_mean[ch_num], color=colors[ch_num], linewidth=1)
                    plt.title('shank0'); plt.axvline(x=(0.1*self.ephys_samprate))
                    plt.xticks(np.arange(0,18000,18000/5), np.arange(0,600,600/5))
                elif 32 <= ch_num < 64:
                    plt.subplot(1,4,2)
                    plt.plot(rev_resp_mean[ch_num], color=colors[ch_num-32], linewidth=1)
                    plt.title('shank1'); plt.axvline(x=(0.1*self.ephys_samprate))
                    plt.xticks(np.arange(0,18000,18000/5), np.arange(0,600,600/5))
                elif 64 <= ch_num < 96:
                    plt.subplot(1,4,3)
                    plt.plot(rev_resp_mean[ch_num], color=colors[ch_num-64], linewidth=1)
                    plt.title('shank2'); plt.axvline(x=(0.1*self.ephys_samprate))
                    plt.xticks(np.arange(0,18000,18000/5), np.arange(0,600,600/5))
                elif 96 <= ch_num < 128:
                    plt.subplot(1,4,4)
                    plt.plot(rev_resp_mean[ch_num], color=colors[ch_num-96], linewidth=1)
                    plt.title('shank3'); plt.axvline(x=(0.1*self.ephys_samprate))
                    plt.xticks(np.arange(0,18000,18000/5), np.arange(0,600,600/5))
            plt.tight_layout()
            self.detail_pdf.savefig(); plt.close()
            num_cols_to_plot = 4

        fig, axes = plt.subplots(int(np.size(rev_resp_mean,0)/num_cols_to_plot), num_cols_to_plot, figsize=(7,20), sharey=True)
        for ch_num, ax in enumerate(axes.T.flatten()):
            ax.plot(rev_resp_mean[ch_num], linewidth=1)
            ax.axvline(x=(0.1*self.ephys_samprate), linewidth=1)
            ax.axis('off')
            ax.set_title(ch_num)
        plt.tight_layout()
        self.detail_pdf.savefig(); plt.close()

        # Current source density
        csd = np.ones([np.size(rev_resp_mean,0), np.size(rev_resp_mean,1)])
        csd_interval = 2
        for ch in range(2, np.size(rev_resp_mean,0)-2):
            csd[ch] = rev_resp_mean[ch] - 0.5*(rev_resp_mean[ch-csd_interval] + rev_resp_mean[ch+csd_interval])
        # csd between -1 and 1
        self.csd_interp = np.interp(csd, (csd.min(), csd.max()), (-1, +1))
        # visualize csd
        fig, ax = plt.subplots(1,1)
        plt.subplot(1,1,1)
        plt.imshow(self.csd_interp, cmap='jet')
        plt.axes().set_aspect('auto'); plt.colorbar()
        plt.xticks(np.arange(0,18000,18000/5), np.arange(0,600,600/5))
        plt.xlabel('msec'); plt.ylabel('channel')
        plt.axvline(x=(0.1*self.ephys_samprate), color='k')
        plt.title('revchecker csd')
        self.detail_pdf.savefig(); plt.close()
        
        # assign the deepest deflection to lfp, the center of layer 4, to have depth 0
        # channels above will have negative depth, channels below will have positive depth
        # adding or subtracting "depth" with a step size of 1
        if self.num_channels==64:
            shank0_layer4cent = np.argmin(np.min(rev_resp_mean[0:32,int(self.ephys_samprate*0.1):int(self.ephys_samprate*0.3)], axis=1))
            shank1_layer4cent = np.argmin(np.min(rev_resp_mean[32:64,int(self.ephys_samprate*0.1):int(self.ephys_samprate*0.3)], axis=1))
            shank0_ch_positions = list(range(32)) - shank0_layer4cent; shank1_ch_positions = list(range(32)) - shank1_layer4cent
            self.lfp_depth = [shank0_ch_positions, shank1_ch_positions]
            self.layer4_centers = [shank0_layer4cent, shank1_layer4cent]
        elif self.num_channels==16:
            layer4cent = np.argmin(np.min(rev_resp_mean, axis=1))
            self.lfp_depth = [list(range(16)) - layer4cent]
            self.layer4_centers = [layer4cent]
        elif self.num_channels==128:
            shank0_layer4cent = np.argmin(np.min(rev_resp_mean[0:32, int(self.ephys_samprate*0.1):int(self.ephys_samprate*0.3)], axis=1))
            shank1_layer4cent = np.argmin(np.min(rev_resp_mean[32:64, int(self.ephys_samprate*0.1):int(self.ephys_samprate*0.3)], axis=1))
            shank2_layer4cent = np.argmin(np.min(rev_resp_mean[64:96, int(self.ephys_samprate*0.1):int(self.ephys_samprate*0.3)], axis=1))
            shank3_layer4cent = np.argmin(np.min(rev_resp_mean[96:128, int(self.ephys_samprate*0.1):int(self.ephys_samprate*0.3)], axis=1))
            shank0_ch_positions = list(range(32)) - shank0_layer4cent; shank1_ch_positions = list(range(32)) - shank1_layer4cent
            shank2_ch_positions = list(range(32)) - shank2_layer4cent; shank3_ch_positions = list(range(32)) - shank3_layer4cent
            self.lfp_depth = [shank0_ch_positions, shank1_ch_positions, shank2_ch_positions, shank3_ch_positions]
            self.layer4_centers = [shank0_layer4cent, shank1_layer4cent, shank2_layer4cent, shank3_layer4cent]
        self.rev_resp_mean = rev_resp_mean

    def calc_flashed_responses(self):

        self.Rc_world = xr.open_dataset(self.world_path)

        vid = self.Rc_world.WORLD_video.values.astype(np.uint8)
        worldT = self.Rc_world.timestamps.values
        eyeT = self.eyeT

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

        num_frames = np.size(vid, 0)
        vid_width = np.size(vid, 1)
        vid_height = np.size(vid, 2)

        kmeans_input = vid.reshape(num_frames, vid_width*vid_height)

        _, labels, _ = cv2.kmeans(kmeans_input.astype(np.float32),
                                    2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        label_diff = np.diff(np.ndarray.flatten(labels))

        stim_state = scipy.interpolate.interp1d(worldT[:-1]-self.ephysT0, label_diff, bounds_error=False)(eyeT)
        eventT = eyeT[np.where((stim_state<-0.1)+(stim_state>0.1))]

        Rc_psth = np.zeros([len(self.cells.index.values), 2001]) # shape = [unit#, time]
        for i, ind in tqdm(enumerate(self.cells.index.values)):
            unit_spikeT = self.cells.loc[ind, 'spikeT']
            if len(unit_spikeT) < 10 :
                continue # skip units that didn't fire enough
            Rc_psth[i,:] = self.calc_kde_PSTH(unit_spikeT, eventT)

        self.Rc_psth = Rc_psth
        self.Rc_eventT = eventT

    def save_as_df(self):

        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        unit_data = pd.DataFrame([])
        for unit_num, ind in enumerate(self.cells.index):
            cols = [
                'contrast',
                'contrast_tuning_bins',
                'contrast_tuning',
                'contrast_tuning_err',
                'spike_triggered_average',
                'spike_triggered_variance',
                'saccade_rightT',
                'saccade_leftT',
                'saccade_rightPSTH',
                'saccade_leftPSTH',
                'pupilradius_tuning_bins',
                'pupilradius_tuning',
                'pupilradius_tuning_err',
                'theta_tuning_bins',
                'theta_tuning',
                'theta_tuning_err',
                'phi_tuning_bins',
                'phi_tuning',
                'phi_tuning_err',
                'eyeT',
                'dEye_dpf',
                'dEye',
                'theta',
                'phi',
                'ballspeed',
                'ballspeed_tuning_bins',
                'ballspeed_tuning',
                'ballspeed_tuning_err',
                'response_by_channel',
                'current_source_density',
                'relative_depth',
                'layer4cent',
                'stim_PSTH',
                'stim_times',
                'worldT'
            ]
            unit_df = pd.DataFrame(pd.Series([
                self.contrast,
                self.contrast_tuning_bins,
                self.contrast_tuning[unit_num],
                self.contrast_tuning_err[unit_num],
                self.sta[unit_num],
                self.stv[unit_num],
                self.all_eyeR,
                self.all_eyeL,
                self.rightsacc_avg[unit_num],
                self.leftsacc_avg[unit_num],
                self.pupilradius_tuning_bins,
                self.pupilradius_tuning[unit_num],
                self.pupilradius_tuning_err[unit_num],
                self.theta_tuning_bins,
                self.theta_tuning[unit_num],
                self.theta_tuning_err[unit_num],
                self.phi_tuning_bins,
                self.phi_tuning[unit_num],
                self.phi_tuning_err[unit_num],
                self.eyeT,
                self.dEye,
                self.dEye_dps,
                self.theta,
                self.phi,
                self.ball_speed,
                self.ballspeed_tuning_bins,
                self.ballspeed_tuning[unit_num],
                self.ballspeed_tuning_err[unit_num],
                self.rev_resp_mean,
                self.csd_interp,
                self.lfp_depth,
                self.layer4_centers,
                self.Rc_psth[unit_num],
                self.Rc_eventT,
                self.worldT.values
            ]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]

            unit_data = pd.concat([unit_data, unit_df], axis=0)

        data_out = pd.concat([self.cells, unit_data], axis=1)

        # Add a prefix to all column names
        data_out = data_out.add_prefix('Rc_')

        data_out['session'] = self.session_name

        data_out.to_hdf(os.path.join(self.recording_path,
                       (self.recording_name+'_ephys_props.h5')), 'w')

    def save_as_dict(self):

        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        stim = 'Rc'
        save_dict = {
            'CRF': self.contrast_tuning_bins,
            'CRF_err': self.contrast_tuning,
            'CRF_bins':self.contrast_tuning_err,
            'STA': self.sta,
            'STV': self.stv,
            'saccade_rightT': self.all_eyeR,
            'saccade_leftT': self.all_eyeL,
            'saccade_rightPSTH': self.rightsacc_avg,
            'saccade_leftPSTH': self.leftsacc_avg,
            'PupilRadius_tuning': self.pupilradius_tuning,
            'PupilRadius_bins': self.pupilradius_tuning_bins,
            'PupilRadius_err': self.pupilradius_tuning_err,
            'Theta_tuning': self.theta_tuning,
            'Theta_bins': self.theta_tuning_bins,
            'Theta_err': self.theta_tuning_err,
            'Phi_tuning': self.phi_tuning,
            'Phi_bins': self.phi_tuning_bins,
            'Phi_err': self.phi_tuning_err,
            'eyeT': self.eyeT,
            'dEye': self.dEye_dps,
            'dEye_dpf': self.dEye,
            'theta': self.theta,
            'phi': self.phi,
            'ballspeed': self.ball_speed, 
            'ballspeed_tuning_bins': self.ballspeed_tuning_bins,
            'ballspeed_tuning': self.ballspeed_tuning,
            'ballspeed_tuning_err': self.ballspeed_tuning_err,
            'stim_PSTH': self.Rc_psth,
            'sim_times': self.Rc_eventT,
            'stim_CSD': self.csd_interp,
            'laminar_depth_LFP': self.lfp_depth,
            'center_layer4': self.layer4_centers,
            'rev_resp_mean': self.rev_resp_mean,
            'session': self.session_name,
            'stim': stim,
            'worldT': self.worldT.values
        }

        for key, val in save_dict.items():
            if type(val)==xr.DataArray:
                save_dict[key] = val.values
        
        # Merge Phy2 cell data in
        save_cells = self.cells.to_dict(orient='list')
        save_dict = {**save_cells, **save_dict}

        # Add 'FmLt' to the start of each key
        for old_key in list(save_dict.keys()):
            save_dict['{}_{}'.format(stim,old_key)] = save_dict.pop(old_key)

        savepath = os.path.join(self.recording_path,
                                '{}_ephys_props.h5'.format(self.recording_name))

        fmEphys.write_h5(savepath, save_dict)

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        self.detail_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_diagnostic_analysis_figures.pdf')))

        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()
        
        print('getting depth from reversing checkboard stimulus')
        self.revchecker_laminar_depth()

        print('Stimulus PSTHs')
        self.calc_flashed_responses()

        print('closing pdfs')
        self.detail_pdf.close()
        self.diagnostic_pdf.close()

        print('saving ephys file')
        self.save_as_df()

class HeadFixedSparseNoise(fmEphys.Ephys):
    def __init__(self, cfg, recording_name, recording_path):
        fmEphys.Ephys.__init__(self, cfg, recording_name, recording_path)

        self.fm = False
        self.stim = 'sn'

        self.Sn_dStim_thresh = 1e5
        self.Sn_rf_change_thresh = 30
        self.frameshift = 4

    def calc_RF_stim(self, unit_sta, vid):
        flat_unit_sta = unit_sta.copy().flatten()
        on_y, on_x = np.unravel_index(np.argmax(flat_unit_sta), unit_sta.shape)
        off_y, off_x = np.unravel_index(np.argmin(flat_unit_sta), unit_sta.shape)
        on_stim_history = vid[:,on_y*2,on_x*2]
        off_stim_history = vid[:,off_y*2,off_x*2]
        return on_stim_history, (on_x, on_y), off_stim_history, (off_x, off_y)

    def sort_lum(self, unit_stim, eventT, eyeT, flips):
        event_eyeT = np.zeros(len(eventT))
        for i, t in enumerate(eventT.values):
            event_eyeT[i] = eyeT[np.argmin(np.abs(t-eyeT))]
        gray = np.nanmedian(unit_stim)
        
        shifted_flips = flips+self.frameshift
        if np.max(shifted_flips) > (unit_stim.size-self.frameshift):
            shifted_flips = shifted_flips[:-1]
            event_eyeT = event_eyeT[:-1]
            
        rf_off = event_eyeT.copy(); rf_on = event_eyeT.copy(); only_global = event_eyeT.copy()

        off_bool = unit_stim[shifted_flips]<(gray-self.Sn_rf_change_thresh)
        offT = rf_off[off_bool] # light-to-dark transitions, as a timestamp in ephys eyeT timebase
        # offInds = flips[np.where(off_bool)[0]]
        
        on_bool = unit_stim[shifted_flips]>(gray+self.Sn_rf_change_thresh)
        onT = rf_on[on_bool] # same for dark-to-light transitions
        # onInds = flips[np.where(on_bool)[0]]
        
        background_bool = (unit_stim[shifted_flips]>(gray-self.Sn_rf_change_thresh)) & (unit_stim[shifted_flips]<(gray+self.Sn_rf_change_thresh))
        backgroundT = only_global[background_bool] # stim did not change from baseline enoguh
        # backgroundInds = flips[np.where(background_bool)[0]]
        
        return event_eyeT, offT, onT, backgroundT
    
    def calc_Sn_psth(self):

        _offset_time=(1/120)

        # Read it in again, since the origional will not still be held in memory
        self.Sn_world = xr.open_dataset(self.world_path)
        vid = self.Sn_world.WORLD_video.values.astype(np.uint8).astype(float)

        self.unit_stim_eventT = {}

        # when does the stimulus change?
        dStim = np.sum(np.abs(np.diff(vid, axis=0)), axis=(1,2))
        flips = np.argwhere((dStim[1:]>self.Sn_dStim_thresh) * (dStim[:-1]<self.Sn_dStim_thresh)).flatten()

        eventT = self.worldT[flips+1] - self.ephysT0

        rf_xy = np.zeros([len(self.cells.index.values),4]) # [unit#, on x, on y, off x, off y]
        on_Sn_psth = np.zeros([len(self.cells.index.values), 2001, 4]) # shape = [unit#, time, all/ltd/on/not_rf]
        off_Sn_psth = np.zeros([len(self.cells.index.values), 2001, 4])
        for cell_i, ind in tqdm(enumerate(self.cells.index.values)):
            unit_sta = self.sta[cell_i]
            on_stim_history, on_xy, off_stim_history, off_xy = self.calc_RF_stim(unit_sta, vid)
            rf_xy[cell_i,0] = on_xy[0]; rf_xy[cell_i,1] = on_xy[1]
            rf_xy[cell_i,2] = off_xy[0]; rf_xy[cell_i,3] = off_xy[1]
            # spikes
            unit_spikeT = self.cells.loc[ind, 'spikeT']
            if len(unit_spikeT)<10: # if a unit never fired during revchecker
                on_Sn_psth[cell_i,:,:] = np.empty(2001)*np.nan
                off_Sn_psth[cell_i,:,:] = np.empty(2001)*np.nan
                continue
            # on subunit
            all_eventT, offT, onT, backgroundT = self.sort_lum(on_stim_history, eventT, self.eyeT, flips)
            if len(offT)==0 or len(onT)==0:
                on_Sn_psth[cell_i,:,:] = np.empty(2001)*np.nan
                continue

            _event_names = ['allT', 'darkT', 'lightT', 'bckgndT']
            unit_stim_eventT = {}
            for i, n in enumerate(_event_names):
                unit_stim_eventT['onSubunit_eventT_{}'.format(n)] = [all_eventT, offT, onT, backgroundT][i] + _offset_time

            # print('all={} off={}, on={}, background={}'.format(len(all_eventT), len(offT), len(onT), len(backgroundT)))
            on_Sn_psth[cell_i,:,0] = self.calc_kde_PSTH(unit_spikeT, all_eventT+_offset_time)
            on_Sn_psth[cell_i,:,1] = self.calc_kde_PSTH(unit_spikeT, offT+_offset_time)
            on_Sn_psth[cell_i,:,2] = self.calc_kde_PSTH(unit_spikeT, onT+_offset_time)
            on_Sn_psth[cell_i,:,3] = self.calc_kde_PSTH(unit_spikeT, backgroundT+_offset_time)
            
            # off subunit
            all_eventT, offT, onT, backgroundT = self.sort_lum(off_stim_history, eventT, self.eyeT, flips)
            if len(offT)==0 or len(onT)==0:
                off_Sn_psth[cell_i,:,:] = np.empty(2001)*np.nan
                continue

            for i, n in enumerate(_event_names):
                unit_stim_eventT['offSubunit_eventT_{}'.format(n)] = [all_eventT, offT, onT, backgroundT][i] + _offset_time

            # print('all={} off={}, on={}, background={}'.format(len(all_eventT), len(offT), len(onT), len(backgroundT)))
            off_Sn_psth[i,:,0] = self.calc_kde_PSTH(unit_spikeT, all_eventT+_offset_time)
            off_Sn_psth[i,:,1] = self.calc_kde_PSTH(unit_spikeT, offT+_offset_time)
            off_Sn_psth[i,:,2] = self.calc_kde_PSTH(unit_spikeT, onT+_offset_time)
            off_Sn_psth[i,:,3] = self.calc_kde_PSTH(unit_spikeT, backgroundT+_offset_time)

            self.unit_stim_eventT[cell_i] = unit_stim_eventT

        self.on_Sn_psth = on_Sn_psth
        self.off_Sn_psth = off_Sn_psth
        self.rf_xy = rf_xy
        self.all_stimT = eventT+_offset_time

    def save_as_df(self):

        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        unit_data = pd.DataFrame([])
        for unit_num, ind in enumerate(self.cells.index):
            cols = [
                'contrast',
                'contrast_tuning_bins',
                'contrast_tuning',
                'contrast_tuning_err',
                'spike_triggered_average',
                'spike_triggered_variance',
                'saccade_rightT',
                'saccade_leftT',
                'saccade_rightPSTH',
                'saccade_leftPSTH',
                'pupilradius_tuning_bins',
                'pupilradius_tuning',
                'pupilradius_tuning_err',
                'theta_tuning_bins',
                'theta_tuning',
                'theta_tuning_err',
                'phi_tuning_bins',
                'phi_tuning',
                'phi_tuning_err',
                'eyeT',
                'dEye_dpf',
                'dEye_dps',
                'theta',
                'phi',
                'ballspeed',
                'ballspeed_tuning_bins',
                'ballspeed_tuning',
                'ballspeed_tuning_err',
                'stim_PSTH_onSub_all',
                'stim_PSTH_onSub_darkRF',
                'stim_PSTH_onSub_lightRF',
                'stim_PSTH_onSub_bckgndRF',
                'stim_PSTH_offSub_all',
                'stim_PSTH_offSub_darkRF',
                'stim_PSTH_offSub_lightRF',
                'stim_PSTH_offSub_bckgndRF',
                'approx_onSub_RF_coords',
                'approx_offSub_RF_coords',
                'stimT_all_shared',
                'stimT_unit_offSub_allT',
                'stimT_unit_offSub_darkT',
                'stimT_unit_offSub_lightT',
                'stimT_unit_offSub_bckgndT',
                'stimT_unit_onSub_allT',
                'stimT_unit_onSub_darkT',
                'stimT_unit_onSub_lightT',
                'stimT_unit_onSub_bckgndT',
                'worldT'
            ]
            unit_df = pd.DataFrame(pd.Series([
                self.contrast,
                self.contrast_tuning_bins,
                self.contrast_tuning[unit_num],
                self.contrast_tuning_err[unit_num],
                self.sta[unit_num],
                self.stv[unit_num],
                self.all_eyeR,
                self.all_eyeL,
                self.rightsacc_avg[unit_num],
                self.leftsacc_avg[unit_num],
                self.pupilradius_tuning_bins,
                self.pupilradius_tuning[unit_num],
                self.pupilradius_tuning_err[unit_num],
                self.theta_tuning_bins,
                self.theta_tuning[unit_num],
                self.theta_tuning_err[unit_num],
                self.phi_tuning_bins,
                self.phi_tuning[unit_num],
                self.phi_tuning_err[unit_num],
                self.eyeT,
                self.dEye,
                self.dEye_dps,
                self.theta,
                self.phi,
                self.ball_speed,
                self.ballspeed_tuning_bins,
                self.ballspeed_tuning[unit_num],
                self.ballspeed_tuning_err[unit_num],
                self.on_Sn_psth[unit_num,:,0],
                self.on_Sn_psth[unit_num,:,1],
                self.on_Sn_psth[unit_num,:,2],
                self.on_Sn_psth[unit_num,:,3],
                self.off_Sn_psth[unit_num,:,0],
                self.off_Sn_psth[unit_num,:,1],
                self.off_Sn_psth[unit_num,:,2],
                self.off_Sn_psth[unit_num,:,3],
                self.rf_xy[unit_num,:2],
                self.rf_xy[unit_num,2:],
                self.all_stimT,
                self.unit_stim_eventT[ind]['offSubunit_eventT_allT'],
                self.unit_stim_eventT[ind]['offSubunit_eventT_darkT'],
                self.unit_stim_eventT[ind]['offSubunit_eventT_lightT'],
                self.unit_stim_eventT[ind]['offSubunit_eventT_bckgndT'],
                self.unit_stim_eventT[ind]['onSubunit_eventT_allT'],
                self.unit_stim_eventT[ind]['onSubunit_eventT_darkT'],
                self.unit_stim_eventT[ind]['onSubunit_eventT_lightT'],
                self.unit_stim_eventT[ind]['onSubunit_eventT_bckgndT'],
                self.worldT.values
            ]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]

            unit_data = pd.concat([unit_data, unit_df], axis=0)

        data_out = pd.concat([self.cells, unit_data], axis=1)

        # Add a prefix to all column names
        data_out = data_out.add_prefix('Sn_')

        data_out['session'] = self.session_name

        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def save_as_dict(self):

        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        stim = 'Sn'
        save_dict = {
            'CRF': self.contrast_tuning_bins,
            'CRF_err': self.contrast_tuning,
            'CRF_bins':self.contrast_tuning_err,
            'STA': self.sta,
            'STV': self.stv,
            'saccade_rightT': self.all_eyeR,
            'saccade_leftT': self.all_eyeL,
            'saccade_rightPSTH': self.rightsacc_avg,
            'saccade_leftPSTH': self.leftsacc_avg,
            'PupilRadius_tuning': self.pupilradius_tuning,
            'PupilRadius_bins': self.pupilradius_tuning_bins,
            'PupilRadius_err': self.pupilradius_tuning_err,
            'Theta_tuning': self.theta_tuning,
            'Theta_bins': self.theta_tuning_bins,
            'Theta_err': self.theta_tuning_err,
            'Phi_tuning': self.phi_tuning,
            'Phi_bins': self.phi_tuning_bins,
            'Phi_err': self.phi_tuning_err,
            'eyeT': self.eyeT,
            'dEye': self.dEye_dps,
            'dEye_dpf': self.dEye,
            'theta': self.theta,
            'phi': self.phi,
            'ballspeed': self.ball_speed, 
            'ballspeed_tuning_bins': self.ballspeed_tuning_bins,
            'ballspeed_tuning': self.ballspeed_tuning,
            'ballspeed_tuning_err': self.ballspeed_tuning_err,
            'stim_PSTH_onSubReg': self.on_Sn_psth,
            'stim_PSTH_offSubReg': self.off_Sn_psth,
            'stim_PSTH_center_RF_used': self.rf_xy,
            'session': self.session_name,
            'stim': stim,
            'worldT': self.worldT.values
        }

        for key, val in save_dict.items():
            if type(val)==xr.DataArray:
                save_dict[key] = val.values
        
        # Merge Phy2 cell data in
        save_cells = self.cells.to_dict(orient='list')
        save_dict = {**save_cells, **save_dict}

        # Add 'FmLt' to the start of each key
        for old_key in save_dict.keys():
            save_dict['{}_{}'.format(stim,old_key)] = save_dict.pop(old_key)

        savepath = os.path.join(self.recording_path,
                                '{}_ephys_props.h5'.format(self.recording_name))

        fmEphys.write_h5(savepath, save_dict)

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        self.detail_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_diagnostic_analysis_figures.pdf')))
        
        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        self.calc_Sn_psth()

        print('closing pdfs')
        self.detail_pdf.close(); self.diagnostic_pdf.close()

        print('saving ephys file')
        self.save_as_df()
        
class HeadFixedGratings(fmEphys.Ephys):
    def __init__(self, cfg, recording_name, recording_path):
        fmEphys.Ephys.__init__(self, cfg, recording_name, recording_path)
        self.fm = False
        self.stim = 'gt'

        self.ori_x = np.arange(8)*45


    def stim_psth(self, lower=-0.5, upper=1.5, dt=0.1):
        """ Calculate and plot PSTH relative to stimulus onset
        """
        bins = np.arange(lower, upper+dt, dt)
        fig = plt.figure(figsize=(10, int(np.ceil(self.n_cells / 2))))
        # empty array into which psth will be saved
        psth = np.zeros([self.n_cells, len(bins)-1])
        # iterate through units
        for i, ind in enumerate(self.cells.index):
            plt.subplot(int(np.ceil(self.n_cells/4)), 4, i+1)
            # empty array for psth of this unit
            this_psth = np.zeros(len(bins)-1)
            for t in self.stim_start:
                # get a histogram of spike times in each of the stimulus bins
                hist, edges = np.histogram(self.cells.at[ind,'spikeT']-t, bins)
                # make this cumulative
                this_psth = this_psth + hist
            # normalize spikes in bins to the number of times the stim had an onset
            this_psth = this_psth / len(self.stim_start)
            # then normalize to length of time for each bin
            this_psth = this_psth / dt
            # plot histogram as a line
            plt.plot(bins[0:-1] + dt / 2, this_psth)
            plt.ylim(0, np.nanmax(this_psth) * 1.2)
            # add psth from this unit to array of all units
            psth[i,:] = this_psth
        plt.xlabel('time'); plt.ylabel('sp/sec')
        plt.title('gratings psth')
        plt.tight_layout(); plt.close()
        self.grating_psth = psth
        plt.tight_layout(); self.detail_pdf.savefig(); plt.close()

    def gratings_analysis(self, xrg=40, yrg=25):
        """
        xrg, yrg -- pixel range to define monitor
        """
        # setup
        nf = np.size(self.img_norm, 0) - 1
        u_mn = np.zeros((nf, 1)); v_mn = np.zeros((nf, 1))
        sx_mn = np.zeros((nf, 1)); sy_mn = np.zeros((nf, 1))
        flow_norm = np.zeros((nf, np.size(self.img_norm,1), np.size(self.img_norm,2), 2))
        
        # find screen
        meanx = np.mean(self.std_im>0, axis=0)
        xcent = np.int(np.sum(meanx * np.arange(len(meanx))) / np.sum(meanx))
        meany = np.mean(self.std_im>0, axis=1)
        ycent = np.int(np.sum(meany * np.arange(len(meany))) / np.sum(meany))

        # animation of optic flow
        fig, ax = plt.subplots(1,1,figsize = (16,8))
        for f in tqdm(range(nf)):
            frm = np.uint8(32*(self.img_norm[f,:,:]+4))
            frm2 = np.uint8(32*(self.img_norm[f+1,:,:]+4))
            flow_norm[f,:,:,:] = cv2.calcOpticalFlowFarneback(frm,frm2, None, 0.5, 3, 30, 3, 7, 1.5, 0)
            u = flow_norm[f,:,:,0]; v = -flow_norm[f,:,:,1] # negative to fix sign for y axis in images
            sx = cv2.Sobel(frm, cv2.CV_64F, 1, 0, ksize=11)
            sy = -cv2.Sobel(frm, cv2.CV_64F, 0, 1, ksize=11) # negative to fix sign for y axis in images
            sx[self.std_im<20] = 0; sy[self.std_im<20] = 0; # get rid of values outside of monitor
            sy[sx<0] = -sy[sx<0] # make vectors point in positive x direction (so opposite sides of grating don't cancel)
            sx[sx<0] = -sx[sx<0]
            sy[np.abs(sx/sy)<0.15] = np.abs(sy[np.abs(sx/sy)<0.15])
            u_mn[f] = np.mean(u[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg])
            v_mn[f]= np.mean(v[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg]); 
            sx_mn[f] = np.mean(sx[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg])
            sy_mn[f] = np.mean(sy[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg])
        scr_contrast = np.empty(self.worldT.size)
        for i in range(self.worldT.size):
            scr_contrast[i] = np.nanmean(np.abs(self.img_norm[i, ycent-25:ycent+25, xcent-40:xcent+40]))
        scr_contrast = scipy.signal.medfilt(scr_contrast, 11)
        stimOn = np.double(scr_contrast>0.5)
        self.stim_start = np.array(self.worldT[np.where(np.diff(stimOn)>0)])
        
        # shift everything forward so that t=0 is centered between frame 0 and frame 1
        self.stim_onsets_ = self.stim_start.copy()

        stim_end = np.array(self.worldT[np.where(np.diff(stimOn)<0)])
        stim_end = stim_end[stim_end>self.stim_start[0]]
        self.stim_start = self.stim_start[self.stim_start<stim_end[-1]]
        grating_th = np.zeros(len(self.stim_start))
        grating_mag = np.zeros(len(self.stim_start))
        grating_dir = np.zeros(len(self.stim_start))
        dI = np.zeros(len(self.stim_start))
        for i in range(len(self.stim_start)):
            tpts = np.where((self.worldT>self.stim_start[i] + 0.025) & (self.worldT<stim_end[i]-0.025))
            mag = np.sqrt(sx_mn[tpts]**2 + sy_mn[tpts]**2)
            this = np.where(mag[:,0] > np.percentile(mag,25))
            goodpts = np.array(tpts)[0,this]
            stim_sx = np.nanmedian(sx_mn[tpts])
            stim_sy = np.nanmedian(sy_mn[tpts])
            stim_u = np.nanmedian(u_mn[tpts])
            stim_v = np.nanmedian(v_mn[tpts])
            grating_th[i] = np.arctan2(stim_sy, stim_sx)
            grating_mag[i] = np.sqrt(stim_sx**2 + stim_sy**2)
            grating_dir[i] = np.sign(stim_u*stim_sx + stim_v*stim_sy) # dot product of gratient and flow gives direction
            dI[i] = np.mean(np.diff(self.img_norm[tpts, ycent, xcent])**2) # rate of change of image give temporal frequency
        self.grating_ori = grating_th.copy()
        self.grating_ori[grating_dir<0] = self.grating_ori[grating_dir<0] + np.pi
        self.grating_ori = self.grating_ori - np.min(self.grating_ori)
        grating_tf = np.zeros(len(self.stim_start))
        grating_tf[dI>0.5] = 1;  # spatial frequencies: 0=low, 1=high
        ori_cat = np.floor((self.grating_ori+np.pi/16) / (np.pi/4))
        
        plt.figure()
        plt.plot(range(15), ori_cat[:15]); plt.xlabel('first 15 stims'); plt.ylabel('ori cat')
        self.diagnostic_pdf.savefig()

        km = sklearn.cluster.KMeans(n_clusters=3).fit(np.reshape(grating_mag, (-1,1)))
        sf_cat = km.labels_
        order = np.argsort(np.reshape(km.cluster_centers_, 3))
        sf_catnew = sf_cat.copy()
        for i in range(3):
            sf_catnew[sf_cat == order[i]] = i
        self.sf_cat = sf_catnew.copy()

        plt.figure(figsize=(8,8))
        plt.scatter(grating_mag, self.grating_ori, c=ori_cat)
        plt.xlabel('grating magnitude'); plt.ylabel('theta')
        self.diagnostic_pdf.savefig(); plt.close()

        ntrial = np.zeros((3,8))
        for i in range(3):
            for j in range(8):
                ntrial[i,j] = np.sum((sf_cat==i) & (ori_cat==j))
        plt.figure()
        plt.imshow(ntrial, vmin=0, vmax=2*np.mean(ntrial))
        plt.colorbar(); plt.xlabel('orientations')
        plt.ylabel('sfs'); plt.title('trials per condition')
        self.diagnostic_pdf.savefig(); plt.close()

        # plotting grating orientation and tuning curves
        edge_win = 0.025
        self.grating_rate = np.zeros((len(self.cells), len(self.stim_start)))
        self.spont_rate = np.zeros((len(self.cells), len(self.stim_start)))
        self.ori_tuning = np.zeros((len(self.cells), 8, 3))
        self.ori_tuning_tf = np.zeros((len(self.cells), 8, 3, 2))
        self.drift_spont = np.zeros(len(self.cells))
        plt.figure(figsize=(12, self.n_cells*2))
        for c, ind in enumerate(self.cells.index):
            sp = self.cells.at[ind,'spikeT'].copy()
            for i in range(len(self.stim_start)):
                self.grating_rate[c, i] = np.sum((sp > self.stim_start[i]+edge_win) & (sp < stim_end[i])) / (stim_end[i] - self.stim_start[i] - edge_win)
            for i in range(len(self.stim_start)-1):
                self.spont_rate[c, i] = np.sum((sp > stim_end[i]+edge_win) & (sp < self.stim_start[i+1])) / (self.stim_start[i+1] - stim_end[i] - edge_win)  
            for ori in range(8):
                for sf in range(3):
                    self.ori_tuning[c, ori, sf] = np.mean(self.grating_rate[c, (ori_cat==ori) & (sf_cat==sf)])
                    for tf in range(2):
                        self.ori_tuning_tf[c, ori, sf, tf] = np.mean(self.grating_rate[c, (ori_cat==ori) & (sf_cat ==sf) & (grating_tf==tf)])
            self.drift_spont[c] = np.mean(self.spont_rate[c, :])
            plt.subplot(self.n_cells, 4, 4*c+1)
            plt.scatter(self.grating_ori, self.grating_rate[c,:], c=sf_cat)
            plt.plot(3*np.ones(len(self.spont_rate[c,:])), self.spont_rate[c,:], 'r.')
            
            plt.subplot(self.n_cells, 4, 4*c+2)
            plt.plot(self.ori_x, self.ori_tuning[c,:,0], label='low sf')
            plt.plot(self.ori_x, self.ori_tuning[c,:,1], label='mid sf')
            plt.plot(self.ori_x, self.ori_tuning[c,:,2], label='high sf')
            plt.plot([np.min(self.ori_x), np.max(self.ori_x)], [self.drift_spont[c], self.drift_spont[c]], 'r:', label='spont')
            try:
                plt.ylim(0, np.nanmax(self.ori_tuning_tf[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.legend()

            plt.subplot(self.n_cells, 4, 4*c+3)
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,0,0], label='low sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,1,0], label='mid sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,2,0], label='high sf')
            plt.plot([np.min(self.ori_x), np.max(self.ori_x)], [self.drift_spont[c], self.drift_spont[c]], 'r:',label ='spont')
            try:
                plt.ylim(0, np.nanmax(self.ori_tuning_tf[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.legend()

            plt.subplot(self.n_cells, 4, 4*c+4)
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,0,1], label='low sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,1,1], label='mid sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,2,1], label='high sf')
            plt.plot([np.min(self.ori_x), np.max(self.ori_x)], [self.drift_spont[c], self.drift_spont[c]], 'r:', label='spont')
            try:
                plt.ylim(0, np.nanmax(self.ori_tuning_tf[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.legend()
        
        plt.tight_layout(); self.detail_pdf.savefig(); plt.close()

        # roll orientation tuning curves
        # ori_cat maps orientations so that ind=0 is the bottom-right corner of the monitor
        # index of sf_cat ascend moving counter-clockwise
        # ind=1 are rightward gratings; ind=5 are leftward gratings

        # shape is (cell, ori, sf), so rolling axis=1 shifts orientations so make rightward gratings 0deg
        self.ori_tuning_meantf = np.roll(self.ori_tuning, shift=-1, axis=1)
        # shape is (cell, ori, sf, tf), so again roll axis=1 to fix gratings orientations
        self.ori_tuning_tf = np.roll(self.ori_tuning_tf, shift=-1, axis=1)

        for c in range(np.size(self.ori_tuning_tf,0)):

            _tuning = self.ori_tuning_tf[c,:,:,:].copy()

            self.ori_tuning_tf[c,:,:,:] = np.roll(_tuning, 1, axis=1)

    def gratings_psths(self):
        self.gt_kde_psth = np.zeros([len(self.cells.index.values),
                            3001])*np.nan
        for i, ind in enumerate(self.cells.index.values):
            _spikeT = self.cells.loc[ind,'spikeT']
            self.gt_kde_psth[i,:] = self.calc_kde_PSTH(_spikeT, self.stim_onsets_, edgedrop=30, win=1500)

    def save_as_df(self):

        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        unit_data = pd.DataFrame([])
        for unit_num, ind in enumerate(self.cells.index):
            cols = [
                'contrast',
                'contrast_tuning_bins',
                'contrast_tuning',
                'contrast_tuning_err',
                'spike_triggered_average',
                'spike_triggered_variance',
                'saccade_rightT',
                'saccade_leftT',
                'saccade_rightPSTH',
                'saccade_leftPSTH',
                'pupilradius_tuning_bins',
                'pupilradius_tuning',
                'pupilradius_tuning_err',
                'theta_tuning_bins',
                'theta_tuning',
                'theta_tuning_err',
                'phi_tuning_bins',
                'phi_tuning',
                'phi_tuning_err',
                'eyeT',
                'dEye_dpf',
                'dEye_dps',
                'theta',
                'phi',
                'ballspeed',
                'ballspeed_tuning_bins',
                'ballspeed_tuning',
                'ballspeed_tuning_err',
                'grating_psth',
                'grating_ori',
                'ori_tuning_mean_tf',
                'ori_tuning_tf',
                'drift_spont',
                'spont_rate',
                'grating_rate',
                'sf_cat',
                'stim_PSTH',
                'stimT',
                'worldT'
            ]
            unit_df = pd.DataFrame(pd.Series([
                self.contrast,
                self.contrast_tuning_bins,
                self.contrast_tuning[unit_num],
                self.contrast_tuning_err[unit_num],
                self.sta[unit_num],
                self.stv[unit_num],
                self.all_eyeR,
                self.all_eyeL,
                self.rightsacc_avg[unit_num],
                self.leftsacc_avg[unit_num],
                self.pupilradius_tuning_bins,
                self.pupilradius_tuning[unit_num],
                self.pupilradius_tuning_err[unit_num],
                self.theta_tuning_bins,
                self.theta_tuning[unit_num],
                self.theta_tuning_err[unit_num],
                self.phi_tuning_bins,
                self.phi_tuning[unit_num],
                self.phi_tuning_err[unit_num],
                self.eyeT,
                self.dEye,
                self.dEye_dps,
                self.theta,
                self.phi,
                self.ball_speed,
                self.ballspeed_tuning_bins,
                self.ballspeed_tuning[unit_num],
                self.ballspeed_tuning_err[unit_num],
                self.gt_kde_psth[unit_num],
                self.grating_ori,
                self.ori_tuning_meantf[unit_num],
                self.ori_tuning_tf[unit_num],
                self.drift_spont[unit_num],
                self.spont_rate[unit_num],
                self.grating_rate[unit_num],
                self.sf_cat[unit_num],
                self.gt_kde_psth[unit_num],
                self.stim_onsets_,
                self.worldT.values
                ]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]

            unit_data = pd.concat([unit_data, unit_df], axis=0)

        data_out = pd.concat([self.cells, unit_data], axis=1)

        # Add a prefix to all column names
        data_out = data_out.add_prefix('Gt_')

        data_out['session'] = self.session_name

        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def save_as_dict(self):

        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        stim = 'Gt'
        save_dict = {
            'CRF': self.contrast_tuning_bins,
            'CRF_err': self.contrast_tuning,
            'CRF_bins':self.contrast_tuning_err,
            'STA': self.sta,
            'STV': self.stv,
            'saccade_rightT': self.all_eyeR,
            'saccade_leftT': self.all_eyeL,
            'saccade_rightPSTH': self.rightsacc_avg,
            'saccade_leftPSTH': self.leftsacc_avg,
            'PupilRadius_tuning': self.pupilradius_tuning,
            'PupilRadius_bins': self.pupilradius_tuning_bins,
            'PupilRadius_err': self.pupilradius_tuning_err,
            'Theta_tuning': self.theta_tuning,
            'Theta_bins': self.theta_tuning_bins,
            'Theta_err': self.theta_tuning_err,
            'Phi_tuning': self.phi_tuning,
            'Phi_bins': self.phi_tuning_bins,
            'Phi_err': self.phi_tuning_err,
            'eyeT': self.eyeT,
            'dEye': self.dEye_dps,
            'dEye_dpf': self.dEye,
            'theta': self.theta,
            'phi': self.phi,
            'ballspeed': self.ball_speed, 
            'ballspeed_tuning_bins': self.ballspeed_tuning_bins,
            'ballspeed_tuning': self.ballspeed_tuning,
            'ballspeed_tuning_err': self.ballspeed_tuning_err,
            'stim_psth': self.gt_kde_psth,
            'stim_tuning': self.ori_tuning_tf,
            'stim_onsets': self.stim_onsets_,
            'stim_rate': self.grating_rate,
            'stim_ori': self.grating_ori,
            'drift_spont': self.drift_spont,
            'spont_rate': self.spont_rate,
            'sf_cat': self.sf_cat,
            'session': self.session_name,
            'stim': stim,
            'worldT': self.worldT.values
        }

        for key, val in save_dict.items():
            if type(val)==xr.DataArray:
                save_dict[key] = val.values
        
        # Merge Phy2 cell data in
        save_cells = self.cells.to_dict(orient='list')
        save_dict = {**save_cells, **save_dict}

        # Add 'FmLt' to the start of each key
        for old_key in save_dict.keys():
            save_dict['{}_{}'.format(stim,old_key)] = save_dict.pop(old_key)

        savepath = os.path.join(self.recording_path,
                                '{}_ephys_props.h5'.format(self.recording_name))

        fmEphys.write_h5(savepath, save_dict)

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        # self.overview_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_overview_analysis_figures.pdf')))
        self.detail_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_diagnostic_analysis_figures.pdf')))

        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        print('running analysis for gratings stimulus')
        self.gratings_analysis()

        print('PSTHs')
        self.gratings_psths()

        print('closing pdfs')
        self.detail_pdf.close()
        self.diagnostic_pdf.close()

        print('saving ephys file')
        self.save_as_df()