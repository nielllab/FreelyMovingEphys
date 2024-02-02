"""
fmEphys/utils/gratings.py

Head-fixed Drifting Gratings response properties.


Written by DMM, 2021
"""


import os
import cv2
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.interpolate
import scipy.signal
import sklearn.cluster

import fmEphys as fme


class HeadFixedGratings(fme.Ephys):


    def __init__(self, cfg, recording_name, recording_path):

        fme.Ephys.__init__(self, cfg, recording_name, recording_path)
        self.fm = False
        self.stim = 'gt'

        self.ori_x = np.arange(8)*45


    def stim_psth(self, lower=-0.5, upper=1.5, dt=0.1):
        """ Calculate and plot PSTH relative to stimulus onset.

        This is not used anymore -- PSTHs now calculated using kernel
        density estimation method in Ephys class, which HeadFixedGratings
        inherits from.
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

        plt.xlabel('time')
        plt.ylabel('sp/sec')
        plt.title('gratings psth')
        plt.tight_layout()
        plt.close()

        self.grating_psth = psth

        plt.tight_layout()
        self.detail_pdf.savefig()
        plt.close()


    def gratings_analysis(self):
        """ Analyze drifting grating stimulus.
        """

        # Pixel range to define screen
        xrg=40
        yrg=25

        # Initialize arrays for stimulus data
        nf = np.size(self.img_norm, 0) - 1

        u_mn = np.zeros((nf, 1))
        v_mn = np.zeros((nf, 1))

        sx_mn = np.zeros((nf, 1))
        sy_mn = np.zeros((nf, 1))

        flow_norm = np.zeros((nf,
                              np.size(self.img_norm,1),
                              np.size(self.img_norm,2),
                              2))
        
        # Find screen
        meanx = np.mean(self.std_im > 0, axis=0)
        xcent = np.int(np.sum(meanx * np.arange(len(meanx))) / np.sum(meanx))
        
        meany = np.mean(self.std_im > 0, axis=1)
        ycent = np.int(np.sum(meany * np.arange(len(meany))) / np.sum(meany))


        # Optic flow
        fig, ax = plt.subplots(1, 1, figsize=(16,8))

        for f in tqdm(range(nf)):

            # two frames to compare when calculating optic flow
            frm = np.uint8(32*(self.img_norm[f,:,:]+4))
            frm2 = np.uint8(32*(self.img_norm[f+1,:,:]+4))

            flow_norm[f,:,:,:] = cv2.calcOpticalFlowFarneback(
                                        frm,  # current frame
                                        frm2, # next frame
                                        None, # no preexisting flow data
                                        0.5,  # pyramid image scale
                                        3,    # number of pyramid levels
                                        30,   # window size
                                        3,    # num iterations
                                        7,    # pixel neighborhood size
                                        1.5,  # gaussian std
                                        0)     # flags
            
            # Flow vectors
            u = flow_norm[f,:,:,0]

            # Vector v has negative to fix sign for y axis
            # in images (same is done for sy below)
            v = -flow_norm[f,:,:,1]

            sx = cv2.Sobel(frm, cv2.CV_64F, 1, 0, ksize=11)

            sy = -cv2.Sobel(frm, cv2.CV_64F, 0, 1, ksize=11)

            # Get rid of values outside of monitor
            sx[self.std_im < 20] = 0
            sy[self.std_im < 20] = 0

            # Make vectors point in positive x direction
            # so opposite sides of grating don't cancel
            sy[sx < 0] = -sy[sx < 0] 
            sx[sx < 0] = -sx[sx < 0]

            sy[np.abs(sx/sy) < 0.15] = np.abs(sy[np.abs(sx/sy) < 0.15])

            u_mn[f] = np.mean(u[ycent-yrg:ycent+yrg,
                                xcent-xrg:xcent+xrg])
            
            v_mn[f]= np.mean(v[ycent-yrg:ycent+yrg,
                               xcent-xrg:xcent+xrg])
            
            sx_mn[f] = np.mean(sx[ycent-yrg:ycent+yrg,
                                  xcent-xrg:xcent+xrg])
            
            sy_mn[f] = np.mean(sy[ycent-yrg:ycent+yrg,
                                  xcent-xrg:xcent+xrg])

        scr_contrast = np.empty(self.worldT.size)

        for i in range(self.worldT.size):
            scr_contrast[i] = np.nanmean(np.abs(self.img_norm[i,
                                                              ycent-25:ycent+25,
                                                              xcent-40:xcent+40]))
            
        scr_contrast = scipy.signal.medfilt(scr_contrast, 11)

        stimOn = np.double(scr_contrast > 0.5)

        self.stim_start = np.array(self.worldT[np.where(np.diff(stimOn) > 0)])
        
        # Shift everything forward so that t=0 is centered
        # between frame 0 and frame 1
        self.stim_onsets_ = self.stim_start.copy()

        stim_end = np.array(self.worldT[np.where(np.diff(stimOn) < 0)])
        stim_end = stim_end[stim_end > self.stim_start[0]]

        self.stim_start = self.stim_start[self.stim_start < stim_end[-1]]

        grating_th = np.zeros(len(self.stim_start))
        grating_mag = np.zeros(len(self.stim_start))
        grating_dir = np.zeros(len(self.stim_start))
        dI = np.zeros(len(self.stim_start))

        for i in range(len(self.stim_start)):
            
            if i >= len(self.stim_start):
                continue

            tpts = np.where((self.worldT > self.stim_start[i] + 0.025)
                          & (self.worldT < stim_end[i] - 0.025))
            
            if np.size(tpts)==0:
                
                # if a spurious stimulus start, with no presented frames in the worldcam video, was
                # included as a stim presentation, throw it out and shorten the returned arrays by one
                # stimulus event

                grating_th = grating_th[:-1]
                grating_mag = grating_mag[:-1]
                grating_dir = grating_dir[:-1]
                dI = dI[:-1]

                self.stim_start = np.delete(self.stim_start, obj=i, axis=0)
                continue

            
            mag = np.sqrt(sx_mn[tpts]**2 + sy_mn[tpts]**2)

            this = np.where(mag[:,0] > np.percentile(mag,25))

            goodpts = np.array(tpts)[0,this]

            stim_sx = np.nanmedian(sx_mn[tpts])
            stim_sy = np.nanmedian(sy_mn[tpts])
            stim_u = np.nanmedian(u_mn[tpts])
            stim_v = np.nanmedian(v_mn[tpts])

            grating_th[i] = np.arctan2(stim_sy, stim_sx)
            grating_mag[i] = np.sqrt(stim_sx**2 + stim_sy**2)

            # Dot product of gratient and flow gives direction
            grating_dir[i] = np.sign(stim_u*stim_sx + stim_v*stim_sy)

            # Rate of change of image give temporal frequency
            dI[i] = np.mean(np.diff(self.img_norm[tpts, ycent, xcent])**2)

        self.grating_ori = grating_th.copy()

        self.grating_ori[grating_dir < 0] = self.grating_ori[grating_dir < 0] + np.pi

        self.grating_ori = self.grating_ori - np.min(self.grating_ori)

    
        # Spatial frequencies: 0=low, 1=high
        grating_tf = np.zeros(len(self.stim_start))
        grating_tf[dI > 0.5] = 1

        # Category of orientations
        ori_cat = np.floor((self.grating_ori+np.pi/16) / (np.pi/4))
        
        plt.figure()
        plt.plot(range(15), ori_cat[:15])
        plt.xlabel('first 15 stims')
        plt.ylabel('ori cat')
        self.diagnostic_pdf.savefig()
        plt.close()

        # Cluster spatial frequencies
        km = sklearn.cluster.KMeans(n_clusters=3).fit(np.reshape(grating_mag, (-1,1)))
        sf_cat = km.labels_

        order = np.argsort(np.reshape(km.cluster_centers_, 3))

        sf_catnew = sf_cat.copy()

        for i in range(3):
            sf_catnew[sf_cat == order[i]] = i

        self.sf_cat = sf_catnew.copy()

        plt.figure(figsize=(8,8))
        plt.scatter(grating_mag, self.grating_ori, c=ori_cat)
        plt.xlabel('grating magnitude')
        plt.ylabel('theta')
        self.diagnostic_pdf.savefig()
        plt.close()

        ntrial = np.zeros((3,8))

        for i in range(3):
            for j in range(8):
                ntrial[i, j] = np.sum((sf_cat==i) & (ori_cat==j))

        plt.figure()
        plt.imshow(ntrial, vmin=0, vmax=2*np.mean(ntrial))
        plt.colorbar()
        plt.xlabel('orientations')
        plt.ylabel('sfs')
        plt.title('trials per condition')
        self.diagnostic_pdf.savefig()
        plt.close()

        # Plot grating orientations and tuning curves
        edge_win = 0.025
        self.grating_rate = np.zeros((len(self.cells), len(self.stim_start)))
        self.spont_rate = np.zeros((len(self.cells), len(self.stim_start)))
        self.ori_tuning = np.zeros((len(self.cells), 8, 3))
        self.ori_tuning_tf = np.zeros((len(self.cells), 8, 3, 2))
        self.drift_spont = np.zeros(len(self.cells))

        plt.figure(figsize=(12, self.n_cells * 2))

        for c, ind in enumerate(self.cells.index):

            sp = self.cells.at[ind,'spikeT'].copy()

            for i in range(len(self.stim_start)):

                self.grating_rate[c,i] = np.sum((sp > self.stim_start[i]+edge_win) &                        \
                            (sp < stim_end[i])) / (stim_end[i] - self.stim_start[i] - edge_win)
            
            for i in range(len(self.stim_start)-1):

                self.spont_rate[c,i] = np.sum((sp > stim_end[i]+edge_win) &
                            (sp < self.stim_start[i+1])) / (self.stim_start[i+1] - stim_end[i] - edge_win)  
            
            for ori in range(8):

                for sf in range(3):

                    self.ori_tuning[c,ori,sf] = np.mean(self.grating_rate[c,
                                                                (ori_cat==ori) & (sf_cat==sf)])
                    
                    for tf in range(2):
                        self.ori_tuning_tf[c,ori,sf,tf] = np.mean(self.grating_rate[c,
                                                                (ori_cat==ori) & (sf_cat ==sf) & (grating_tf==tf)])
            
            self.drift_spont[c] = np.mean(self.spont_rate[c, :])

            plt.subplot(self.n_cells, 4, 4*c+1)

            plt.scatter(self.grating_ori, self.grating_rate[c,:], c=sf_cat)
            plt.plot(3*np.ones(len(self.spont_rate[c,:])),
                     self.spont_rate[c,:], 'r.')
            
            plt.subplot(self.n_cells, 4, 4*c+2)
            plt.plot(self.ori_x, self.ori_tuning[c,:,0], label='low sf')
            plt.plot(self.ori_x, self.ori_tuning[c,:,1], label='mid sf')
            plt.plot(self.ori_x, self.ori_tuning[c,:,2], label='high sf')

            plt.plot([np.min(self.ori_x), np.max(self.ori_x)],
                     [self.drift_spont[c], self.drift_spont[c]],
                     'r:', label='spont')
            
            try:
                plt.ylim(0, np.nanmax(self.ori_tuning_tf[c,:,:]*1.2))
            
            except ValueError:
                plt.ylim(0,1)

            plt.legend()

            plt.subplot(self.n_cells, 4, 4*c+3)
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,0,0], label='low sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,1,0], label='mid sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,2,0], label='high sf')

            plt.plot([np.min(self.ori_x), np.max(self.ori_x)],
                     [self.drift_spont[c], self.drift_spont[c]], 'r:',label ='spont')
            
            try:
                plt.ylim(0, np.nanmax(self.ori_tuning_tf[c,:,:]*1.2))
                
            except ValueError:
                plt.ylim(0,1)
            
            plt.legend()

            plt.subplot(self.n_cells, 4, 4*c+4)
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,0,1], label='low sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,1,1], label='mid sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,2,1], label='high sf')

            plt.plot([np.min(self.ori_x), np.max(self.ori_x)],
                     [self.drift_spont[c], self.drift_spont[c]], 'r:', label='spont')
            
            try:
                plt.ylim(0, np.nanmax(self.ori_tuning_tf[c,:,:]*1.2))

            except ValueError:
                plt.ylim(0,1)

            plt.legend()
        
        plt.tight_layout()
        self.detail_pdf.savefig()
        plt.close()

        # roll orientation tuning curves
        # ori_cat maps orientations so that ind=0 is the bottom-right corner of the monitor
        # index of sf_cat ascend moving counter-clockwise
        # ind=1 are rightward gratings
        # ind=5 are leftward gratings

        # shape is (cell, ori, sf), so rolling axis=1 shifts orientations so make rightward gratings 0deg
        self.ori_tuning_meantf = np.roll(self.ori_tuning, shift=-1, axis=1)

        # shape is (cell, ori, sf, tf), so again roll axis=1 to fix gratings orientations
        self.ori_tuning_tf = np.roll(self.ori_tuning_tf, shift=-1, axis=1)

        for c in range(np.size(self.ori_tuning_tf,0)):

            _tuning = self.ori_tuning_tf[c,:,:,:].copy()

            self.ori_tuning_tf[c,:,:,:] = np.roll(_tuning, 1, axis=1)


    def gratings_psths(self):

        self.gt_kde_psth = np.zeros([len(self.cells.index.values), 3001]) * np.nan

        for i, ind in enumerate(self.cells.index.values):

            _spikeT = self.cells.loc[ind,'spikeT']

            self.gt_kde_psth[i,:] = self.calc_kde_PSTH(_spikeT,
                                                       self.stim_onsets_,
                                                       edgedrop=30,
                                                       win=1500)


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
                ]),
                dtype=object).T
            
            unit_df.columns = cols
            unit_df.index = [ind]

            unit_data = pd.concat([unit_data, unit_df], axis=0)

        data_out = pd.concat([self.cells, unit_data], axis=1)

        # Add a prefix to all column names
        data_out = data_out.add_prefix('Gt_')

        data_out['session'] = self.session_name

        data_out.to_hdf(os.path.join(self.recording_path,
                       (self.recording_name+'_ephys_props.h5')), 'w')


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

        fme.write_h5(savepath, save_dict)

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path,
                                   (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path,
                                   (self.recording_name+'_ephys_props.h5')))

        self.detail_pdf = PdfPages(os.path.join(self.recording_path,
                                    (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path,
                                    (self.recording_name + '_diagnostic_analysis_figures.pdf')))

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

