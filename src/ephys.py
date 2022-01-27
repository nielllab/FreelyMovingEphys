"""
FreelyMovingEphys/src/ephys.py
"""
import os, json, cv2, platform, subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm
import scipy.linalg as linalg
import scipy.sparse as sparse
import scipy.signal as signal
from scipy.interpolate import interp1d
import wavio
from sklearn.linear_model import LinearRegression
from scipy.ndimage import shift as imshift
from itertools import chain
if platform.system() == 'Linux':
    mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
else:
    mpl.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

from src.base import BaseInput
from src.utils.correlation import nanxcorr
from src.utils.path import find, list_subdirs

class Ephys(BaseInput):
    def __init__(self, config, recording_name, recording_path):
        BaseInput.__init__(self, config, recording_name, recording_path)

        self.channel_map_path = self.config['paths']['channel_map_path']

        self.highlight_neuron = self.config['options']['neuron_to_highlight']
        self.save_diagnostic_video = self.config['options']['ephys_videos']
        self.do_rough_glm_fit = self.config['internals']['do_rough_glm_fit']
        self.do_glm_model_preprocessing = self.config['internals']['do_glm_model_preprocessing']
        self.probe = self.config['options']['probe']
        self.num_channels = next(int(num) for num in ['128','64','16'] if num in self.probe)
        self.ephys_samprate = self.config['internals']['ephys_samprate']

        # timebase
        self.model_dt = 0.025
        self.trange = np.arange(-1, 1.1, self.model_dt)
        self.trange_x = 0.5*(self.trange[0:-1]+ self.trange[1:])
        self.model_eye_use_thresh = 10
        self.model_active_thresh = 40
        self.darkness_thresh = 100
        self.contrast_range = np.arange(0,1.2,0.1)

        self.default_ephys_offset = 0.1
        self.default_ephys_drift_rate = -0.000114

        self.session_name = '_'.join(self.recording_name.split('_')[:4])
    
    def gather_base_files(self):
        self.reye_path = os.path.join(self.recording_path, self.recording_name + '_REYE.nc')
        if not os.path.isfile(self.reye_path):
            self.reye_path = os.path.join(self.recording_path, self.recording_name + '_Reye.nc')
        self.world_path = os.path.join(self.recording_path, self.recording_name + '_world.nc')
        self.ephys_json_path = os.path.join(self.recording_path, self.recording_name + '_ephys_merge.json')
        self.ephys_bin_path = os.path.join(self.recording_path, self.recording_name + '_Ephys.bin')

    def gather_hf_files(self):
        self.gather_base_files()
        self.running_ball_path = os.path.join(self.recording_path, self.recording_name + '_speed.nc')

    def gather_fm_files(self):
        self.gather_base_files()
        self.topcam_path = os.path.join(self.recording_path, self.recording_name + '_TOP1.nc')
        self.imu_path = os.path.join(self.recording_path, self.recording_name + '_imu.nc')

    def read_binary_file(self, do_remap=True):
        """ Read in ephys binary and remap channels.

        Parameters:
        do_remap (bool): if True, remap channels while reading in binary

        Returns:
        ephys (pd.DataFrame): ephys data with shape (time, channel)
        """
        # set up data types to read binary file into
        dtypes = np.dtype([('ch'+str(i),np.uint16) for i in range(0,self.num_channels)])
        # read in binary file
        ephys = pd.DataFrame(np.fromfile(self.ephys_bin_path, dtypes, -1, ''))
        if do_remap:
            # open channel map file
            with open(self.channel_map_path, 'r') as fp:
                all_maps = json.load(fp)
            # get channel map for the current probe
            ch_map = all_maps[self.probe]
            # remap with known order of channels
            ephys = ephys.iloc[:,[i-1 for i in list(ch_map)]]
        return ephys

    def butter_bandpass(self, lfp, lowcut=1, highcut=300, fs=30000, order=5):
        """ Apply bandpass filter to ephys LFP along time dimension, axis=0.

        Parameters:
        lfp (np.array): ephys LFP with shape (time, channel)
        lowcut (int): low end of frequency cut off
        highcut (int): high end of frequency cut off
        fs (int): sample rate
        order (int): order of filter

        Returns:
        filt (np.array): filtered data with shape (time, channel)
        """
        nyq = 0.5 * fs # Nyquist frequency
        low = lowcut / nyq # low cutoff
        high = highcut / nyq # high cutoff
        sos = butter(order, [low, high], btype='bandpass', output='sos')
        filt = sosfiltfilt(sos, lfp, axis=0)
        return filt

    def spike_raster(self):
        fig, ax = plt.subplots()
        for i, ind in enumerate(self.cells.index):
            # array of spike times
            sp = np.array(self.cells.at[ind,'spikeT'])
            # make vertical line for each time the unit fires
            plt.vlines(sp[sp<10], i-0.25, i+0.25)
            plt.xlim(0, 10) # in sec
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        plt.xlabel('secs'); plt.ylabel('unit number')
        plt.ylim([self.n_cells, 0])
        self.detail_pdf.savefig(); plt.close()

    def eye_position(self):
        good_pts = np.sum(~np.isnan(self.theta))/len(self.theta)
        plt.figure()
        plt.plot(self.theta, self.phi, 'k.', markersize=4)
        plt.xlabel('theta'); plt.ylabel('phi')
        plt.title('frac good='+str(np.round(good_pts,3)))
        self.detail_pdf.savefig(); plt.close()

    def check_theta(self):
        # flip the order of frames in an every-other fashion
        th_switch = np.zeros(np.shape(self.theta))
        th_switch[0:-1:2] = np.array(self.theta[1::2])
        th_switch[1::2] = np.array(self.theta[0:-1:2])
        # plot will be of 5sec starting 35sec into the video
        start = 35*60; stop = 40*60
        fig, ax = plt.subplots(121)
        plt.subplot(1,2,1)
        plt.plot(self.theta[start:stop]); plt.title('theta')
        plt.subplot(1,2,2)
        plt.plot(th_switch[start:stop]); plt.title('theta switch')
        plt.tight_layout()
        self.diagnostic_pdf.savefig(); plt.close()

    def check_imu_eye_alignment(self, t1, offset, ccmax):
        plt.subplot(1,2,1)
        plt.plot(self.eyeT[t1*60], offset)
        plt.xlabel('secs'); plt.ylabel('offset (secs)')
        plt.subplot(1,2,2)
        plt.plot(self.eyeT[t1*60], ccmax)
        plt.xlabel('secs'); plt.ylabel('max cc')
        plt.tight_layout()
        self.diagnostic_pdf.savefig(); plt.close()

    def plot_regression_timing_fit(self, dataT, offset):
        dataT = dataT[~np.isnan(dataT)]
        offset = offset[~np.isnan(dataT)]

        plt.figure()
        plt.plot(dataT, offset, 'k.')
        plt.plot(dataT, self.ephys_offset + dataT * self.ephys_drift_rate, color='r')
        plt.xlabel('secs'); plt.ylabel('offset (secs)')
        plt.title('offset0='+str(np.round(self.ephys_offset, 3))+' drift rate='+str(np.round(self.ephys_drift_rate, 3)))
        self.diagnostic_pdf.savefig(); plt.close()

    def head_and_eye_diagnostics(self):
        plt.figure()
        plt.plot(self.eyeT[:-1], np.diff(self.theta), label='dTheta')
        plt.plot(self.imuT-0.1, (self.gyro_z_raw-3)*10, label='raw gyro z')
        plt.xlim(30,40); plt.ylim(-12,12); plt.legend(); plt.xlabel('secs')
        self.diagnostic_pdf.savefig(); plt.close()

        gyro_z_interp = interp1d(self.imuT, self.gyro_z, bounds_error=False)
        plt.subplots(1,2)
        plt.subplot(1,2,1)
        plt.plot(self.eyeT[0:-1], self.dEye, label='dEye')
        plt.plot(self.eyeT, gyro_z_interp(self.eyeT), label='dHead')
        plt.xlim(37,39); plt.ylim(-10,10)
        plt.legend(); plt.ylabel('deg'); plt.xlabel('secs')
        plt.subplot(1,2,2)
        plt.plot(self.eyeT[0:-1], np.nancumsum(gyro_z_interp(self.eyeT[0:-1])), label='head position')
        plt.plot(self.eyeT[0:-1], np.nancumsum(gyro_z_interp(self.eyeT[0:-1])+self.dEye), label='gaze position')
        plt.plot(self.eyeT[1:], self.theta[0:-1], label='eye position')
        plt.xlim(35,40); plt.ylim(-30,30); plt.legend(); plt.ylabel('deg'); plt.xlabel('secs')
        plt.tight_layout()
        self.diagnostic_pdf.savefig(); plt.close()

    def estimate_shift_worldcam(self, max_frames=3600, num_iter=5000, term_eps=1e-4):
        # get eye displacement for each worldcam frame
        th_interp = interp1d(self.eyeT, self.theta, bounds_error=False)
        phi_interp = interp1d(self.eyeT, self.phi, bounds_error=False)
        dTheta = np.diff(th_interp(self.worldT))
        dPhi = np.diff(phi_interp(self.worldT))
        # calculate x-y shift for each worldcam frame  
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, term_eps)
        warp_mode = cv2.MOTION_TRANSLATION
        cc = np.zeros(max_frames)
        xshift = np.zeros(max_frames)
        yshift = np.zeros(max_frames)
        warp_all = np.zeros([6, max_frames])
        # get shift between adjacent frames
        for i in tqdm(range(max_frames)):
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try: 
                (cc[i], warp_matrix) = cv2.findTransformECC(self.world_vid[i,:,:], self.world_vid[i+1,:,:], warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
                xshift[i] = warp_matrix[0,2]
                yshift[i] = warp_matrix[1,2]
            except:
                cc[i] = np.nan
                xshift[i]=np.nan
                yshift[i] = np.nan
        # perform regression to predict frameshift based on eye shifts
        # set up models
        xmodel = LinearRegression()
        ymodel = LinearRegression()
        # eye data as predictors
        eyeData = np.zeros([max_frames, 2])
        eyeData[:,0] = dTheta[0:max_frames]
        eyeData[:,1] = dPhi[0:max_frames]
        # shift in x and y as outputs
        xshiftdata = xshift[0:max_frames]
        yshiftdata = yshift[0:max_frames]
        # only use good data
        # not nans, good correlation between frames, small eye movements (no sacccades, only compensatory movements)
        usedata = ~np.isnan(eyeData[:,0]) & ~np.isnan(eyeData[:,1]) & (cc>0.95)  & (np.abs(eyeData[:,0])<2) & (np.abs(eyeData[:,1])<2) & (np.abs(xshiftdata)<5) & (np.abs(yshiftdata)<5)
        # fit xshift
        xmodel.fit(eyeData[usedata,:],xshiftdata[usedata])
        xmap = xmodel.coef_
        xrscore = xmodel.score(eyeData[usedata,:],xshiftdata[usedata])
        # fit yshift
        ymodel.fit(eyeData[usedata,:],yshiftdata[usedata])
        ymap = ymodel.coef_
        yrscore = ymodel.score(eyeData[usedata,:],yshiftdata[usedata])
        # diagnostic plots
        fig = plt.figure(figsize=(8,6))
        plt.subplot(2,2,1)
        plt.plot(dTheta[0:max_frames], xshift[0:max_frames],'.')
        plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12)
        plt.ylim(-12,12); plt.xlabel('dTheta'); plt.ylabel('xshift')
        plt.title('xmap='+str(xmap))
        plt.subplot(2,2,2)
        plt.plot(dTheta[0:max_frames], yshift[0:max_frames],'.')
        plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12)
        plt.ylim(-12,12); plt.xlabel('dTheta'); plt.ylabel('yshift')
        plt.title('ymap='+str(ymap))
        plt.subplot(2,2,3)
        plt.plot(dPhi[0:max_frames], xshift[0:max_frames],'.')
        plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12)
        plt.ylim(-12,12); plt.xlabel('dPhi'); plt.ylabel('xshift')
        plt.subplot(2,2,4)
        plt.plot(dPhi[0:max_frames],yshift[0:max_frames],'.')
        plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-12,12)
        plt.ylim(-12,12); plt.xlabel('dPhi'); plt.ylabel('yshift')
        plt.tight_layout()

        self.xcorrection = xmap
        self.ycorrection = ymap

        self.diagnostic_pdf.savefig(); plt.close()

    def calc_sta(self, lag=2, do_rotation=False, using_spike_sorted=True):
        nks = np.shape(self.small_world_vid[0,:,:])
        all_sta = np.zeros([self.n_cells, np.shape(self.small_world_vid)[1], np.shape(self.small_world_vid)[2]])
        plt.subplots(np.ceil(self.n_cells/7).astype('int'), 7, figsize=(35,np.int(np.ceil(self.n_cells/3))), dpi=50)
        if using_spike_sorted:
            cell_inds = self.cells.index
        elif not using_spike_sorted:
            cell_inds = range(self.n_cells)
        for c, ind in enumerate(cell_inds):
            sp = self.model_nsp[c,:].copy()
            sp = np.roll(sp, -lag)
            sta = self.model_vid.T @ sp
            sta = np.reshape(sta, nks)
            nsp = np.sum(sp)
            plt.subplot(np.ceil(self.n_cells/7).astype('int'), 7, c+1)
            ch = int(self.cells.at[ind,'ch'])
            if self.num_channels == 64 or self.num_channels == 128:
                shank = np.floor(ch/32); site = np.mod(ch,32)
            else:
                shank = 0; site = ch
            plt.title(f'ind={ind!s} nsp={nsp!s}\n ch={ch!s} shank={shank!s}\n site={site!s}',fontsize=5)
            plt.axis('off')
            if nsp > 0:
                sta = sta / nsp
                sta = sta - np.mean(sta)
                if do_rotation:
                    sta = np.fliplr(np.flipud(sta))
                plt.imshow(sta, vmin=-0.3 ,vmax=0.3, cmap='seismic')
            else:
                sta = np.nan
                # plt.imshow(np.zeros([120,160]))
            all_sta[c,:,:] = sta
        plt.tight_layout()
        self.sta = all_sta
        self.detail_pdf.savefig(); plt.close()

    def calc_multilag_sta(self, lag_range=np.arange(-2,8,2)):
        nks = np.shape(self.small_world_vid[0,:,:])
        plt.subplots(self.n_cells, 5, figsize=(6, np.int(np.ceil(self.n_cells/2))), dpi=300)
        for c, ind in enumerate(self.cells.index):
            for lag_ind, lag in enumerate(lag_range):
                sp = self.model_nsp[c,:].copy()
                sp = np.roll(sp, -lag)
                sta = self.model_vid.T @ sp
                sta = np.reshape(sta,nks)
                nsp = np.sum(sp)
                plt.subplot(self.n_cells, 5, (c*5)+lag_ind+1)
                if nsp > 0:
                    sta = sta / nsp
                    sta = sta - np.mean(sta)
                    plt.imshow(sta, vmin=-0.3, vmax=0.3, cmap='seismic')
                else:
                    sta = np.nan
                    # plt.imshow(np.zeros([120,160]))
                if c == 0:
                    plt.title(str(np.round(lag*self.model_dt*1000)) + 'msec', fontsize=5)
                plt.axis('off')
            plt.tight_layout()
        self.detail_pdf.savefig(); plt.close()

    def calc_stv(self):
        nks = np.shape(self.small_world_vid[0,:,:])
        sq_model_vid = self.model_vid**2
        lag = 2
        all_stv = np.zeros((self.n_cells, np.shape(self.small_world_vid)[1], np.shape(self.small_world_vid)[2]))
        plt.subplots(np.ceil(self.n_cells/7).astype('int'), 7, figsize=(35,np.int(np.ceil(self.n_cells/3))), dpi=50)
        mean_sq_img_norm = np.mean(self.small_world_vid**2, axis=0)
        for c, ind in enumerate(self.cells.index):
            sp = self.model_nsp[c,:].copy()
            sp = np.roll(sp, -lag)
            stv = np.nan_to_num(sq_model_vid,0).T @ sp
            stv = np.reshape(stv, nks)
            nsp = np.sum(sp)
            plt.subplot(np.ceil(self.n_cells/7).astype('int'), 7, c+1)
            if nsp > 0:
                stv = stv / nsp
                stv = stv - mean_sq_img_norm
                plt.imshow(stv, vmin=-1, vmax=1, cmap='cividis')
            else:
                stv = np.nan
                # plt.imshow(np.zeros([120,160]))
            all_stv[c,:,:] = stv
            plt.axis('off')
        self.stv = all_stv
        plt.tight_layout()
        self.detail_pdf.savefig(); plt.close()

    def calc_tuning(self, variable, variable_range, useT, label):
        scatter = np.zeros((self.n_cells, len(variable)))
        tuning = np.zeros((self.n_cells, len(variable_range)-1))
        tuning_err = tuning.copy()
        var_cent = np.zeros(len(variable_range)-1)
        for j in range(len(variable_range)-1):
            var_cent[j] = 0.5*(variable_range[j] + variable_range[j+1])
        for i, ind in enumerate(self.cells.index):
            rateInterp = interp1d(self.model_t[0:-1], self.cells.at[ind,'rate'], bounds_error=False)
            scatter[i,:] = rateInterp(useT)
            for j in range(len(variable_range)-1):
                usePts = (variable>=variable_range[j]) & (variable<variable_range[j+1])
                tuning[i,j] = np.nanmean(scatter[i, usePts])
                tuning_err[i,j] = np.nanstd(scatter[i, usePts]) / np.sqrt(np.count_nonzero(usePts))
        fig = plt.subplots(np.ceil(self.n_cells/7).astype('int'), 7, figsize=(35,np.int(np.ceil(self.n_cells/3))), dpi=50)
        for i, ind in enumerate(self.cells.index):
            plt.subplot(np.ceil(self.n_cells/7).astype('int'), 7, i+1)
            plt.errorbar(var_cent, tuning[i,:], yerr=tuning_err[i,:])
            try:
                plt.ylim(0,np.nanmax(tuning[i,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.xlim([variable_range[0], variable_range[-1]]); plt.title(ind, fontsize=5)
            plt.xlabel(label, fontsize=5); plt.ylabel('sp/sec', fontsize=5)
            plt.xticks(fontsize=5); plt.yticks(fontsize=5)
        plt.tight_layout()
        self.detail_pdf.savefig(); plt.close()
        return var_cent, tuning, tuning_err

    def saccade_psth(self, right, left, label):
        rightavg = np.zeros((self.n_cells, self.trange.size-1))
        leftavg = np.zeros((self.n_cells, self.trange.size-1))
        fig = plt.subplots(np.ceil(self.n_cells/7).astype('int'), 7, figsize=(35,np.int(np.ceil(self.n_cells/3))), dpi=50)
        for i, ind in enumerate(self.cells.index):
            for s in np.array(right):
                hist, _ = np.histogram(self.cells.at[ind,'spikeT']-s, self.trange)
                rightavg[i,:] = rightavg[i,:] + hist / (right.size*np.diff(self.trange))
            for s in np.array(left):
                hist, _ = np.histogram(self.cells.at[ind,'spikeT']-s, self.trange)
                leftavg[i,:] = leftavg[i,:]+ hist/(left.size*np.diff(self.trange))
            plt.subplot(np.ceil(self.n_cells/7).astype('int'), 7, i+1)
            plt.plot(self.trange_x, rightavg[i,:], color='tab:blue')
            plt.plot(self.trange_x, leftavg[i,:], color='tab:red')
            maxval = np.max(np.maximum(rightavg[i,:], leftavg[i,:]))
            plt.vlines(0, 0, maxval*1.5, linestyles='dotted', colors='k')
            plt.xlim([-0.5, 0.5])
            plt.ylim([0, maxval*1.2])
            plt.ylabel('sp/sec')
            plt.xlabel('sec')
            plt.title(str(ind)+' '+label)
        plt.tight_layout()
        self.detail_pdf.savefig(); plt.close()
        return rightavg, leftavg

    def fit_glm_rfs(self, nks):
        nT = np.shape(self.model_nsp)[1]
        x = self.glm_model_vid.copy()
        # image dimensions
        nk  = nks[0] * nks[1]
        n_cells = np.shape(self.model_nsp)[0]
        # subtract mean and renormalize -- necessary?
        mn_img = np.mean(x[self.model_use, :], axis=0)
        x = x - mn_img
        img_std = np.std(x[self.model_use, :], axis=0)
        x[:, img_std==0] = 0
        x = np.nan_to_num(x / img_std, 0)
        x = np.append(x, np.ones((nT, 1)), axis=1) # append column of ones
        x = x[self.model_use, :]
        # set up prior matrix (regularizer)
        # L2 prior
        Imat = np.eye(nk)
        Imat = linalg.block_diag(Imat, np.zeros((1,1)))
        # smoothness prior
        consecutive = np.ones((nk, 1))
        consecutive[nks[1]-1::nks[1]] = 0
        diff = np.zeros((1,2))
        diff[0,0] = -1
        diff[0,1] = 1
        Dxx = sparse.diags((consecutive @ diff).T, np.array([0, 1]), (nk-1, nk))
        Dxy = sparse.diags((np.ones((nk,1))@ diff).T, np.array([0, nks[1]]), (nk - nks[1], nk))
        Dx = Dxx.T @ Dxx + Dxy.T @ Dxy
        D  = linalg.block_diag(Dx.toarray(), np.zeros((1,1)))      
        # summed prior matrix
        Cinv = D + Imat
        lag_list = [-4,-2,0,2,4]
        lambdas = 1024 * (2**np.arange(0,16))
        nlam = len(lambdas)
        # set up empty arrays for receptive field and cross correlation
        sta_all = np.zeros((n_cells, len(lag_list), nks[0], nks[1]))
        cc_all = np.zeros((n_cells,len(lag_list)))
        # iterate through units
        for celln in tqdm(range(n_cells)):
            # iterate through timing lags
            for lag_ind, lag in enumerate(lag_list):
                sps = np.roll(self.model_nsp[celln,:], -lag)
                sps = sps[self.model_use]
                nT = len(sps)
                # split training and test data
                test_frac = 0.3
                ntest = int(nT*test_frac)
                x_train = x[ntest:,:] ; sps_train = sps[ntest:]
                x_test = x[:ntest,:]; sps_test = sps[:ntest]
                # calculate a few terms
                sta = x_train.T @ sps_train / np.sum(sps_train)
                XXtr = x_train.T @ x_train
                XYtr = x_train.T @ sps_train
                msetrain = np.zeros((nlam, 1))
                msetest = np.zeros((nlam, 1))
                w_ridge = np.zeros((nk+1, nlam))
                # initial guess
                w = sta
                # loop over regularization strength
                for l in range(len(lambdas)):  
                    # calculate MAP estimate               
                    w = np.linalg.solve(XXtr + lambdas[l]*Cinv, XYtr) # equivalent of \ (left divide) in matlab
                    w_ridge[:,l] = w
                    # calculate test and training rms error
                    msetrain[l] = np.mean((sps_train - x_train@w)**2)
                    msetest[l] = np.mean((sps_test - x_test@w)**2)
                # select best cross-validated lambda for RF
                best_lambda = np.argmin(msetest)
                w = w_ridge[:, best_lambda]
                ridge_rf = w_ridge[:, best_lambda]
                sta_all[celln, lag_ind, :, :] = np.reshape(w[:-1], nks)
                # predicted firing rate
                sp_pred = x_test @ ridge_rf
                # bin the firing rate to get smooth rate vs time
                bin_length = 80
                sp_smooth = (np.convolve(sps_test, np.ones(bin_length), 'same')) / (bin_length*self.model_dt)
                pred_smooth = (np.convolve(sp_pred, np.ones(bin_length), 'same')) / (bin_length*self.model_dt)
                # a few diagnostics
                err = np.mean((sp_smooth-pred_smooth)**2)
                cc = np.corrcoef(sp_smooth, pred_smooth)
                cc_all[celln, lag_ind] = cc[0,1]
        # figure of receptive fields
        fig = plt.figure(figsize=(10,np.int(np.ceil(n_cells/3))),dpi=50)
        for celln in tqdm(range(n_cells)):
            for lag_ind, lag in enumerate(lag_list):
                crange = np.max(np.abs(sta_all[celln,:,:,:]))
                plt.subplot(n_cells, 6, (celln*6)+lag_ind+1)
                plt.imshow(sta_all[celln, lag_ind, :, :], vmin=-crange, vmax=crange, cmap='seismic')
                plt.title('cc={:.2f}'.format(cc_all[celln,lag_ind]), fontsize=5)
        self.detail_pdf.savefig(); plt.close()
        self.glm_rf = sta_all
        self.glm_cc = cc_all

    def diagnostic_video(self):
        raise NotImplementedError

    def diagnostic_audio(self, start=0):
        units = self.cells.index.values
        # timerange
        tr = [start, start+15]
        sp = np.array(self.cells.at[units[self.highlight_neuron],'spikeT']) - tr[0]
        sp = sp[sp>0]
        datarate = 30000
        # compute waveform samples
        tmax = tr[1] - tr[0]
        t = np.linspace(0, tr[1]-tr[0], (tr[1]-tr[0])*self.ephys_samprate, endpoint=False)
        x = np.zeros(np.size(t))
        for spt in sp[sp<tmax]:
            x[np.int64(spt*self.ephys_samprate) : np.int64(spt*self.ephys_samprate +30)] = 1
            x[np.int64(spt*self.ephys_samprate)+31 : np.int64(spt*self.ephys_samprate +60)] = -1
        # write the samples to a file
        self.diagnostic_audio_path = os.path.join(self.recording_path, (self.recording_name+'_unit'+str(self.highlight_neuron)+'.wav'))
        wavio.write(self.diagnostic_audio_path, x, self.ephys_samprate, sampwidth=1)

    def merge_video_with_audio(self):
        merge_mp4_name = os.path.join(self.recording_path, (self.recording_name+'_unit'+str(self.highlight_neuron)+'_merge.mp4'))
        subprocess.call(['ffmpeg', '-i', self.diagnostic_video_path, '-i', self.diagnostic_audio_path, '-c:v', 'copy', '-c:a', 'aac', '-y', merge_mp4_name])

    def open_cells(self, do_sorting=True):
        self.ephys_data = pd.read_json(self.ephys_json_path)
        if do_sorting:
            # sort units by shank and site order
            self.ephys_data = self.ephys_data.sort_values(by='ch', axis=0, ascending=True)
            self.ephys_data = self.ephys_data.reset_index()
            self.ephys_data = self.ephys_data.drop('index', axis=1)
        # spike times
        self.ephys_data['spikeTraw'] = self.ephys_data['spikeT']
        # select good cells from phy2
        self.cells = self.ephys_data.loc[self.ephys_data['group']=='good']
        self.units = self.cells.index.values
        # get number of good units
        self.n_cells = len(self.cells)
        # make a raster plot
        self.spike_raster()

    def open_eyecam(self):
        self.eye_data = xr.open_dataset(self.reye_path)
        self.eye_vid = self.eye_data['REYE_video'].astype(np.uint8)
        self.eyeT = self.eye_data.timestamps.copy()
        self.eyeT = self.eyeT.values
        # plot eye timestamps
        plt.subplots(1,2)
        plt.subplot(1,2,1)
        plt.plot(np.diff(self.eyeT)[0:-1:10])
        plt.xticks(np.linspace(0, (len(self.eyeT)-1)/10, 10))
        plt.xlabel('frame')
        plt.ylabel('eyecam deltaT')
        plt.subplot(1,2,2)
        plt.hist(np.diff(self.eyeT), bins=100)
        plt.xlabel('eyecam deltaT')
        self.diagnostic_pdf.savefig(); plt.close()
        self.eye_params = self.eye_data['REYE_ellipse_params']
        # define theta, phi and zero-center
        th = np.rad2deg(self.eye_params.sel(ellipse_params = 'theta').values)
        phi = np.rad2deg(self.eye_params.sel(ellipse_params = 'phi').values)
        self.theta = th - np.nanmean(th)
        self.phi = phi - np.nanmean(phi)
        # plot of theta vs phi
        self.eye_position()
        # plot theta vs theta switch -- check if deinterlacing was done correctly
        self.check_theta()
        # plot eye variables
        plt.subplots(4,1)
        for count, val in enumerate(self.eye_params.ellipse_params[0:4]):
            plt.subplot(4, 1, count+1)
            plt.plot(self.eyeT[0:-1:10], self.eye_params.sel(ellipse_params=val)[0:-1:10])
            plt.ylabel(val.values)
        plt.tight_layout()
        self.diagnostic_pdf.savefig(); plt.close()

    def summary_fig(self, hist_dt=1):
        hist_t = np.arange(0, np.max(self.worldT), hist_dt)

        plt.subplots(self.n_cells+3, 1,figsize=(12, int(np.ceil(self.n_cells/2))))

        if not self.fm:
            # running speed
            plt.subplot(self.n_cells+3, 1, 1)
            plt.plot(self.ballT, self.ball_speed, 'k')
            plt.xlim(0, np.max(self.worldT)); plt.ylabel('cm/sec'); plt.title('running speed')
        elif self.fm:
            plt.subplot(self.n_cells+3, 1, 1)
            plt.plot(self.topT, self.top_speed, 'k')
            plt.xlim(0, np.max(self.worldT)); plt.ylabel('cm/sec'); plt.title('running speed')
        
        # pupil diameter
        plt.subplot(self.n_cells+3, 1, 2)
        plt.plot(self.eyeT, self.longaxis, 'k')
        plt.xlim(0, np.max(self.worldT)); plt.ylabel('pxls'); plt.title('pupil radius')
        
        # worldcam contrast
        plt.subplot(self.n_cells+3, 1, 3)
        plt.plot(self.worldT, self.contrast)
        plt.xlim(0, np.max(self.worldT)); plt.ylabel('contrast a.u.'); plt.title('contrast')
        
        # raster
        for i, ind in enumerate(self.cells.index):
            rate, bins = np.histogram(self.cells.at[ind,'spikeT'], hist_t)
            plt.subplot(self.n_cells+3, 1, i+4)
            plt.plot(bins[0:-1], rate, 'k')
            plt.xlim(bins[0], bins[-1]); plt.ylabel('unit ' + str(ind))

        plt.tight_layout(); self.detail_pdf.savefig(); plt.close()

    def open_worldcam(self, dwnsmpl=0.5):
        # open data
        world_data = xr.open_dataset(self.world_path)
        world_vid_raw = world_data.WORLD_video.astype(np.uint8).values
        # raw video size
        sz = world_vid_raw.shape
        # resize if size is larger than the target 60x80
        if sz[1]>=160:
            self.world_vid = np.zeros((sz[0], int(sz[1]*dwnsmpl), int(sz[2]*dwnsmpl)), dtype='uint8')
            for f in range(sz[0]):
                self.world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(int(sz[2]*dwnsmpl),int(sz[1]*dwnsmpl)))
        else:
            self.world_vid = world_vid_raw.copy()
        plt.figure()
        plt.imshow(np.mean(self.world_vid, axis=0))
        plt.title('mean world image')
        self.diagnostic_pdf.savefig()
        plt.close()
        # world timestamps
        self.worldT = world_data.timestamps.copy()
        # plot timing
        fig = plt.subplots(1,2,figsize=(15,6))
        plt.subplot(1,2,1)
        plt.plot(np.diff(self.worldT)[0:-1:10])
        plt.xlabel('every 10th frame')
        plt.ylabel('deltaT')
        plt.title('worldcam')
        plt.subplot(1,2,2)
        plt.hist(np.diff(self.worldT), 100)
        plt.xlabel('deltaT')
        self.diagnostic_pdf.savefig(); plt.close()

    def open_topcam(self):
        top_data = xr.open_dataset(self.topcam_path)
        # top_vid = top_data.TOP1_video.astype(np.uint8).values.copy()
        self.topT = top_data.timestamps.values.copy()
        self.top_speed = top_data.TOP1_props.sel(prop='speed').values.copy()
        self.top_head_yaw = np.rad2deg(top_data.TOP1_props.sel(prop='head_yaw').values.copy())
        self.top_body_yaw = np.rad2deg(top_data.TOP1_props.sel(prop='body_yaw').values.copy())
        self.top_body_head_diff = np.rad2deg(top_data.TOP1_props.sel(prop='body_head_diff').values.copy())
        self.top_movement_yaw = np.rad2deg(top_data.TOP1_props.sel(prop='movement_yaw').values.copy())
        self.top_movement_minus_body = top_data.TOP1_props.sel(prop='movement_minus_body').values.copy()
        self.top_forward_run = top_data.TOP1_props.sel(prop='forward_run').values.copy()
        self.top_backward_run = top_data.TOP1_props.sel(prop='backward_run').values.copy()
        self.top_fine_motion = top_data.TOP1_props.sel(prop='fine_motion').values.copy()
        self.top_immobility = top_data.TOP1_props.sel(prop='immobility').values.copy()

    def open_imu(self):
        imu_data = xr.open_dataset(self.imu_path)
        try:
            self.imuT_raw = imu_data.IMU_data.sample # imu timestamps
            imu_channels = imu_data.IMU_data # imu dample data
        except AttributeError:
            self.imuT_raw = imu_data.__xarray_dataarray_variable__.sample
            imu_channels = imu_data.__xarray_dataarray_variable__
        # raw gyro values
        self.gyro_x_raw = imu_channels.sel(channel='gyro_x_raw').values
        self.gyro_y_raw = imu_channels.sel(channel='gyro_y_raw').values
        self.gyro_z_raw = imu_channels.sel(channel='gyro_z_raw').values
        # gyro values in degrees
        self.gyro_x = imu_channels.sel(channel='gyro_x').values
        self.gyro_y = imu_channels.sel(channel='gyro_y').values
        self.gyro_z = imu_channels.sel(channel='gyro_z').values
        # pitch and roll in deg
        self.roll = imu_channels.sel(channel='roll').values
        self.pitch = imu_channels.sel(channel='pitch').values
        # figure of gyro z
        plt.figure()
        plt.plot(self.gyro_x[0:100*60])
        plt.title('gyro z (deg)')
        plt.xlabel('frame')
        self.diagnostic_pdf.savefig(); plt.close()

    def open_running_ball(self):
        running_ball_data = xr.open_dataset(self.running_ball_path).BALL_data
        try:
            self.ball_speed = running_ball_data.sel(move_params='speed_cmpersec')
            self.ballT = running_ball_data.sel(move_params='timestamps')
        except:
            self.ball_speed = running_ball_data.sel(frame='speed_cmpersec')
            self.ballT = running_ball_data.sel(frame='timestamps')
        plt.figure()
        plt.plot(self.ballT,self.ball_speed)
        plt.xlabel('sec'); plt.ylabel('running speed (cm/sec)')
        self.diagnostic_pdf.savefig(); plt.close()

    def drop_slow_data(self, slow_thresh=0.03, win=3):
        isfast = np.diff(self.eyeT) <= slow_thresh
        isslow = sorted(list(set(chain.from_iterable([list(range(int(i)-win,int(i)+(win+1))) for i in np.where(isfast==False)[0]]))))
        self.theta[isslow] = np.nan
        self.phi[isslow] = np.nan

    def set_ephys_offset_and_drift(self):
        if self.fm:
            self.ephys_offset = np.nan
            self.ephys_drift_rate = np.nan
        
        elif not self.fm:
            self.ephys_offset = self.default_ephys_offset
            self.ephys_drift_rate = self.default_ephys_drift_rate

    def align_time(self):
        self.ephysT0 = self.ephys_data.iloc[0,12]
        self.eyeT = self.eyeT - self.ephysT0
        if self.eyeT[0] < -600:
            self.eyeT = self.eyeT + 8*60*60 # 8hr offset for some data
        self.worldT = self.worldT - self.ephysT0
        if self.worldT[0] < -600:
            self.worldT = self.worldT + 8*60*60

        if self.fm:
            self.imuT_raw = self.imuT_raw - self.ephysT0
            self.topT = self.topT - self.ephysT0
        elif not self.fm:
            self.ballT = self.ballT - self.ephysT0

        if self.config['internals']['drop_window_around_missing_data']:
            # drop a window of frames aroud missing timestamps
            self.drop_slow_data()
        
        # calculate eye veloctiy
        self.dEye = np.diff(self.theta)

        self.set_ephys_offset_and_drift()

        if np.isnan(self.ephys_drift_rate) and np.isnan(self.ephys_offset):
            # plot eye velocity against head movements
            plt.figure
            plt.plot(self.eyeT[0:-1], -self.dEye, label='-dEye')
            plt.plot(self.imuT_raw, self.gyro_z, label='gyro z')
            plt.legend()
            plt.xlim(0,10); plt.xlabel('secs'); plt.ylabel('gyro (deg/s)')
            self.diagnostic_pdf.savefig(); plt.close()

            lag_range = np.arange(-0.2, 0.2, 0.002)
            cc = np.zeros(np.shape(lag_range))
            t1 = np.arange(5, len(self.dEye)/60 - 120, 20).astype(int) # was np.arange(5,1600,20), changed for shorter videos
            t2 = t1 + 60
            offset = np.zeros(np.shape(t1))
            ccmax = np.zeros(np.shape(t1))
            imu_interp = interp1d(self.imuT_raw, self.gyro_z)
            for tstart in tqdm(range(len(t1))):
                for l in range(len(lag_range)):
                    try:
                        c, lag = nanxcorr(-self.dEye[t1[tstart]*60 : t2[tstart]*60] , imu_interp(self.eyeT[t1[tstart]*60 : t2[tstart]*60]+lag_range[l]), 1)
                        cc[l] = c[1]
                    except:
                        cc[l] = np.nan
                offset[tstart] = lag_range[np.argmax(cc)]    
                ccmax[tstart] = np.max(cc)
            offset[ccmax<0.2] = np.nan

            # figure
            self.check_imu_eye_alignment(t1, offset, ccmax)
    
            # fit regression to timing drift
            model = LinearRegression()
            dataT = np.array(self.eyeT[t1*60 + 30*60])
            model.fit(dataT[~np.isnan(offset)].reshape(-1,1), offset[~np.isnan(offset)]) 
            self.ephys_offset = model.intercept_
            self.ephys_drift_rate = model.coef_
            self.plot_regression_timing_fit(dataT, offset)

        if self.fm:
            self.imuT = self.imuT_raw - (self.ephys_offset + self.imuT_raw * self.ephys_drift_rate)

        for i in self.ephys_data.index:
            self.ephys_data.at[i,'spikeT'] = np.array(self.ephys_data.at[i,'spikeTraw']) - (self.ephys_offset + np.array(self.ephys_data.at[i,'spikeTraw']) * self.ephys_drift_rate)
        self.cells = self.ephys_data.loc[self.ephys_data['group']=='good']

    def estimate_visual_scene(self):
        # get needed x/y correction
        self.estimate_shift_worldcam()
        theta_interp = interp1d(self.eyeT, self.theta, bounds_error=False)
        phi_interp = interp1d(self.eyeT, self.phi, bounds_error=False)
        # apply to each frame
        for f in tqdm(range(np.shape(self.world_vid)[0])):
            self.world_vid[f,:,:] = imshift(self.world_vid[f,:,:],(-np.int8(theta_interp(self.worldT[f])*self.ycorrection[0] + phi_interp(self.worldT[f])*self.ycorrection[1]),
                                                                   -np.int8(theta_interp(self.worldT[f])*self.xcorrection[0] + phi_interp(self.worldT[f])*self.xcorrection[1])))

    def drop_static_worldcam_pxls(self):
        # get std of worldcam
        std_im = np.std(self.world_vid, axis=0)
        # normalize video
        img_norm = (self.world_vid - np.mean(self.world_vid, axis=0)) / std_im
        # drop static worldcam pixels
        std_im[std_im<20] = 0
        self.img_norm = img_norm * (std_im>0)
        self.std_im = std_im
        # contrast
        self.contrast = np.empty(self.worldT.size)
        for i in range(self.worldT.size):
            self.contrast[i] = np.nanstd(self.img_norm[i,:,:])
        # contrast over time
        plt.figure()
        plt.plot(self.contrast[2000:3000])
        plt.xlabel('frames')
        plt.ylabel('worldcam contrast')
        self.diagnostic_pdf.savefig(); plt.close()
        # std of worldcam image
        fig = plt.figure()
        plt.imshow(std_im)
        plt.colorbar()
        plt.title('worldcam std img')
        self.diagnostic_pdf.savefig(); plt.close()

    def firing_rate_at_new_timebase(self):
        self.model_t = np.arange(0, np.max(self.worldT), self.model_dt)
        self.cells['rate'] = np.nan
        self.cells['rate'] = self.cells['rate'].astype(object)
        for i, ind in enumerate(self.cells.index):
            self.cells.at[ind,'rate'], _ = np.histogram(self.cells.at[ind,'spikeT'], self.model_t)
        self.cells['rate'] = self.cells['rate'] / self.model_dt
        
    def worldcam_at_new_timebase(self, dwnsmpl=0.5):
        """ Create interpolator for movie data so we can evaluate at same timebins are firing rate.
        """
        sz = np.shape(self.img_norm)
        self.small_world_vid = np.zeros((sz[0], np.int(sz[1]*dwnsmpl), np.int(sz[2]*dwnsmpl)))
        for f in range(sz[0]):
            self.small_world_vid[f,:,:] = cv2.resize(self.img_norm[f,:,:],(np.int(sz[2]*dwnsmpl),np.int(sz[1]*dwnsmpl)))
        mov_interp = interp1d(self.worldT, self.small_world_vid, axis=0, bounds_error=False)

        # model video for STAs, STVs, etc.
        nks = np.shape(self.small_world_vid[0,:,:])
        nk = nks[0]*nks[1]
        self.model_vid = np.zeros((len(self.model_t), nk))
        for i in range(len(self.model_t)):
            self.model_vid[i,:] = np.reshape(mov_interp(self.model_t[i]+self.model_dt/2), nk)
        self.model_vid[np.isnan(self.model_vid)] = 0
    
    def topcam_props_at_new_timebase(self):
        self.top_speed_interp = interp1d(self.topT, self.top_speed, bounds_error=False)(self.model_t+self.model_dt/2)
        self.top_forward_run_interp = interp1d(self.topT, self.top_forward_run, bounds_error=False)(self.model_t+self.model_dt/2)
        self.top_fine_motion_interp = interp1d(self.topT, self.top_fine_motion, bounds_error=False)(self.model_t+self.model_dt/2)
        self.top_backward_run_interp = interp1d(self.topT, self.top_backward_run, bounds_error=False)(self.model_t+self.model_dt/2)
        self.top_immobility_interp = interp1d(self.topT, self.top_immobility, bounds_error=False)(self.model_t+self.model_dt/2)
        self.top_head_yaw_interp = interp1d(self.topT, self.top_head_yaw, bounds_error=False)(self.model_t+self.model_dt/2)
        self.top_body_yaw_interp = interp1d(self.topT, self.top_body_yaw, bounds_error=False)(self.model_t+self.model_dt/2)
        self.top_movement_yaw_interp = interp1d(self.topT, self.top_movement_yaw, bounds_error=False)(self.model_t+self.model_dt/2)

    def setup_model_spikes(self):
        # sta/stv setup
        self.model_nsp = np.zeros((self.n_cells, len(self.model_t)))
        # get binned spike rate
        bins = np.append(self.model_t, self.model_t[-1]+self.model_dt)
        for i, ind in enumerate(self.cells.index):
            self.model_nsp[i,:], _ = np.histogram(self.cells.at[ind,'spikeT'], bins)

    def rough_glm_setup(self):
        # get eye position
        self.model_theta = interp1d(self.eyeT, self.theta, bounds_error=False)(self.model_t+self.model_dt/2)
        self.model_phi =interp1d(self.eyeT, self.phi, bounds_error=False)(self.model_t+self.model_dt/2)
        
        # get active times
        if self.fm:
            self.model_raw_gyro_z = interp1d(self.imuT, (self.gyro_z_raw - np.nanmean(self.gyro_z_raw)*7.5), bounds_error=False)(self.model_t)
            self.model_gyro_z = interp1d(self.imuT, self.gyro_z, bounds_error=False)(self.model_t)
            self.model_roll = interp1d(self.imuT, self.roll, bounds_error=False)(self.model_t)
            self.model_pitch = interp1d(self.imuT, self.pitch, bounds_error=False)(self.model_t)
            
            self.model_active = np.convolve(np.abs(self.model_raw_gyro_z), np.ones(np.int(1/self.model_dt)), 'same')
            self.model_use = np.where((np.abs(self.model_theta) < self.model_eye_use_thresh) & (np.abs(self.model_phi) < self.model_eye_use_thresh)& (self.model_active > self.model_active_thresh))[0]
        else:
            self.model_use = np.array([True for i in range(len(self.model_theta))])
        
        # get video ready for glm
        downsamp = 0.25
        testimg = self.img_norm[0,:,:]
        testimg = cv2.resize(testimg, (int(np.shape(testimg)[1]*downsamp), int(np.shape(testimg)[0]*downsamp)))
        testimg = testimg[5:-5, 5:-5]; # remove area affected by eye movement correction
        
        resize_img_norm = np.zeros([np.size(self.img_norm,0), np.int(np.shape(testimg)[0]*np.shape(testimg)[1])])
        for i in tqdm(range(np.size(self.img_norm, 0))):
            smallvid = cv2.resize(self.img_norm[i,:,:],
                                 (np.int(np.shape(self.img_norm)[2]*downsamp), np.int(np.shape(self.img_norm)[1]*downsamp)),
                                 interpolation=cv2.INTER_LINEAR_EXACT)
            smallvid = smallvid[5:-5, 5:-5]
            resize_img_norm[i,:] = np.reshape(smallvid, np.shape(smallvid)[0]*np.shape(smallvid)[1])
        self.glm_model_vid = interp1d(self.worldT, resize_img_norm, 'nearest', axis=0, bounds_error=False)(self.model_t)
        nks = np.shape(smallvid)
        nk = nks[0]*nks[1]
        self.glm_model_vid[np.isnan(self.glm_model_vid)] = 0

    def get_active_times_without_glm(self):
        model_raw_gyro_z = interp1d(self.imuT, (self.gyro_z_raw - np.nanmean(self.gyro_z_raw)*7.5), bounds_error=False)(self.model_t)
        self.model_active = np.convolve(np.abs(model_raw_gyro_z), np.ones(np.int(1/self.model_dt)), 'same')

    def head_and_eye_movements(self):
        plt.figure()
        plt.hist(self.dEye, bins=21, range=(-10,10), density=True)
        plt.xlabel('dTheta')
        self.detail_pdf.savefig(); plt.close()

        if self.fm:
            print('deye dhead')
            self.dHead = np.diff(interp1d(self.imuT, self.gyro_z, bounds_error=False)(self.eyeT))
            self.dGaze = self.dEye + self.dHead

            plt.figure()
            plt.hist(self.dHead, bins=21, range=(-10,10))
            plt.xlabel('dHead')
            self.detail_pdf.savefig(); plt.close()

            plt.figure()
            plt.hist(self.dGaze, bins=21, range=(-10,10))
            plt.xlabel('dGaze')
            self.detail_pdf.savefig(); plt.close()
            
            plt.figure()
            plt.plot(self.dEye, self.dHead, 'k.')
            plt.xlabel('dEye'); plt.ylabel('dHead')
            plt.xlim((-10,10)); plt.ylim((-10,10))
            plt.plot([-10,10], [10,-10], 'r:')
            self.detail_pdf.savefig(); plt.close()

        # all eye movements
        print('all eye movements')
        sthresh = (5 if self.fm else 3)
        left = self.eyeT[(np.append(self.dEye, 0) > sthresh)]
        right = self.eyeT[(np.append(self.dEye, 0) < -sthresh)]
        self.rightsacc_avg, self.leftsacc_avg = self.saccade_psth(right, left, 'all dEye')

        if self.fm:
            # plot gaze shifting eye movements
            print('gaze-shift deye')
            sthresh = 5
            left = self.eyeT[(np.append(self.dEye, 0) > sthresh) & (np.append(self.dGaze,0) > sthresh)]
            right = self.eyeT[(np.append(self.dEye, 0) < -sthresh) & (np.append(self.dGaze, 0) < -sthresh)]
            self.rightsacc_avg_gaze_shift_dEye, self.leftsacc_avg_gaze_shift_dEye = self.saccade_psth(right, left, 'gaze-shift dEye')
            
            print('comp deye')
            # plot compensatory eye movements    
            sthresh = 3
            left = self.eyeT[(np.append(self.dEye, 0) > sthresh) & (np.append(self.dGaze, 0) < 1)]
            right = self.eyeT[(np.append(self.dEye, 0) < -sthresh) & (np.append(self.dGaze, 0) > -1)]
            self.rightsacc_avg_comp_dEye, self.leftsacc_avg_comp_dEye = self.saccade_psth(right, left, 'comp dEye')
            
            print('gaze-shift dhead')
            # plot gaze shifting head movements
            sthresh = 3
            left = self.eyeT[(np.append(self.dHead, 0) > sthresh) & (np.append(self.dGaze, 0) > sthresh)]
            right = self.eyeT[(np.append(self.dHead, 0) < -sthresh) & (np.append(self.dGaze, 0) < -sthresh)]
            self.rightsacc_avg_gaze_shift_dHead, self.leftsacc_avg_gaze_shift_dHead = self.saccade_psth(right, left, 'gaze-shift dHead')
            
            print('comp dhead')
            # plot compensatory head movements
            sthresh = 3
            left = self.eyeT[(np.append(self.dHead,0) > sthresh) & (np.append(self.dGaze, 0) < 1)]
            right = self.eyeT[(np.append(self.dHead,0) < -sthresh) & (np.append(self.dGaze,0) > -1)]
            self.rightsacc_avg_comp_dHead, self.leftsacc_avg_comp_dHead = self.saccade_psth(right, left, 'comp dHead')

    def movement_tuning(self):
        if self.fm:
            # get active times only
            active_interp = interp1d(self.model_t, self.model_active, bounds_error=False)
            active_imu = active_interp(self.imuT.values)
            use = np.where(active_imu > 40)
            imuT_use = self.imuT[use]

            # spike rate vs gyro x
            gx_range = np.linspace(-400,400,10)
            active_gx = self.gyro_x[use]
            self.gyrox_tuning_bins, self.gyrox_tuning, self.gyrox_tuning_err = self.calc_tuning(active_gx, gx_range, imuT_use, 'gyro x')

            # spike rate vs gyro y
            gy_range = np.linspace(-400,400,10)
            active_gy = self.gyro_y[use]
            self.gyroy_tuning_bins, self.gyroy_tuning, self.gyroy_tuning_err = self.calc_tuning(active_gy, gy_range, imuT_use, 'gyro y')
            
            # spike rate vs gyro z
            gz_range = np.linspace(-400,400,10)
            active_gz = self.gyro_z[use]
            self.gyroz_tuning_bins, self.gyroz_tuning, self.gyroz_tuning_err = self.calc_tuning(active_gz, gz_range, imuT_use, 'gyro z')

            # roll vs spike rate
            roll_range = np.linspace(-30,30,10)
            active_roll = self.roll[use]
            self.roll_tuning_bins, self.roll_tuning, self.roll_tuning_err = self.calc_tuning(active_roll, roll_range, imuT_use, 'head roll')

            # pitch vs spike rate
            pitch_range = np.linspace(-30,30,10)
            active_pitch = self.pitch[use]
            self.pitch_tuning_bins, self.pitch_tuning, self.pitch_tuning_err = self.calc_tuning(active_pitch, pitch_range, imuT_use, 'head pitch')

            # subtract mean from roll and pitch to center around zero
            centered_pitch = self.pitch - np.mean(self.pitch)
            centered_roll = self.roll - np.mean(self.roll)

            # interpolate to match eye timing
            pitch_interp = interp1d(self.imuT, centered_pitch, bounds_error=False)(self.eyeT)
            roll_interp = interp1d(self.imuT, centered_roll, bounds_error=False)(self.eyeT)

            # pitch vs theta
            plt.figure()
            plt.plot(pitch_interp[::100], self.theta[::100], 'k.'); plt.xlabel('head pitch'); plt.ylabel('theta')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
            self.diagnostic_pdf.savefig(); plt.close()

            # roll vs phi
            plt.figure()
            plt.plot(roll_interp[::100], self.phi[::100], 'k.'); plt.xlabel('head roll'); plt.ylabel('phi')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')
            self.diagnostic_pdf.savefig(); plt.close()

            # roll vs theta
            plt.figure()
            plt.plot(roll_interp[::100], self.theta[::100], 'k.'); plt.xlabel('head roll'); plt.ylabel('theta')
            plt.ylim([-60,60]); plt.xlim([-60,60])
            self.diagnostic_pdf.savefig(); plt.close()

            # pitch vs phi
            plt.figure()
            plt.plot(pitch_interp[::100], self.phi[::100], 'k.'); plt.xlabel('head pitch'); plt.ylabel('phi')
            plt.ylim([-60,60]); plt.xlim([-60,60])
            self.diagnostic_pdf.savefig(); plt.close()

            # histogram of pitch values
            plt.figure()
            plt.hist(centered_pitch, bins=50); plt.xlabel('head pitch')
            self.diagnostic_pdf.savefig(); plt.close()

            # histogram of pitch values
            plt.figure()
            plt.hist(centered_roll, bins=50); plt.xlabel('head roll')
            self.diagnostic_pdf.savefig(); plt.close()

            # histogram of th values
            plt.figure()
            plt.hist(self.theta, bins=50); plt.xlabel('theta')
            self.diagnostic_pdf.savefig(); plt.close()

            # histogram of pitch values
            plt.figure()
            plt.hist(self.phi, bins=50); plt.xlabel('phi')
            self.diagnostic_pdf.savefig(); plt.close()

        elif not self.fm:
            ball_speed_range = [0, 0.01, 0.1, 0.2, 0.5, 1.0]
            self.ballspeed_tuning_bins, self.ballspeed_tuning, self.ballspeed_tuning_err = self.calc_tuning(self.ball_speed, ball_speed_range, self.ballT, 'running speed')

    def pupil_tuning(self):
        # pupil radius
        self.longaxis = self.eye_params.sel(ellipse_params='longaxis').copy()
        self.norm_longaxis = (self.longaxis - np.mean(self.longaxis)) / np.std(self.longaxis)
        
        # pupil radius over time
        plt.figure()
        plt.plot(self.eyeT, self.norm_longaxis, 'k')
        plt.xlabel('sec')
        plt.ylabel('normalized pupil radius')
        self.detail_pdf.savefig(); plt.close()

        # rate vs pupil radius
        radius_range = np.linspace(10,50,10)
        self.pupilradius_tuning_bins, self.pupilradius_tuning, self.pupilradius_tuning_err = self.calc_tuning(self.longaxis, radius_range, self.eyeT, 'pupil radius')

        # normalize eye position
        self.norm_theta = (self.theta - np.nanmean(self.theta)) / np.nanstd(self.theta)
        self.norm_phi = (self.phi - np.nanmean(self.phi)) / np.nanstd(self.phi)

        plt.figure()
        plt.plot(self.eyeT[:3600], self.norm_theta[:3600], 'k')
        plt.xlabel('sec'); plt.ylabel('norm theta')
        self.diagnostic_pdf.savefig(); plt.close()

        # theta tuning
        theta_range = np.linspace(-30,30,10)
        self.theta_tuning_bins, self.theta_tuning, self.theta_tuning_err = self.calc_tuning(self.theta, theta_range, self.eyeT, 'theta')

        # phi tuning
        phi_range = np.linspace(-30,30,10)
        self.phi_tuning_bins, self.phi_tuning, self.phi_tuning_err = self.calc_tuning(self.phi, phi_range, self.eyeT, 'phi')

    def mua_power_laminar_depth(self):
        # don't run for freely moving, at least for now, because recordings can be too long to fit ephys binary into memory
        # was only a problem for a 128ch recording
        # but hf recordings should be sufficient length to get good estimate
        # read in ephys binary
        lfp_ephys = self.read_binary_file()
        # subtract mean in time dim and apply bandpass filter
        ephys_center_sub = lfp_ephys - np.mean(lfp_ephys,0)
        filt_ephys = self.butter_bandpass(ephys_center_sub, order=6)
        # get lfp power profile for each channel
        ch_num = np.size(filt_ephys,1)
        lfp_power_profiles = np.zeros([ch_num])
        for ch in range(ch_num):
            lfp_power_profiles[ch] = np.sqrt(np.mean(filt_ephys[:,ch]**2)) # multiunit LFP power profile
        # median filter
        lfp_power_profiles_filt = signal.medfilt(lfp_power_profiles)
        if self.probe=='DB_P64-8':
            ch_spacing = 25/2
        else:
            ch_spacing = 25
        if self.num_channels==64:
            norm_profile_sh0 = lfp_power_profiles_filt[:32]/np.max(lfp_power_profiles_filt[:32])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            norm_profile_sh1 = lfp_power_profiles_filt[32:64]/np.max(lfp_power_profiles_filt[32:64])
            layer5_cent_sh1 = np.argmax(norm_profile_sh1)
            lfp_power_profiles = [norm_profile_sh0, norm_profile_sh1]
            lfp_layer5_centers = [layer5_cent_sh0, layer5_cent_sh1]
            plt.subplots(1,2)
            plt.subplot(1,2,1)
            plt.plot(norm_profile_sh0,range(0,32))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            plt.subplot(1,2,2)
            plt.plot(norm_profile_sh1,range(0,32))
            plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
            plt.title('shank1')
            plt.tight_layout(); self.detail_pdf.savefig(); plt.close()
        elif self.num_channels==16:
            norm_profile_sh0 = lfp_power_profiles_filt[:16]/np.max(lfp_power_profiles_filt[:16])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            self.lfp_power_profiles = [norm_profile_sh0]
            self.lfp_layer5_centers = [layer5_cent_sh0]
            plt.figure()
            plt.tight_layout()
            plt.plot(norm_profile_sh0,range(0,16))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([17,-1]); plt.yticks(ticks=list(range(-1,17)),labels=(ch_spacing*np.arange(18)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            self.detail_pdf.savefig(); plt.close()
        elif self.num_channels==128:
            norm_profile_sh0 = lfp_power_profiles_filt[:32]/np.max(lfp_power_profiles_filt[:32])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            norm_profile_sh1 = lfp_power_profiles_filt[32:64]/np.max(lfp_power_profiles_filt[32:64])
            layer5_cent_sh1 = np.argmax(norm_profile_sh1)
            norm_profile_sh2 = lfp_power_profiles_filt[64:96]/np.max(lfp_power_profiles_filt[64:96])
            layer5_cent_sh2 = np.argmax(norm_profile_sh2)
            norm_profile_sh3 = lfp_power_profiles_filt[96:128]/np.max(lfp_power_profiles_filt[96:128])
            layer5_cent_sh3 = np.argmax(norm_profile_sh3)
            self.lfp_power_profiles = [norm_profile_sh0, norm_profile_sh1, norm_profile_sh2, norm_profile_sh3]
            self.lfp_layer5_centers = [layer5_cent_sh0, layer5_cent_sh1, layer5_cent_sh2, layer5_cent_sh3]
            plt.subplots(1,4)
            plt.subplot(1,4,1)
            plt.plot(norm_profile_sh0,range(0,32))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            plt.subplot(1,4,2)
            plt.plot(norm_profile_sh1,range(0,32))
            plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
            plt.title('shank1')
            plt.subplot(1,4,3)
            plt.plot(norm_profile_sh2,range(0,32))
            plt.plot(norm_profile_sh2[layer5_cent_sh2]+0.01,layer5_cent_sh2,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh2*ch_spacing)))
            plt.title('shank2')
            plt.subplot(1,4,4)
            plt.plot(norm_profile_sh3,range(0,32))
            plt.plot(norm_profile_sh3[layer5_cent_sh3]+0.01,layer5_cent_sh3,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh3*ch_spacing)))
            plt.title('shank3')
            plt.tight_layout(); self.detail_pdf.savefig(); plt.close()

    def calculate_gaze(self):
        self.gaze = self.theta + self.top_head_yaw_interp

    def base_ephys_analysis(self):
        print('gathering files')
        self.gather_fm_files()
        print('opening worldcam')
        self.open_worldcam()
        if self.fm:
            print('opening topcam')
            self.open_topcam()
            print('opening imu')
            self.open_imu()
        if not self.fm:
            print('opening running ball')
            self.open_running_ball()
        print('opening ephys')
        self.open_cells()
        print('opening eyecam')
        self.open_eyecam()
        print('aligning timestamps to ephys')
        self.align_time()
        if self.fm and self.stim != 'dk' and self.do_rough_glm_fit:
            print('shifting worldcam for eye movements')
            self.estimate_visual_scene()
        print('dropping static worldcam pixels')
        self.drop_static_worldcam_pxls()
        if self.save_diagnostic_video:
            print('writing diagnostic video')
            self.diagnostic_video()
            self.diagnostic_audio()
            self.merge_video_with_audio()
        if self.fm:
            print('a few more diagnostic figures')
            self.head_and_eye_diagnostics()
        print('firing rates at new timebase')
        self.firing_rate_at_new_timebase()
        print('contrast response functions')
        self.contrast_tuning_bins, self.contrast_tuning, self.contrast_tuning_err = self.calc_tuning(self.contrast, self.contrast_range, self.worldT, 'contrast')
        print('mua power profile laminar depth')
        if not self.fm:
            self.mua_power_laminar_depth()
        print('interpolating worldcam data to match model timebase')
        self.worldcam_at_new_timebase()
        if self.fm:
            print('interpolating topcam data to match model timebase')
            self.topcam_props_at_new_timebase()
        self.setup_model_spikes()
        print('calculating stas')
        self.calc_sta()
        print('calculating multilag stas')
        self.calc_multilag_sta()
        print('calculating stvs')
        self.calc_stv()
        if self.do_rough_glm_fit and ((self.fm and self.stim == 'lt') or self.stim == 'wn'):
            print('using glm to get receptive fields')
            self.rough_glm_setup()
            self.fit_glm_rfs()
        elif self.fm and (self.stim == 'dk' or not self.do_rough_glm_fit):
            print('getting active times without glm')
            self.get_active_times_without_glm()
        if not self.do_rough_glm_fit and self.do_glm_model_preprocessing and ((self.fm and self.stim == 'lt') or self.stim == 'wn'):
            print('preparing inputs for full glm model')
            self.rough_glm_setup()
        print('saccade psths')
        self.head_and_eye_movements()
        print('getting gaze')
        if self.fm and self.stim=='lt':
            self.calculate_gaze()
        print('tuning to pupil properties')
        self.pupil_tuning()
        print('tuning to movement signals')
        self.movement_tuning()
