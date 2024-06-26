"""
fmEphys/utils/ephys.py




Written by DMM, 2021
"""


import os
import cv2
import json
import wavio
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from itertools import chain
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
import scipy.linalg
import scipy.sparse
import scipy.interpolate
import scipy.ndimage
import sklearn.neighbors
import sklearn.linear_model

import fmEphys as fme


class Ephys(fme.BaseInput):
    """ Analyze the ephys data for a particular stimulus.

    This class is inherited by classes used for specific stimuli,
    and contains the generic functions used for all stimuli.

    Parameters
    ----------
    cfg : dict
        The config dictionary.
    recording_name : str
        The name of the recording.
        e.g.,
    recording_path : str
        The path to the recording directory.
        e.g., /ephys_recordings/[date]/[animal]/[stimulus]/
    """


    def __init__(self, cfg, recording_name, recording_path):
        fme.BaseInput.__init__(self, cfg, recording_name, recording_path)

        # Save figures into a pdf?
        # if false, figures will be shown and not saved
        # turn off if modules are being run outside of the
        # pipeline, where the pdf object might not be defined
        self.figs_in_pdf = True

        # Use the path for the probes.json file in the /utils/ directory.
        self.channel_map_path = os.path.join(os.path.split(__file__)[0], 'probes.json')

        # A particular neuron to highlight in some figures.
        self.highlight_neuron = self.cfg['highlight_cell']
        
        # (No longer used) rough GLM fit code
        self.do_rough_glm_fit = False
        self.do_glm_model_preprocessing = False

        # What model of probe was used for the recording. This should be a string
        # that matches one of the keys in the probes.json file (e.g. 'DB_P64_8')
        self.probe = self.cfg['probe']

        # How many channels are there in the probe?
        self.num_channels = next(int(num) for num in ['128','64','16'] if num in self.probe)
        # Ephys sample rate (in kHz)
        self.ephys_samprate = self.cfg['ephys_samprate']

        # Timebase for GLM binned spike rate
        self.model_dt = 0.025
        self.trange = np.arange(-1, 1.1, self.model_dt)

        # Time bins used for hold PSTHs that used a histogram
        # to bin spikes
        self.trange_x = 0.5*(self.trange[0:-1]+ self.trange[1:])

        # Thresholds to characterize movement/locomotion states
        self.model_eye_use_thresh = 10
        self.model_active_thresh = 40
        self.darkness_thresh = 100

        # Contrast range for spike-triggered average
        self.contrast_range = np.arange(0,1.2,0.1)

        # Thresholds for eye/head movements (in deg/sec)
        # Speed the head must move in a single camera frame to count as any movement type)
        self.shifted_head = 60
        # Gaze must be slower than this to count as a compensatory movement
        self.still_gaze = 120
        # Gaze must move faster than this to count as a gaze shift
        self.shifted_gaze = 240

        # PSTH time bins for PSTH calculated using kernel density estimation.
        # The bins are in msec, start at -1 sec and end at 1 sec after the event
        # onset at 0 sec.
        self.psth_bins = np.linspace(-1000, 1000, 2001)

        # There is a timing offset between
        self.default_ephys_offset = 0.1
        self.default_ephys_drift_rate = -0.000114

        self.session_name = '_'.join(self.recording_name.split('_')[:4])
    

    def gather_base_files(self):
        """ Gather the preprocessed files that exist for all recordings.
        """

        # Right eye camera
        self.reye_path = os.path.join(self.recording_path,
                                      self.recording_name + '_REYE.nc')
        # Because the naming convention for the right eye camera file changed, some of the
        # older recordings will have an .nc file named with lowercase characters.
        if not os.path.isfile(self.reye_path):
            self.reye_path = os.path.join(self.recording_path,
                                          self.recording_name + '_Reye.nc')

        # World camera
        self.world_path = os.path.join(self.recording_path,
                                       self.recording_name + '_world.nc')
        
        # Ephys json, which is written when spike-sorted ephys data is split apart from
        # multiple stimuli.
        self.ephys_json_path = os.path.join(self.recording_path,
                                            self.recording_name + '_ephys_merge.json')
        
        # Ephys binary file, which was never merged with other stimuli since it was not
        # spike sorted
        self.ephys_bin_path = os.path.join(self.recording_path,
                                           self.recording_name + '_Ephys.bin')


    def gather_hf_files(self):
        """ Gather files that are unique to head-fixed recordings.
        """

        # First, gather the files that are common to all recordings
        self.gather_base_files()

        # Treadmill (running ball)
        self.running_ball_path = os.path.join(self.recording_path,
                                              self.recording_name + '_speed.nc')


    def gather_fm_files(self):
        """ Gather files that are unique to freely moving recordings.
        """

        # First, gather the files that are common to all recordings
        self.gather_base_files()

        # Topdown camera
        self.topcam_path = os.path.join(self.recording_path,
                                        self.recording_name + '_TOP1.nc')
        
        # Intertial measurement unit
        self.imu_path = os.path.join(self.recording_path,
                                     self.recording_name + '_imu.nc')


    def read_binary_file(self, do_remap=True):
        """ Read the ephys binary file and remap channels.

        Parameters
        ----------
        do_remap : bool
            If True, remap channels into their physically meaningful
            order. If False, keep the channels in the order they were
            recorded.
        
        Returns
        -------
        ephys : pd.DataFrame
            Ephys data with shape (time, channel).
        
        """

        # Open channel map file
        with open(self.channel_map_path, 'r') as fp:
            all_maps = json.load(fp)
            
        # Get channel map for this probe
        ch_map = all_maps[self.probe]['map']
        num_ch = all_maps[self.probe]['nCh']
        
        # Set up data types to read binary file into
        dtypes = np.dtype([('ch'+str(i),np.uint16) for i in range(0,num_ch)])
        
        # Read in binary file
        ephys = pd.DataFrame(np.fromfile(self.ephys_bin_path, dtypes, -1, ''))
        
        if do_remap:
            # Remap with known order of channels
            ephys = ephys.iloc[:,[i-1 for i in list(ch_map)]]

        return ephys


    def butter_bandpass(self, lfp, lowcut=1, highcut=300, fs=30000, order=5):
        """ Apply bandpass filter to ephys LFP along time dimension.

        Parameters
        ----------
        lfp : np.array
            Ephys LFP with shape (time, channel).
        lowcut : int
            Low end of frequency cut off.
        highcut : int
            High end of frequency cut off.
        fs : int
            Sample rate.
        order : int
            Order of filter.

        Returns
        -------
        filt : np.array
            Filtered data with shape (time, channel).

        """

        # Nyquist frequency
        nyq = 0.5 * fs 

        # Low cutoff
        low = lowcut / nyq

        # High cutoff
        high = highcut / nyq

        # Apply butterworth filter.
        sos = scipy.signal.butter(order,
                                  [low, high],
                                  btype='bandpass', output='sos')
        
        # Apply filter forward and backward to avoid phase shift.
        filt = scipy.signal.sosfiltfilt(sos, lfp, axis=0)

        return filt


    def spike_raster(self):
        """ Plot a spike raster.

        Only plots the raster for the first 10 seconds of the
        recording. Saves the raster into the the detailed PDF.

        """

        fig, ax = plt.subplots()

        for i, ind in enumerate(self.cells.index):
            
            # Array of spike times
            sp = np.array(self.cells.at[ind,'spikeT'])

            # Make vertical line for each time the unit fires
            plt.vlines(sp[sp<10], i-0.25, i+0.25)

            plt.xlim(0, 10) # in sec
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        plt.xlabel('secs')
        plt.ylabel('unit number')
        plt.ylim([self.n_cells, 0])
        plt.tight_layout()

        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def eye_position(self):
        """ Plot eye position.
        """

        good_pts = np.sum(~np.isnan(self.theta))/len(self.theta)

        plt.figure()
        plt.plot(self.theta, self.phi, 'k.', markersize=4)
        plt.xlabel('theta')
        plt.ylabel('phi')
        plt.title('frac good='+str(np.round(good_pts,3)))
        plt.tight_layout()

        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def check_theta(self):
        """ Plot theta to make sure that the frames are in the
        correct order.

        When the video is deinterlaced, frames are interleaved,
        but if video was encoded by Bonsai in an unexpected format,
        the frames could be in the order 1, 0, 3, 2, 5, 4, 7, 6, ...
        If the deinterlacing was done correctly, these plots of theta
        will look smooth, and the panel labeled 'theta switch' will
        have have y values that jitter up and down, since each pair
        of frames was inserted in the wrong order. If theta switch is
        smoother, that means that the deinterlacing went wrong.
        """

        # Flip the order of frames in an every-other fashion
        th_switch = np.zeros(np.shape(self.theta))
        th_switch[0:-1:2] = np.array(self.theta[1::2])
        th_switch[1::2] = np.array(self.theta[0:-1:2])

        # Plot will be of 5 sec starting 35 sec into the video
        start = 35*60
        stop = 40*60

        fig, ax = plt.subplots(121)

        plt.subplot(1,2,1)
        plt.plot(self.theta[start:stop])
        plt.title('theta')
        
        plt.subplot(1,2,2)
        plt.plot(th_switch[start:stop])
        plt.title('theta switch')
        plt.tight_layout()

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def check_imu_eye_alignment(self, t1, offset, ccmax):
        """ Plot the timing offset between the eye camera and IMU.

        Parameters
        ----------
        t1 : int
            Time in seconds to start plot.
        offset : np.array
            Offset between eye camera and IMU.
        ccmax : np.array
            Maximum cross-correlation between eye camera and IMU.

        """

        plt.subplot(1,2,1)
        plt.plot(self.eyeT[t1*60], offset)
        plt.xlabel('secs')
        plt.ylabel('offset (secs)')

        plt.subplot(1,2,2)
        plt.plot(self.eyeT[t1*60], ccmax)
        plt.xlabel('secs')
        plt.ylabel('max cc')
        plt.tight_layout()

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def plot_regression_timing_fit(self, dataT, offset):
        """ Plot the timing offset and drift between eyecam and IMU.

        Parameters
        ----------
        dataT : np.array
            Eye timestamps for set time period.
        offset : np.array
            Offset between eye camera and IMU.

        """

        # Drop any NaNs in the inputs
        dataT = dataT[~np.isnan(dataT)]
        offset = offset[~np.isnan(dataT)]

        plt.figure()

        # Offset between IMU over time
        plt.plot(dataT, offset, 'k.')

        # Timestamps corrected using the offset and drift calculated
        # from the regression
        plt.plot(dataT,
                 self.ephys_offset + dataT * self.ephys_drift_rate,
                 color='r')
        
        plt.xlabel('secs')
        plt.ylabel('offset (secs)')
        plt.title('offset0='+str(np.round(self.ephys_offset,
                    3))+' drift rate='+str(np.round(self.ephys_drift_rate, 3)))
        plt.tight_layout()

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def head_and_eye_diagnostics(self):
        """ Plot head and eye velocity.
        """

        # Plot dEye
        plt.figure()
        plt.plot(self.eyeT[:-1],
                 np.diff(self.theta),
                 label='dTheta')
        
        # Plot dHead
        plt.plot(self.imuT-0.1,
                 (self.gyro_z_raw-3)*10,
                 label='raw gyro z')
        
        plt.xlim(30,40)
        plt.ylim(-12,12)
        plt.legend()
        plt.xlabel('secs')

        plt.tight_layout()

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # Interpolate dHead
        gyro_z_interp = scipy.interpolate.interp1d(self.imuT,
                                                   self.gyro_z,
                                                   bounds_error=False)

        plt.subplots(1,2)

        plt.subplot(1,2,1)
        plt.plot(self.eyeT[0:-1], self.dEye,
                 label='dEye')
        plt.plot(self.eyeT, gyro_z_interp(self.eyeT),
                 label='dHead')
        plt.xlim(37,39)
        plt.ylim(-10,10)
        plt.legend()
        plt.ylabel('deg')
        plt.xlabel('secs')
        
        plt.subplot(1,2,2)
        plt.plot(self.eyeT[0:-1],
                 np.nancumsum(gyro_z_interp(self.eyeT[0:-1])),
                 label='head position')
        
        plt.plot(self.eyeT[0:-1],
                 np.nancumsum(gyro_z_interp(self.eyeT[0:-1])+self.dEye),
                 label='gaze position')
        
        plt.plot(self.eyeT[1:],
                 self.theta[0:-1],
                 label='eye position')
        
        plt.xlim(35,40)
        plt.ylim(-30,30)
        plt.legend()
        plt.ylabel('deg')
        plt.xlabel('secs')

        plt.tight_layout()

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def estimate_shift_worldcam(self, max_frames=3600, num_iter=5000, term_eps=1e-4):
        """ Estimate worldcam shift from eye movements

        Parameters
        ----------
        max_frames : int
            Number of frames to use for estimating shift.
        num_iter : int
            Number of iterations.
        term_eps : float
            Termination epsilon.
        """

        # Get eye displacement for each worldcam frame
        th_interp = scipy.interpolate.interp1d(self.eyeT,
                                               self.theta, bounds_error=False)
        phi_interp = scipy.interpolate.interp1d(self.eyeT,
                                                self.phi, bounds_error=False)
        
        dTheta = np.diff(th_interp(self.worldT))
        dPhi = np.diff(phi_interp(self.worldT))

        # Calculate x/y shift for each worldcam frame  
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    num_iter, term_eps)
        warp_mode = cv2.MOTION_TRANSLATION
        cc = np.zeros(max_frames)
        xshift = np.zeros(max_frames)
        yshift = np.zeros(max_frames)
        warp_all = np.zeros([6, max_frames])
        
        # get shift between adjacent frames
        for i in tqdm(range(max_frames)):
            warp_matrix = np.eye(2, 3,
                                 dtype=np.float32)
            
            try: 
                (cc[i], warp_matrix) = cv2.findTransformECC(self.world_vid[i,:,:],
                                                            self.world_vid[i+1,:,:],
                                                            warp_matrix,
                                                            warp_mode,
                                                            criteria,
                                                            inputMask=None,
                                                            gaussFiltSize=1)
                xshift[i] = warp_matrix[0,2]
                yshift[i] = warp_matrix[1,2]
            
            except:
                cc[i] = np.nan
                xshift[i]=np.nan
                yshift[i] = np.nan
        
        # perform regression to predict frameshift based on eye shifts
        # set up models
        xmodel = sklearn.linear_model.LinearRegression()
        ymodel = sklearn.linear_model.LinearRegression()
        
        # eye data as predictors
        eyeData = np.zeros([max_frames, 2])
        eyeData[:,0] = dTheta[0:max_frames]
        eyeData[:,1] = dPhi[0:max_frames]
        
        # shift in x and y as outputs
        xshiftdata = xshift[0:max_frames]
        yshiftdata = yshift[0:max_frames]
        
        # only use good data
        # not nans, good correlation between frames, small eye movements (no sacccades, only compensatory movements)
        usedata = ~np.isnan(eyeData[:,0]) &                 \
                  ~np.isnan(eyeData[:,1]) &                 \
                  (cc>0.95) &                               \
                  (np.abs(eyeData[:,0])<2) &                \
                  (np.abs(eyeData[:,1])<2) &                \
                  (np.abs(xshiftdata)<5) &                  \
                  (np.abs(yshiftdata)<5)
        
        # fit xshift
        xmodel.fit(eyeData[usedata,:],
                   xshiftdata[usedata])
        xmap = xmodel.coef_
        xrscore = xmodel.score(eyeData[usedata,:],
                               xshiftdata[usedata])
        
        # fit yshift
        ymodel.fit(eyeData[usedata,:],
                   yshiftdata[usedata])
        ymap = ymodel.coef_
        yrscore = ymodel.score(eyeData[usedata,:],
                               yshiftdata[usedata])
        
        # diagnostic plots
        fig = plt.figure(figsize=(8,6))
        
        plt.subplot(2,2,1)
        plt.plot(dTheta[0:max_frames], xshift[0:max_frames], '.')
        plt.plot([-5, 5], [5, -5], 'r')
        plt.xlim(-12,12)
        plt.ylim(-12,12)
        plt.xlabel('dTheta')
        plt.ylabel('xshift')
        plt.title('xmap='+str(xmap))
        
        plt.subplot(2,2,2)
        plt.plot(dTheta[0:max_frames], yshift[0:max_frames], '.')
        plt.plot([-5, 5], [5, -5], 'r')
        plt.xlim(-12,12)
        plt.ylim(-12,12)
        plt.xlabel('dTheta')
        plt.ylabel('yshift')
        plt.title('ymap='+str(ymap))
        
        plt.subplot(2,2,3)
        plt.plot(dPhi[0:max_frames], xshift[0:max_frames], '.')
        plt.plot([-5, 5], [5, -5], 'r')
        plt.xlim(-12,12)
        plt.ylim(-12,12)
        plt.xlabel('dPhi')
        plt.ylabel('xshift')
        
        plt.subplot(2,2,4)
        plt.plot(dPhi[0:max_frames], yshift[0:max_frames], '.')
        plt.plot([-5, 5], [5, -5], 'r')
        plt.xlim(-12,12)
        plt.ylim(-12,12)
        plt.xlabel('dPhi')
        plt.ylabel('yshift')
        plt.tight_layout()

        self.xcorrection = xmap
        self.ycorrection = ymap

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def calc_sta(self, lag=2, do_rotation=False, using_spike_sorted=True):
        
        nks = np.shape(self.small_world_vid[0,:,:])

        all_sta = np.zeros([self.n_cells,
                            np.shape(self.small_world_vid)[1],
                            np.shape(self.small_world_vid)[2]])
        
        plt.subplots(np.ceil(self.n_cells/7).astype('int'), 7,
                     figsize=(35, np.int(np.ceil(self.n_cells/3))),
                     dpi=50)
        
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
                shank = np.floor(ch/32)
                site = np.mod(ch,32)
            else:
                shank = 0
                site = ch
            
            plt.title(f'ind={ind!s} nsp={nsp!s}\n ch={ch!s} shank={shank!s}\n site={site!s}',
                      fontsize=5)
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

        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def calc_multilag_sta(self, lag_range=np.arange(-2,8,2)):

        nks = np.shape(self.small_world_vid[0,:,:])

        plt.subplots(self.n_cells, 5,
                     figsize=(6, np.int(np.ceil(self.n_cells/2))), dpi=300)
        
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

        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def calc_stv(self):

        nks = np.shape(self.small_world_vid[0,:,:])

        sq_model_vid = self.model_vid**2
        lag = 2
        all_stv = np.zeros((self.n_cells,
                            np.shape(self.small_world_vid)[1],
                            np.shape(self.small_world_vid)[2]))
        
        plt.subplots(np.ceil(self.n_cells/7).astype('int'), 7,
                     figsize=(35,np.int(np.ceil(self.n_cells/3))), dpi=50)
        
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

        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def calc_tuning(self, variable, variable_range, useT, label):

        scatter = np.zeros((self.n_cells, len(variable)))
        tuning = np.zeros((self.n_cells, len(variable_range)-1))

        tuning_err = tuning.copy()
        var_cent = np.zeros(len(variable_range)-1)

        for j in range(len(variable_range)-1):

            var_cent[j] = 0.5*(variable_range[j] + variable_range[j+1])

        for i, ind in enumerate(self.cells.index):

            rateInterp = scipy.interpolate.interp1d(self.model_t[0:-1],
                                    self.cells.at[ind,'rate'],
                                    bounds_error=False)
            
            scatter[i,:] = rateInterp(useT)

            for j in range(len(variable_range)-1):

                usePts = (variable>=variable_range[j]) & (variable<variable_range[j+1])
                tuning[i,j] = np.nanmean(scatter[i, usePts])
                tuning_err[i,j] = np.nanstd(scatter[i, usePts]) / np.sqrt(np.count_nonzero(usePts))
        
        fig = plt.subplots(np.ceil(self.n_cells/7).astype('int'), 7,
                           figsize=(35,np.int(np.ceil(self.n_cells/3))), dpi=50)
        
        for i, ind in enumerate(self.cells.index):

            plt.subplot(np.ceil(self.n_cells/7).astype('int'), 7, i+1)
            plt.errorbar(var_cent, tuning[i,:], yerr=tuning_err[i,:])
            
            try:
                plt.ylim(0,np.nanmax(tuning[i,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            
            plt.xlim([variable_range[0], variable_range[-1]])
            plt.title(ind, fontsize=5)
            plt.xlabel(label, fontsize=5)
            plt.ylabel('sp/sec', fontsize=5)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)

        plt.tight_layout()

        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()
        
        return var_cent, tuning, tuning_err


    def saccade_psth(self, right, left, label):

        rightavg = np.zeros((self.n_cells, self.trange.size-1))
        leftavg = np.zeros((self.n_cells, self.trange.size-1))

        fig = plt.subplots(np.ceil(self.n_cells/7).astype('int'), 7,
                           figsize=(35,np.int(np.ceil(self.n_cells/3))), dpi=50)
        
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

        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        return rightavg, leftavg


    def fit_glm_rfs(self):

        downsamp = 0.25

        testimg = self.img_norm[0,:,:]
        testimg = cv2.resize(testimg,
                             (int(np.shape(testimg)[1]*downsamp),
                              int(np.shape(testimg)[0]*downsamp)))

        # remove area affected by eye movement correction
        testimg = testimg[5:-5, 5:-5]
        
        resize_img_norm = np.zeros([
            np.size(self.img_norm,0),
            np.int(np.shape(testimg)[0] * np.shape(testimg)[1])])

        for i in tqdm(range(np.size(self.img_norm, 0))):

            smallvid = cv2.resize(self.img_norm[i,:,:],
                                 (np.int(np.shape(self.img_norm)[2]*downsamp),
                                  np.int(np.shape(self.img_norm)[1]*downsamp)),
                                 interpolation=cv2.INTER_LINEAR_EXACT)

            smallvid = smallvid[5:-5, 5:-5]

            resize_img_norm[i,:] = np.reshape(smallvid,
                                              np.shape(smallvid)[0] * np.shape(smallvid)[1])

        self.glm_model_vid = scipy.interpolate.interp1d(self.worldT, resize_img_norm,
                                    'nearest', axis=0, bounds_error=False)(self.model_t)

        nks = np.shape(smallvid)

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

        # append column of ones
        x = np.append(x, np.ones((nT, 1)), axis=1)
        x = x[self.model_use, :]

        # set up prior matrix (regularizer)
        # L2 prior
        Imat = np.eye(nk)
        Imat = scipy.linalg.block_diag(Imat, np.zeros((1,1)))

        # smoothness prior
        consecutive = np.ones((nk, 1))
        consecutive[nks[1]-1::nks[1]] = 0

        diff = np.zeros((1,2))
        diff[0,0] = -1
        diff[0,1] = 1

        Dxx = scipy.sparse.diags((consecutive @ diff).T,
                                 np.array([0, 1]),
                                 (nk-1, nk))
        
        Dxy = scipy.sparse.diags((np.ones((nk,1))@ diff).T,
                                 np.array([0, nks[1]]),
                                 (nk - nks[1],
                                 nk))
        
        Dx = Dxx.T @ Dxx + Dxy.T @ Dxy

        D  = scipy.linalg.block_diag(Dx.toarray(),
                                     np.zeros((1,1)))      
        
        # Summed prior matrix
        Cinv = D + Imat
        lag_list = [-4,-2,0,2,4]
        lambdas = 1024 * (2**np.arange(0,16))
        nlam = len(lambdas)
        
        # Set up empty arrays for receptive field and cross correlation
        sta_all = np.zeros((n_cells, len(lag_list), nks[0], nks[1]))
        cc_all = np.zeros((n_cells, len(lag_list)))
        
        # Iterate through units
        for celln in tqdm(range(n_cells)):

            # Iterate through timing lags
            for lag_ind, lag in enumerate(lag_list):

                sps = np.roll(self.model_nsp[celln,:], -lag)
                sps = sps[self.model_use]
                nT = len(sps)

                # Split training and test data
                test_frac = 0.3
                ntest = int(nT*test_frac)

                x_train = x[ntest:,:]
                sps_train = sps[ntest:]

                x_test = x[:ntest,:]
                sps_test = sps[:ntest]

                # Calculate a few terms
                sta = x_train.T @ sps_train / np.sum(sps_train)
                XXtr = x_train.T @ x_train
                XYtr = x_train.T @ sps_train
                msetrain = np.zeros((nlam, 1))
                msetest = np.zeros((nlam, 1))
                w_ridge = np.zeros((nk+1, nlam))

                # Initial guess
                w = sta

                # Loop over regularization strength
                for l in range(len(lambdas)):

                    # Calculate MAP estimate
                    # Equivalent of \ (left divide) in matlab
                    w = np.linalg.solve(XXtr + lambdas[l]*Cinv,
                                        XYtr)
                    
                    w_ridge[:,l] = w

                    # Calculate test and training rms error
                    msetrain[l] = np.mean((sps_train - x_train@w)**2)
                    msetest[l] = np.mean((sps_test - x_test@w)**2)

                # Select best cross-validated lambda for RF
                best_lambda = np.argmin(msetest)

                w = w_ridge[:, best_lambda]
                ridge_rf = w_ridge[:, best_lambda]

                sta_all[celln, lag_ind, :, :] = np.reshape(w[:-1], nks)

                # Predicted firing rate
                sp_pred = x_test @ ridge_rf

                # Bin the firing rate to get smooth rate vs time
                bin_length = 80
                sp_smooth = (np.convolve(sps_test, np.ones(bin_length),
                                        'same')) / (bin_length*self.model_dt)
                
                pred_smooth = (np.convolve(sp_pred, np.ones(bin_length),
                                        'same')) / (bin_length*self.model_dt)
                
                # A few diagnostics
                err = np.mean((sp_smooth-pred_smooth)**2)

                cc = np.corrcoef(sp_smooth, pred_smooth)

                cc_all[celln, lag_ind] = cc[0,1]

        # Figure of receptive fields
        fig = plt.figure(figsize=(10, np.int(np.ceil(n_cells/3))), dpi=50)

        for celln in tqdm(range(n_cells)):

            for lag_ind, lag in enumerate(lag_list):

                crange = np.max(np.abs(sta_all[celln,:,:,:]))

                plt.subplot(n_cells, 6, (celln*6)+lag_ind+1)
                plt.imshow(sta_all[celln, lag_ind, :, :],
                           vmin=-crange, vmax=crange, cmap='seismic')
                
                plt.title('cc={:.2f}'.format(cc_all[celln,lag_ind]), fontsize=5)
        
        plt.tight_layout()
        
        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        self.glm_rf = sta_all
        self.glm_cc = cc_all


    def open_cells(self, do_sorting=True):

        self.ephys_data = pd.read_json(self.ephys_json_path)
        
        if do_sorting:
            # Sort units by shank and site order
            self.ephys_data = self.ephys_data.sort_values(by='ch',
                                        axis=0, ascending=True)
            self.ephys_data = self.ephys_data.reset_index()
            self.ephys_data = self.ephys_data.drop('index', axis=1)
        
        # Spike times
        self.ephys_data['spikeTraw'] = self.ephys_data['spikeT'].copy()
        
        # Select good cells from phy2
        self.cells = self.ephys_data.loc[self.ephys_data['group']=='good']
        self.units = self.cells.index.values
        
        # Get number of good units
        self.n_cells = len(self.cells.index)

        # Make a raster plot
        self.spike_raster()

    def open_eyecam(self):

        self.eye_data = xr.open_dataset(self.reye_path)
        self.eye_vid = self.eye_data['REYE_video'].astype(np.uint8)
        self.eyeT = self.eye_data.timestamps.copy().values
        
        # Plot eye timestamps
        plt.subplots(1,2)

        plt.subplot(1,2,1)
        plt.plot(np.diff(self.eyeT)[0:-1:10])
        plt.xticks(np.linspace(0, (len(self.eyeT)-1)/10, 10))
        plt.xlabel('frame')
        plt.ylabel('eyecam deltaT')
        
        plt.subplot(1,2,2)
        plt.hist(np.diff(self.eyeT), bins=100)
        plt.xlabel('eyecam deltaT')
        plt.tight_layout()

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        self.eye_params = self.eye_data['REYE_ellipse_params']
        
        # Define theta, phi and zero-center
        th = np.rad2deg(self.eye_params.sel(ellipse_params = 'theta').values)
        phi = np.rad2deg(self.eye_params.sel(ellipse_params = 'phi').values)

        self.theta = th - np.nanmean(th)
        self.phi = phi - np.nanmean(phi)

        # Flip phi so that up is positive & down is negative
        self.phi = -self.phi

        # Plot of theta vs phi
        self.eye_position()

        # Plot theta vs theta switch -- check if deinterlacing was done correctly
        self.check_theta()

        # Plot eye variables
        plt.subplots(4,1)
        
        for count, val in enumerate(self.eye_params.ellipse_params[0:4]):

            plt.subplot(4, 1, count+1)
            plt.plot(self.eyeT[0:-1:10],
                     self.eye_params.sel(ellipse_params=val)[0:-1:10])
            plt.ylabel(val.values)
        
        plt.tight_layout()

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def summary_fig(self, hist_dt=1):

        hist_t = np.arange(0, np.max(self.worldT), hist_dt)

        plt.subplots(self.n_cells+3, 1,
                     figsize=(12, int(np.ceil(self.n_cells/2))))

        if not self.fm:

            # Running speed
            plt.subplot(self.n_cells+3, 1, 1)
            plt.plot(self.ballT, self.ball_speed, 'k')
            plt.xlim(0, np.max(self.worldT))
            plt.ylabel('cm/sec')
            plt.title('running speed')
        
        elif self.fm:
            plt.subplot(self.n_cells+3, 1, 1)
            plt.plot(self.topT, self.top_speed, 'k')
            plt.xlim(0, np.max(self.worldT))
            plt.ylabel('cm/sec')
            plt.title('running speed')
        
        # Pupil diameter
        plt.subplot(self.n_cells+3, 1, 2)
        plt.plot(self.eyeT, self.longaxis, 'k')
        plt.xlim(0, np.max(self.worldT))
        plt.ylabel('pxls')
        plt.title('pupil radius')
        
        # Worldcam contrast
        plt.subplot(self.n_cells+3, 1, 3)
        plt.plot(self.worldT, self.contrast)
        plt.xlim(0, np.max(self.worldT))
        plt.ylabel('contrast a.u.')
        plt.title('contrast')
        
        # Raster
        for i, ind in enumerate(self.cells.index):

            rate, bins = np.histogram(self.cells.at[ind,'spikeT'], hist_t)
            
            plt.subplot(self.n_cells+3, 1, i+4)
            plt.plot(bins[0:-1], rate, 'k')
            plt.xlim(bins[0], bins[-1])
            plt.ylabel('unit ' + str(ind))

        plt.tight_layout()

        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def open_worldcam(self, dwnsmpl=0.5):

        # Open data
        world_data = xr.open_dataset(self.world_path)
        world_vid_raw = world_data.WORLD_video.astype(np.uint8).values
        
        # Raw video size
        sz = world_vid_raw.shape
        
        # Resize if size is larger than the target 60x80
        if sz[1]>=160:

            self.world_vid = np.zeros((sz[0],
                                       int(sz[1]*dwnsmpl),
                                       int(sz[2]*dwnsmpl)), dtype='uint8')
            
            for f in range(sz[0]):
                self.world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],
                                                   (int(sz[2]*dwnsmpl),
                                                    int(sz[1]*dwnsmpl)))
                
        else:
            self.world_vid = world_vid_raw.copy()

        plt.figure()
        plt.imshow(np.mean(self.world_vid, axis=0))
        plt.title('mean world image')

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # World timestamps
        self.worldT = world_data.timestamps.copy()

        # Plot timing
        fig = plt.subplots(1,2,figsize=(15,6))
        
        plt.subplot(1,2,1)
        plt.plot(np.diff(self.worldT)[0:-1:10])
        plt.xlabel('every 10th frame')
        plt.ylabel('deltaT')
        plt.title('worldcam')
        
        plt.subplot(1,2,2)
        plt.hist(np.diff(self.worldT), 100)
        plt.xlabel('deltaT')
        plt.tight_layout()

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def open_topcam(self):

        top_data = xr.open_dataset(self.topcam_path)

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

        # Raw gyro values
        self.gyro_x_raw = imu_channels.sel(channel='gyro_x_raw').values
        self.gyro_y_raw = imu_channels.sel(channel='gyro_y_raw').values
        self.gyro_z_raw = imu_channels.sel(channel='gyro_z_raw').values
        
        # Gyro values in degrees
        self.gyro_x = imu_channels.sel(channel='gyro_x').values
        self.gyro_y = imu_channels.sel(channel='gyro_y').values
        self.gyro_z = imu_channels.sel(channel='gyro_z').values
        
        # Pitch and roll in deg
        self.roll = imu_channels.sel(channel='roll').values
        self.pitch = imu_channels.sel(channel='pitch').values

        # figure of gyro z
        plt.figure()
        plt.plot(self.gyro_x[0:100*60])
        plt.title('gyro z (deg)')
        plt.xlabel('frame')

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def open_running_ball(self):

        running_ball_data = xr.open_dataset(self.running_ball_path)
        
        try:
            running_ball_data = running_ball_data.BALL_data
        
        except AttributeError:
            running_ball_data = running_ball_data.__xarray_dataarray_variable__
        
        # If the mouse was completely stationary and no samples were recorded...
        if np.isnan(running_ball_data).all():
            self.ballT = np.arange(0, self.worldT[-1], self.cfg['ball_samprate'])
            self.ball_speed = np.zeros(len(self.ballT))
        
        else:
            try:
                self.ball_speed = running_ball_data.sel(move_params='speed_cmpersec')
                self.ballT = running_ball_data.sel(move_params='timestamps')
            except:
                self.ball_speed = running_ball_data.sel(frame='speed_cmpersec')
                self.ballT = running_ball_data.sel(frame='timestamps')
        
        plt.figure()
        plt.plot(self.ballT, self.ball_speed)
        plt.xlabel('sec')
        plt.ylabel('running speed (cm/sec)')

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


    def drop_slow_data(self, slow_thresh=0.03, win=3):

        isfast = np.diff(self.eyeT) <= slow_thresh
        isslow = sorted(list(set(chain.from_iterable([list(range(int(i)-win, int(i)+(win+1))) for i in np.where(isfast==False)[0]]))))
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
            if self.stim=='lt':
                self.topT = self.topT - self.ephysT0
        
        elif not self.fm:
            self.ballT = self.ballT - self.ephysT0
        
        # Calculate eye veloctiy
        self.dEye = np.diff(self.theta) # deg/frame
        self.dEye_dps = self.dEye / np.diff(self.eyeT) # deg/sec

        self.set_ephys_offset_and_drift()

        if np.isnan(self.ephys_drift_rate) and np.isnan(self.ephys_offset):
            
            # Plot eye velocity against head movements
            plt.figure
            plt.plot(self.eyeT[0:-1], -self.dEye, label='-dEye')
            plt.plot(self.imuT_raw, self.gyro_z, label='gyro z')
            plt.legend()
            plt.xlim(0,10)
            plt.xlabel('secs')
            plt.ylabel('gyro (deg/s)')
            
            if self.figs_in_pdf:
                self.diagnostic_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

            lag_range = np.arange(-0.2, 0.2, 0.002)
            cc = np.zeros(np.shape(lag_range))
            
            # Was np.arange(5,1600,20), changed for shorter videos
            t1 = np.arange(5, len(self.dEye)/60 - 120, 20).astype(int)
            t2 = t1 + 60

            offset = np.zeros(np.shape(t1))
            ccmax = np.zeros(np.shape(t1))
            imu_interp = scipy.interpolate.interp1d(self.imuT_raw, self.gyro_z)

            for tstart in tqdm(range(len(t1))):

                for l in range(len(lag_range)):
                    try:
                        c, lag = fme.nanxcorr(-self.dEye[t1[tstart]*60 : t2[tstart]*60],
                                    imu_interp(self.eyeT[t1[tstart]*60 : t2[tstart]*60]+lag_range[l]),
                                    1)
                        cc[l] = c[1]

                    except:
                        cc[l] = np.nan

                offset[tstart] = lag_range[np.argmax(cc)]    
                ccmax[tstart] = np.max(cc)

            offset[ccmax<0.2] = np.nan

            # Figure
            self.check_imu_eye_alignment(t1, offset, ccmax)
    
            # Fit regression to timing drift
            model = sklearn.linear_model.LinearRegression()
            dataT = np.array(self.eyeT[t1*60 + 30*60])

            model.fit(dataT[~np.isnan(offset)].reshape(-1,1),
                      offset[~np.isnan(offset)]) 
            
            self.ephys_offset = model.intercept_
            self.ephys_drift_rate = model.coef_

            self.plot_regression_timing_fit(dataT, offset)

        if self.fm:
            self.imuT = self.imuT_raw - (self.ephys_offset + self.imuT_raw * self.ephys_drift_rate)

        for i in self.ephys_data.index:

            self.ephys_data.at[i,'spikeT'] = np.array(self.ephys_data.at[i,'spikeTraw']) -        \
                        (self.ephys_offset + np.array(self.ephys_data.at[i,'spikeTraw']) * self.ephys_drift_rate)
        
        self.cells = self.ephys_data.loc[self.ephys_data['group']=='good']


    def estimate_visual_scene(self):

        # Get needed x/y correction
        self.estimate_shift_worldcam()

        theta_interp = scipy.interpolate.interp1d(self.eyeT, self.theta,
                                                  bounds_error=False)
        phi_interp = scipy.interpolate.interp1d(self.eyeT, self.phi,
                                                bounds_error=False)
        
        # Apply to each frame
        for f in tqdm(range(np.shape(self.world_vid)[0])):
            
            self.world_vid[f,:,:] = scipy.ndimage.shift(self.world_vid[f,:,:],
                        (-np.int8(theta_interp(self.worldT[f])*self.ycorrection[0] +        \
                                  phi_interp(self.worldT[f])*self.ycorrection[1]),
                         -np.int8(theta_interp(self.worldT[f])*self.xcorrection[0] +        \
                                  phi_interp(self.worldT[f])*self.xcorrection[1])))


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

        # Contrast over time
        plt.figure()
        plt.plot(self.contrast[2000:3000])
        plt.xlabel('frames')
        plt.ylabel('worldcam contrast')

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # Std of worldcam image
        fig = plt.figure()
        plt.imshow(std_im)
        plt.colorbar()
        plt.title('worldcam std img')

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()


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

        self.small_world_vid = np.zeros((sz[0],
                                         np.int(sz[1]*dwnsmpl),
                                         np.int(sz[2]*dwnsmpl)))
        
        for f in range(sz[0]):
            self.small_world_vid[f,:,:] = cv2.resize(self.img_norm[f,:,:],
                                                     (np.int(sz[2]*dwnsmpl),
                                                      np.int(sz[1]*dwnsmpl)))
            
        mov_interp = scipy.interpolate.interp1d(self.worldT, self.small_world_vid,
                                                axis=0, bounds_error=False)

        # Model video for STAs, STVs, etc.
        nks = np.shape(self.small_world_vid[0,:,:])
        nk = nks[0]*nks[1]
        self.model_vid = np.zeros((len(self.model_t), nk))

        for i in range(len(self.model_t)):
            self.model_vid[i,:] = np.reshape(mov_interp(self.model_t[i]+self.model_dt/2), nk)

        self.model_vid[np.isnan(self.model_vid)] = 0


    def topcam_props_at_new_timebase(self):

        self.top_speed_interp = scipy.interpolate.interp1d(self.topT, self.top_speed,
                                                            bounds_error=False)(self.eyeT)
        self.top_forward_run_interp = scipy.interpolate.interp1d(self.topT, self.top_forward_run,
                                                            bounds_error=False)(self.eyeT)
        self.top_fine_motion_interp = scipy.interpolate.interp1d(self.topT, self.top_fine_motion,
                                                            bounds_error=False)(self.eyeT)
        self.top_backward_run_interp = scipy.interpolate.interp1d(self.topT, self.top_backward_run,
                                                            bounds_error=False)(self.eyeT)
        self.top_immobility_interp = scipy.interpolate.interp1d(self.topT, self.top_immobility,
                                                            bounds_error=False)(self.eyeT)
        self.top_head_yaw_interp = scipy.interpolate.interp1d(self.topT, self.top_head_yaw,
                                                            bounds_error=False)(self.eyeT)
        self.top_body_yaw_interp = scipy.interpolate.interp1d(self.topT, self.top_body_yaw,
                                                            bounds_error=False)(self.eyeT)
        self.top_movement_yaw_interp = scipy.interpolate.interp1d(self.topT, self.top_movement_yaw,
                                                            bounds_error=False)(self.eyeT)


    def setup_model_spikes(self):

        # sta/stv setup
        self.model_nsp = np.zeros((self.n_cells, len(self.model_t)))

        # get binned spike rate
        bins = np.append(self.model_t, self.model_t[-1]+self.model_dt)

        for i, ind in enumerate(self.cells.index):
            self.model_nsp[i,:], _ = np.histogram(self.cells.at[ind,'spikeT'], bins)


    def rough_glm_setup(self):

        # Get eye position
        self.model_theta = scipy.interpolate.interp1d(self.eyeT, self.theta,
                                    bounds_error=False)(self.model_t+self.model_dt/2)
        
        self.model_phi = scipy.interpolate.interp1d(self.eyeT, self.phi,
                                    bounds_error=False)(self.model_t+self.model_dt/2)
        
        # Get active times
        if self.fm:
            self.model_raw_gyro_z = scipy.interpolate.interp1d(self.imuT,
                                        (self.gyro_z_raw - np.nanmean(self.gyro_z_raw)*7.5),
                                         bounds_error=False)(self.model_t)
            self.model_gyro_z = scipy.interpolate.interp1d(self.imuT, self.gyro_z,
                                         bounds_error=False)(self.model_t)
            self.model_roll = scipy.interpolate.interp1d(self.imuT, self.roll,
                                         bounds_error=False)(self.model_t)
            self.model_pitch = scipy.interpolate.interp1d(self.imuT, self.pitch,
                                         bounds_error=False)(self.model_t)
            
            self.model_active = np.convolve(np.abs(self.model_raw_gyro_z),
                                            np.ones(np.int(1/self.model_dt)), 'same')
            self.model_use = np.where((np.abs(self.model_theta) < self.model_eye_use_thresh) &     \
                                      (np.abs(self.model_phi) < self.model_eye_use_thresh) &       \
                                      (self.model_active > self.model_active_thresh))[0]
        
        else:
            self.model_use = np.array([True for i in range(len(self.model_theta))])
        
        # Get video ready for glm
        downsamp = 0.25
        testimg = self.img_norm[0,:,:]
        testimg = cv2.resize(testimg,
                             (int(np.shape(testimg)[1]*downsamp),
                              int(np.shape(testimg)[0]*downsamp)))
        
        testimg = testimg[5:-5, 5:-5] # remove area affected by eye movement correction
        
        resize_img_norm = np.zeros([np.size(self.img_norm,0),
                                    np.int(np.shape(testimg)[0] * np.shape(testimg)[1])])
        
        for i in tqdm(range(np.size(self.img_norm, 0))):

            smallvid = cv2.resize(self.img_norm[i,:,:],
                                 (np.int(np.shape(self.img_norm)[2]*downsamp),
                                 np.int(np.shape(self.img_norm)[1]*downsamp)),
                                 interpolation=cv2.INTER_LINEAR_EXACT)
            
            smallvid = smallvid[5:-5, 5:-5]

            resize_img_norm[i,:] = np.reshape(smallvid,
                                              np.shape(smallvid)[0] * np.shape(smallvid)[1])
            
        self.glm_model_vid = scipy.interpolate.interp1d(self.worldT,
                                                        resize_img_norm, 'nearest',
                                                        axis=0, bounds_error=False)(self.model_t)
        
        nks = np.shape(smallvid)

        nk = nks[0]*nks[1]

        self.glm_model_vid[np.isnan(self.glm_model_vid)] = 0


    def get_active_times_without_glm(self):

        model_raw_gyro_z = scipy.interpolate.interp1d(self.imuT,
                                (self.gyro_z_raw - np.nanmean(self.gyro_z_raw)*7.5),
                                bounds_error=False)(self.model_t)
        
        self.model_active = np.convolve(np.abs(model_raw_gyro_z),
                                        np.ones(np.int(1/self.model_dt)), 'same')

    
    def drop_nearby_events(self, thin, avoid, win=0.25):
        """Drop events that fall near others.

        When eliminating compensatory eye/head movements which fall right after
        gaze-shifting eye/head movements, `thin` should be the compensatory event
        times.

        Parameters
        ----------
        thin : np.array
            Array of timestamps (as float in units of seconds) that
            should be thinned out, removing any timestamps that fall
            within `win` seconds of timestamps in `avoid`.
        avoid : np.array
            Timestamps to avoid being near.
        win : np.array
            Time (in seconds) that times in `thin` must fall before or
            after items in `avoid` by.
        
        """

        to_drop = np.array([c for c in thin for g in avoid if ((g>(c-win)) & (g<(c+win)))])
        thinned = np.delete(thin, np.isin(thin, to_drop))

        return thinned


    def drop_repeat_events(self, eventT, onset=True, win=0.020):
        """Eliminate saccades repeated over sequential camera frames.

        Saccades sometimes span sequential camera frames, so that two or
        three sequential camera frames are labaled as saccade events, despite
        only being a single eye/head movement. This function keeps only a
        single frame from the sequence, either the first or last in the
        sequence.

        Parameters
        ----------
        eventT : np.array
            Array of saccade times (in seconds as float).
        onset : bool
            If True, a sequence of back-to-back frames labeled as a saccade will
            be reduced to only the first/onset frame in the sequence. If false, the
            last in the sequence will be used.
        win : float
            Distance in time (in seconds) that frames must follow each other to be
            considered repeating. Frames are 0.016 ms, so the default value, 0.020
            requires that frames directly follow one another.

        Returns
        -------
        thinned : np.array
            Array of saccade times, with repeated labels for single events removed.

        """

        duplicates = set([])

        for t in eventT:

            if onset:
                # keep first
                new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
            
            else:
                # keep last
                new = eventT[((t-eventT)<win) & ((t-eventT)>0)]
            duplicates.update(list(new))

        thinned = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
        
        return thinned


    def calc_kde_PSTH(self, spikeT, eventT, bandwidth=10, resample_size=1,
                      edgedrop=15, win=1000):
        """Calculate PSTH for a single unit.

        The Peri-Stimulus Time Histogram (PSTH) will be calculated using Kernel
        Density Estimation by sliding a gaussian along the spike times centered
        on the event time.

        Because the gaussian filter will create artifacts at the edges (i.e. the
        start and end of the time window), it's best to add extra time to the start
        and end and then drop that time from the PSTH, leaving the final PSTH with no
        artifacts at the start and end. The time (in msec) set with `edgedrop` pads
        the start and end with some time which is dropped from the final PSTH before
        the PSTH is returned.

        Parameters
        ----------
        spikeT : np.array
            Array of spike times in seconds and with the type float. Should be 1D and be
            the spike times for a single ephys unit.
        eventT : np.array
            Array of event times (e.g. presentation of stimulus or the time of a saccade)
            in seconds and with the type float.
        bandwidth : int
            Bandwidth of KDE filter in units of milliseconds.
        resample_size : int
            Size of binning when resampling spike rate, in units of milliseconds.
        edgedrop : int
            Time to pad at the start and end, and then dropped, to eliminate edge artifacts.
        win : int
            Window in time to use in positive and negative directions. For win=1000, the
            PSTH will start -1000 ms before the event and end +1000 ms after the event.

        Returns
        -------
        psth : np.array
            Peri-Stimulus Time Histogram

        """

        # Unit conversions
        bandwidth = bandwidth / 1000
        resample_size = resample_size / 1000
        win = win / 1000
        edgedrop = edgedrop / 1000
        edgedrop_ind = int(edgedrop / resample_size)

        bins = np.arange(-win-edgedrop, win+edgedrop+resample_size, resample_size)

        # Timestamps of spikes (`sps`) relative to `eventT`
        sps = []
        for i, t in enumerate(eventT):
            sp = spikeT-t
            # Only keep spikes in this window
            sp = sp[(sp <= (win+edgedrop)) & (sp >= (-win-edgedrop))] 
            sps.extend(sp)
        sps = np.array(sps)

        if len(sps) < 10:
            n_bins = int((win * 1000 * 2) + 1)
            return np.zeros(n_bins)*np.nan

        kernel = sklearn.neighbors.KernelDensity(kernel='gaussian',
                                                 bandwidth=bandwidth).fit(sps[:, np.newaxis])
        density = kernel.score_samples(bins[:, np.newaxis])

        # Multiply by the # spikes to get spike count per point. Divide
        # by # events for rate/event.
        psth = np.exp(density) * (np.size(sps ) / np.size(eventT))

        # Drop padding at start & end to eliminate edge effects.
        psth = psth[edgedrop_ind:-edgedrop_ind]

        return psth


    def psth_plot(self, right, left, title_str):

        fig, axs = plt.subplots(np.ceil(self.n_cells/7).astype('int'), 7,
                                figsize=(35, np.int(np.ceil(self.n_cells/3))),
                                dpi=50)
        axs = axs.flatten()

        for i, ind in enumerate(self.cells.index.values):

            axs[i].plot(self.psth_bins, right[i,:],
                        color='tab:blue', label='right')
            axs[i].plot(self.psth_bins, left[i,:],
                        color='tab:red', label='left')
            maxval = np.max(np.maximum(right[i,:],
                                       left[i,:]))
            if (not np.isfinite(maxval)) or (maxval == 0):
                maxval = 1
                
            axs[i].vlines(0, 0, maxval*1.5, linestyles='dotted', colors='k')
            axs[i].set_xlim([-500, 500])
            axs[i].set_ylim([0, maxval*1.2])
            axs[i].set_ylabel('sp/sec')
            axs[i].set_xlabel('ms')
            axs[i].set_title(str(ind))
            
        axs[0].legend()
        fig.suptitle(title_str)
        fig.tight_layout()

        self.detail_pdf.savefig()
        plt.close()


    def head_and_eye_movements(self):

        ### Histogram of eye velocities
        plt.figure()
        plt.hist(self.dEye_dps, bins=21, density=True)
        plt.xlabel('dTheta')
        plt.tight_layout()

        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        if self.fm:

            # Define dHead and dGaze using eye timestamps
            print('dEye/dHead')
            self.dHead = scipy.interpolate.interp1d(self.imuT,
                                                    self.gyro_z,
                                                    bounds_error=False)(self.eyeT)[:-1]
            
            # dGaze is the sum of eye and head speeds. When they move in opposite directions,
            # they cancel one another.
            self.dGaze = self.dEye_dps + self.dHead

            plt.figure()
            plt.hist(self.dGaze, bins=21, density=True)
            plt.xlabel('dGaze')

            if self.figs_in_pdf:
                self.detail_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

            plt.figure()
            plt.hist(self.dHead, bins=21, density=True)
            plt.xlabel('dHead')

            if self.figs_in_pdf:
                self.detail_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()
            
            plt.figure()
            plt.plot(self.dEye_dps[::20], self.dHead[::20], 'k.')
            plt.xlabel('dEye')
            plt.ylabel('dHead')
            plt.xlim((-900,900))
            plt.ylim((-900,900))
            plt.plot([-900,900], [900,-900], 'r:')

            if self.figs_in_pdf:
                self.detail_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()
        
        ### Define all head/eye movements using dHead

        # only based on eye movement
        print('All eye movements')

        tmp_eyeT = self.eyeT.flatten()[:-1]
        self.all_eyeL = self.drop_repeat_events(tmp_eyeT[(self.dEye_dps > self.shifted_head)])
        self.all_eyeR = self.drop_repeat_events(tmp_eyeT[(self.dEye_dps < -self.shifted_head)])

        self.rightsacc_avg = np.zeros([len(self.cells.index.values), 2001])*np.nan
        self.leftsacc_avg = np.zeros([len(self.cells.index.values), 2001])*np.nan

        if (len(self.all_eyeL) + len(self.all_eyeR)) >= 10:

            for i, ind in tqdm(enumerate(self.cells.index.values)):
                _spikeT = self.cells.loc[ind,'spikeT'].copy()
                self.leftsacc_avg[i,:] = self.calc_kde_PSTH(_spikeT, self.all_eyeL)
                self.rightsacc_avg[i,:] = self.calc_kde_PSTH(_spikeT, self.all_eyeR)

        self.psth_plot(self.rightsacc_avg, self.leftsacc_avg, 'all movements')

        if self.fm:
            ### Define all head/eye movements using dHead

            print('Head eye movements')
            gazeL = tmp_eyeT[(self.dHead > self.shifted_head) &         \
                             (self.dGaze > self.shifted_gaze)]
            gazeR = tmp_eyeT[(self.dHead < -self.shifted_head) &        \
                             (self.dGaze < -self.shifted_gaze)]

            compL = tmp_eyeT[(self.dHead > self.shifted_head) &         \
                             (self.dGaze < self.still_gaze) &           \
                             (self.dGaze > -self.still_gaze)]
            compR = tmp_eyeT[(self.dHead < -self.shifted_head) &        \
                             (self.dGaze > -self.still_gaze) &          \
                                (self.dGaze < self.still_gaze)]

            compL = self.drop_nearby_events(compL, gazeL)
            compR = self.drop_nearby_events(compR, gazeR)

            self.compL = self.drop_repeat_events(compL)
            self.compR = self.drop_repeat_events(compR)
            self.gazeL = self.drop_repeat_events(gazeL)
            self.gazeR = self.drop_repeat_events(gazeR)

            print('Gaze shift and compensatory PSTHs')

            self.rightsacc_avg_gaze_shift = np.zeros([len(self.cells.index.values),
                                                      2001])*np.nan
            self.leftsacc_avg_gaze_shift = np.zeros([len(self.cells.index.values),
                                                      2001])*np.nan

            self.rightsacc_avg_comp = np.zeros([len(self.cells.index.values),
                                                2001])*np.nan
            self.leftsacc_avg_comp = np.zeros([len(self.cells.index.values),
                                                2001])*np.nan

            for i, ind in tqdm(enumerate(self.cells.index.values)):
                _spikeT = self.cells.loc[ind,'spikeT'].copy()

                self.rightsacc_avg_gaze_shift[i,:] = self.calc_kde_PSTH(_spikeT, self.gazeR)
                self.leftsacc_avg_gaze_shift[i,:] = self.calc_kde_PSTH(_spikeT, self.gazeL)

                self.rightsacc_avg_comp[i,:] = self.calc_kde_PSTH(_spikeT, self.compR)
                self.leftsacc_avg_comp[i,:] = self.calc_kde_PSTH(_spikeT, self.compL)

            self.psth_plot(self.rightsacc_avg_gaze_shift,
                           self.leftsacc_avg_gaze_shift, 'gaze shift')

            self.psth_plot(self.rightsacc_avg_comp,
                           self.leftsacc_avg_comp, 'compensatory')


    def movement_tuning(self):

        if self.fm:

            # Get active times only
            active_interp = scipy.interpolate.interp1d(self.model_t, self.model_active,
                                                       bounds_error=False)
            active_imu = active_interp(self.imuT.values)
            use = np.where(active_imu > 40)
            imuT_use = self.imuT[use]

            # Spike rate vs gyro x
            gx_range = np.linspace(-400,400,10)
            active_gx = self.gyro_x[use]
            self.gyrox_tuning_bins, self.gyrox_tuning, self.gyrox_tuning_err = self.calc_tuning(active_gx,
                                                                             gx_range, imuT_use, 'gyro x')

            # Spike rate vs gyro y
            gy_range = np.linspace(-400,400,10)
            active_gy = self.gyro_y[use]
            self.gyroy_tuning_bins, self.gyroy_tuning, self.gyroy_tuning_err = self.calc_tuning(active_gy,
                                                                             gy_range, imuT_use, 'gyro y')
            
            # Spike rate vs gyro z
            gz_range = np.linspace(-400,400,10)
            active_gz = self.gyro_z[use]
            self.gyroz_tuning_bins, self.gyroz_tuning, self.gyroz_tuning_err = self.calc_tuning(active_gz,
                                                                             gz_range, imuT_use, 'gyro z')

            # Roll vs spike rate
            roll_range = np.linspace(-30,30,10)
            active_roll = self.roll[use]
            self.roll_tuning_bins, self.roll_tuning, self.roll_tuning_err = self.calc_tuning(active_roll,
                                                                       roll_range, imuT_use, 'head roll')

            # Pitch vs spike rate
            pitch_range = np.linspace(-30,30,10)
            active_pitch = self.pitch[use]
            self.pitch_tuning_bins, self.pitch_tuning, self.pitch_tuning_err = self.calc_tuning(active_pitch,
                                                                         pitch_range, imuT_use, 'head pitch')

            # Subtract mean from roll and pitch to center around zero
            centered_pitch = self.pitch - np.mean(self.pitch)
            centered_roll = self.roll - np.mean(self.roll)

            # Interpolate to match eye timing
            pitch_interp = scipy.interpolate.interp1d(self.imuT, centered_pitch,
                                                      bounds_error=False)(self.eyeT)
            roll_interp = scipy.interpolate.interp1d(self.imuT, centered_roll,
                                                      bounds_error=False)(self.eyeT)

            # Pitch vs theta
            plt.figure()
            plt.plot(pitch_interp[::100], self.theta[::100], 'k.')
            plt.xlabel('head pitch')
            plt.ylabel('theta')
            plt.ylim([-60,60])
            plt.xlim([-60,60])
            plt.plot([-60,60],[-60,60], 'r:')

            if self.figs_in_pdf:
                self.diagnostic_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

            # Roll vs phi
            plt.figure()
            plt.plot(roll_interp[::100], self.phi[::100], 'k.')
            plt.xlabel('head roll')
            plt.ylabel('phi')
            plt.ylim([-60,60])
            plt.xlim([-60,60])
            plt.plot([-60,60],[60,-60], 'r:')

            if self.figs_in_pdf:
                self.diagnostic_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

            # Roll vs theta
            plt.figure()
            plt.plot(roll_interp[::100], self.theta[::100], 'k.')
            plt.xlabel('head roll')
            plt.ylabel('theta')
            plt.ylim([-60,60])
            plt.xlim([-60,60])

            if self.figs_in_pdf:
                self.diagnostic_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

            # Pitch vs phi
            plt.figure()
            plt.plot(pitch_interp[::100], self.phi[::100], 'k.')
            plt.xlabel('head pitch')
            plt.ylabel('phi')
            plt.ylim([-60,60])
            plt.xlim([-60,60])

            if self.figs_in_pdf:
                self.diagnostic_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

            # Histogram of pitch values
            plt.figure()
            plt.hist(centered_pitch, bins=50)
            plt.xlabel('head pitch')

            if self.figs_in_pdf:
                self.diagnostic_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

            # Histogram of pitch values
            plt.figure()
            plt.hist(centered_roll, bins=50)
            plt.xlabel('head roll')

            if self.figs_in_pdf:
                self.diagnostic_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

            # Histogram of th values
            plt.figure()
            plt.hist(self.theta, bins=50)
            plt.xlabel('theta')

            if self.figs_in_pdf:
                self.diagnostic_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

            # histogram of pitch values
            plt.figure()
            plt.hist(self.phi, bins=50)
            plt.xlabel('phi')
            if self.figs_in_pdf:
                self.diagnostic_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

        elif not self.fm:
            ball_speed_range = [0, 0.01, 0.1, 0.2, 0.5, 1.0]
            self.ballspeed_tuning_bins, self.ballspeed_tuning, self.ballspeed_tuning_err = self.calc_tuning(self.ball_speed,
                                                                             ball_speed_range, self.ballT, 'running speed')

    def pupil_tuning(self):

        # Pupil radius
        self.longaxis = self.eye_params.sel(ellipse_params='longaxis').copy()
        self.norm_longaxis = (self.longaxis - np.mean(self.longaxis)) / np.std(self.longaxis)
        
        # Pupil radius over time
        plt.figure()
        plt.plot(self.eyeT, self.norm_longaxis, 'k')
        plt.xlabel('sec')
        plt.ylabel('normalized pupil radius')

        if self.figs_in_pdf:
            self.detail_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # Rate vs pupil radius
        radius_range = np.linspace(10,50,10)
        self.pupilradius_tuning_bins, self.pupilradius_tuning, self.pupilradius_tuning_err = self.calc_tuning(self.longaxis,
                                                                                   radius_range, self.eyeT, 'pupil radius')

        # Normalize eye position
        self.norm_theta = (self.theta - np.nanmean(self.theta)) / np.nanstd(self.theta)
        self.norm_phi = (self.phi - np.nanmean(self.phi)) / np.nanstd(self.phi)

        plt.figure()
        plt.plot(self.eyeT[:3600], self.norm_theta[:3600], 'k')
        plt.xlabel('sec')
        plt.ylabel('norm theta')

        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig()
            plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # Theta tuning
        theta_range = np.linspace(-30,30,10)
        self.theta_tuning_bins, self.theta_tuning, self.theta_tuning_err = self.calc_tuning(self.theta,
                                                                       theta_range, self.eyeT, 'theta')

        # Phi tuning
        phi_range = np.linspace(-30,30,10)
        self.phi_tuning_bins, self.phi_tuning, self.phi_tuning_err = self.calc_tuning(self.phi,
                                                                   phi_range, self.eyeT, 'phi')


    def mua_power_laminar_depth(self):

        # Don't run for freely moving, at least for now, because recordings can
        # be too long to fit ephys binary into memory. Was only a problem for a
        # 128ch recording. But hf recordings should be sufficient length to get
        # good estimate

        # Read in ephys binary
        lfp_ephys = self.read_binary_file()
        
        # Subtract mean in time dim and apply bandpass filter
        ephys_center_sub = lfp_ephys - np.mean(lfp_ephys,0)
        filt_ephys = self.butter_bandpass(ephys_center_sub, order=6)

        # Get lfp power profile for each channel
        lfp_power_profiles = np.zeros([self.num_channels])
        
        for ch in range(self.num_channels):

            # Multiunit LFP power profile
            lfp_power_profiles[ch] = np.sqrt(np.mean(filt_ephys[:,ch]**2))
        
        # Median filter
        lfp_power_profiles_filt = scipy.signal.medfilt(lfp_power_profiles)
        
        if self.probe=='DB_P64-8':
            ch_spacing = 25/2
        else:
            ch_spacing = 25

        channel_map_path = os.path.join(os.path.split(__file__)[0], 'probes.json')

        # Open channel map file
        with open(channel_map_path, 'r') as fp:
            all_maps = json.load(fp)

        ch_spacing = all_maps[self.probe]['site_spacing']

        if self.num_channels==64:

            norm_profile_sh0 = lfp_power_profiles_filt[:32] / np.max(lfp_power_profiles_filt[:32])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)

            norm_profile_sh1 = lfp_power_profiles_filt[32:64] / np.max(lfp_power_profiles_filt[32:64])
            layer5_cent_sh1 = np.argmax(norm_profile_sh1)

            self.lfp_power_profiles = [norm_profile_sh0, norm_profile_sh1]
            self.lfp_layer5_centers = [layer5_cent_sh0, layer5_cent_sh1]
            
            plt.subplots(1,2)

            plt.subplot(1,2,1)
            plt.plot(norm_profile_sh0, range(0,32))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01, layer5_cent_sh0,
                     'r*', markersize=12)
            plt.ylim([33,-1])
            plt.yticks(ticks=list(range(-1,33)),
                       labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')

            plt.subplot(1,2,2)
            plt.plot(norm_profile_sh1, range(0,32))
            plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01, layer5_cent_sh1,
                     'r*', markersize=12)
            plt.ylim([33,-1])
            plt.yticks(ticks=list(range(-1,33)),
                       labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
            plt.title('shank1')
            plt.tight_layout()

            if self.figs_in_pdf:
                self.detail_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

        elif self.num_channels==16:

            norm_profile_sh0 = lfp_power_profiles_filt[:16] / np.max(lfp_power_profiles_filt[:16])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            self.lfp_power_profiles = [norm_profile_sh0]
            self.lfp_layer5_centers = [layer5_cent_sh0]

            plt.figure()
            plt.plot(norm_profile_sh0, range(0,16))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01, layer5_cent_sh0,
                     'r*', markersize=12)
            plt.ylim([17,-1])
            plt.yticks(ticks=list(range(-1,17)),
                       labels=(ch_spacing*np.arange(18)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            plt.tight_layout()

            if self.figs_in_pdf:
                self.detail_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()

        elif self.num_channels==128:

            norm_profile_sh0 = lfp_power_profiles_filt[:32] / np.max(lfp_power_profiles_filt[:32])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)

            norm_profile_sh1 = lfp_power_profiles_filt[32:64] / np.max(lfp_power_profiles_filt[32:64])
            layer5_cent_sh1 = np.argmax(norm_profile_sh1)

            norm_profile_sh2 = lfp_power_profiles_filt[64:96] / np.max(lfp_power_profiles_filt[64:96])
            layer5_cent_sh2 = np.argmax(norm_profile_sh2)

            norm_profile_sh3 = lfp_power_profiles_filt[96:128] / np.max(lfp_power_profiles_filt[96:128])
            layer5_cent_sh3 = np.argmax(norm_profile_sh3)

            self.lfp_power_profiles = [norm_profile_sh0, norm_profile_sh1,
                                       norm_profile_sh2, norm_profile_sh3]
            self.lfp_layer5_centers = [layer5_cent_sh0, layer5_cent_sh1,
                                       layer5_cent_sh2, layer5_cent_sh3]
            
            plt.subplots(1,4)

            plt.subplot(1,4,1)
            plt.plot(norm_profile_sh0, range(0,32))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01, layer5_cent_sh0,
                     'r*', markersize=12)
            plt.ylim([33,-1])
            plt.yticks(ticks=list(range(-1,33)),
                       labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')

            plt.subplot(1,4,2)
            plt.plot(norm_profile_sh1, range(0,32))
            plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01, layer5_cent_sh1,
                     'r*', markersize=12)
            plt.ylim([33,-1])
            plt.yticks(ticks=list(range(-1,33)),
                       labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
            plt.title('shank1')

            plt.subplot(1,4,3)
            plt.plot(norm_profile_sh2, range(0,32))
            plt.plot(norm_profile_sh2[layer5_cent_sh2]+0.01, layer5_cent_sh2,
                     'r*', markersize=12)
            plt.ylim([33,-1])
            plt.yticks(ticks=list(range(-1,33)),
                       labels=(ch_spacing*np.arange(34)-(layer5_cent_sh2*ch_spacing)))
            plt.title('shank2')

            plt.subplot(1,4,4)
            plt.plot(norm_profile_sh3, range(0,32))
            plt.plot(norm_profile_sh3[layer5_cent_sh3]+0.01, layer5_cent_sh3,
                     'r*', markersize=12)
            plt.ylim([33,-1])
            plt.yticks(ticks=list(range(-1,33)),
                       labels=(ch_spacing*np.arange(34)-(layer5_cent_sh3*ch_spacing)))
            plt.title('shank3')

            plt.tight_layout()

            if self.figs_in_pdf:
                self.detail_pdf.savefig()
                plt.close()
            elif not self.figs_in_pdf:
                plt.show()


    def base_ephys_analysis(self):

        print('gathering files')
        if not self.fm:
            self.gather_hf_files()
        elif self.fm:
            self.gather_fm_files()
        
        print('opening worldcam')
        self.open_worldcam()

        if self.fm:
            if self.stim == 'lt':
                
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

        if self.fm:
            print('a few more diagnostic figures')
            self.head_and_eye_diagnostics()

        print('firing rates at new timebase')
        self.firing_rate_at_new_timebase()

        print('contrast response functions')
        self.contrast_tuning_bins, self.contrast_tuning, self.contrast_tuning_err = self.calc_tuning(self.contrast,
                                                                      self.contrast_range, self.worldT, 'contrast')
        
        print('mua power profile laminar depth')
        if self.stim == 'wn':
            self.mua_power_laminar_depth()
        
        print('interpolating worldcam data to match model timebase')
        self.worldcam_at_new_timebase()
        
        if self.fm and self.stim=='lt':
            
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
        
        do_glm = (not self.do_rough_glm_fit and self.do_glm_model_preprocessing)
        do_glm = do_glm and ((self.fm and self.stim == 'lt') or self.stim == 'wn')

        if do_glm:
            
            print('preparing inputs for full glm model')
            self.rough_glm_setup()
        
        print('saccade psths')
        self.head_and_eye_movements()
        
        print('tuning to pupil properties')
        self.pupil_tuning()
        
        print('tuning to movement signals')
        self.movement_tuning()

