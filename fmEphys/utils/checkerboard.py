""" Checkerboard stimulus responses.


"""

from tqdm import tqdm

import numpy as np

import cv2
import scipy.interpolate

import fmEphys

def find_flip_times():


def calc_flash_PSTH(rpath):
    """Calculate neural response to flashed stimulus as PSTH.
    """

    worldcam_data_path = fmEphys.find()
    worldcam_data = fmEphys.read_h5(worldcam_data_path)

    vid = worldcam_data['video'].astype(np.uint8)
    stimT = worldcam_data['time']

    spikeT = worldcam_data['spikeT']

    eyeT = eyecam_data['eyeT']
    ephysT0 = worldcam_data['t0']
    worldT = worldcam_data['worldT']

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    num_frames = np.size(vid, 0)
    vid_width = np.size(vid, 1)
    vid_height = np.size(vid, 2)

    kmeans_input = vid.reshape(num_frames, vid_width*vid_height)

    _, labels, _ = cv2.kmeans(kmeans_input.astype(np.float32), 2, None,
                              criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    label_diff = np.diff(np.ndarray.flatten(labels))
 
    stim_state = scipy.interpolate.interp1d(worldT[:-1]-ephysT0, label_diff,
                                            bounds_error=False)(eyeT)
    eventT = eyeT[np.where((stim_state < -0.1)+(stim_state > 0.1))]

    n_cells = np.size(spikeT,0)
    all_psth = np.zeros([n_cells, 2001]) # shape = [unit#, time]
    
    for i, sps in tqdm(spikeT.items()):
        
        if len(sps) < 10: # skip if unit never fired during revchecker
            continue
        
        all_psth[i,:] = utils.ephys.calc_PSTH(sps, eventT)

    return all_psth


self.revchecker_window_start = 0.1 # sec
self.revchecker_window_end = 0.5 # sec

def calc_depth(cfg, rpath):

    world_vid = worldcam_data['video']

    ephysbin_path = utils.path.find()

    probe_name = cfg['probe']
    nCh, spacing = utils.ephys.probe_to_ch(probe_name)

    ephys = utils.ephys.read_ephysbin(ephysbin_path, nCh, probe_name)

    # Mean-center and bandpass filter
    ephys = ephys - np.mean(ephys, 0)
    ephys = utils.filter.butter_filt(ephys, order=6)

    # Use Kmean clustering to group stimulus frames (k=2). This will seperate out
    # the two checkerboard patterns. Using np.diff will give the transition of
    # the stimulus between the two patterns.
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

    # csd
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

    def save(self):
        unit_data = pd.DataFrame([])
        stim = 'Rc'
        for unit_num, ind in enumerate(self.cells.index):
            cols = [stim+'_'+i for i in ['contrast',
                                        'contrast_tuning_bins',
                                        'contrast_tuning',
                                        'contrast_tuning_err',
                                        'spike_triggered_average',
                                        'spike_triggered_variance',
                                        'rightsacc_avg',
                                        'leftsacc_avg',
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
                                        'response_by_channel',
                                        'current_source_density',
                                        'relative_depth',
                                        'layer4cent']]
            unit_df = pd.DataFrame(pd.Series([self.contrast,
                                    self.contrast_tuning_bins,
                                    self.contrast_tuning[unit_num],
                                    self.contrast_tuning_err[unit_num],
                                    self.sta[unit_num],
                                    self.stv[unit_num],
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
                                    self.layer4_centers]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = self.session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
        data_out = pd.concat([self.cells, unit_data], axis=1)
        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

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

        print('making summary and overview figures')
        # self.summary_fig()

        print('closing pdfs')
        self.detail_pdf.close(); self.diagnostic_pdf.close()

        print('saving ephys file')
        self.save()
