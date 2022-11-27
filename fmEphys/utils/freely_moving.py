""" Neural responses to stimuli

"""

import os
import cv2
import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys

def freely_moving(cfg):
    """
    
    requires cfg dict entries for
        _rname
        _cam
        _stim
        _stim_abbrv
        _rpath
    
    """

    # all the data to save out in to an .h5 file will be added to this
    # dictionary along the way.
    save_dict = {}

    model_dT = 0.025

    # Write a diagnostic PDF of all figures
    pdf_savepath = os.path.join(cfg['_rpath'],
            '{}_{}_diagnostics.pdf'.format(cfg['rname'], cfg['_stim']))
    pdf = PdfPages(pdf_savepath)

    ### Read in preprocessed data.

    # eye data
    eye_data_path = os.path.join(cfg['_rpath'], '{}_reye_preprocessing.h5'.format(cfg['_rname']))
    eye_data = fmEphys.read_h5(eye_data_path)
    
    # world data
    world_data_path = os.path.join(cfg['_rpath'], '{}_world_preprocessing.h5'.format(cfg['_rname']))
    world_data = fmEphys.read_h5(world_data_path)

    # top data
    top_data_path = os.path.join(cfg['_rpath'], '{}_top_preprocessing.h5'.format(cfg['_rname']))
    top_data = fmEphys.read_h5(top_data_path)

    # imu data
    imu_data_path = os.path.join(cfg['_rpath'], '{}_imu_preprocessing.h5'.format(cfg['_rname']))
    imu_data = fmEphys.read_h5(imu_data_path)

    # ephys data
    ephys_data_path = os.path.join(cfg['_rpath'], '{}_ephys_preprocessing.json'.format(cfg['_rname']))
    ephys_data = pd.read_json(ephys_data_path)

    ### Resize data
    
    # The worldcam video should be read in so that each frame has the size (60,80). Axis=0
    # will be number of video frames.
    worldvid = world_data['video_ds'].copy().astype(np.uint8)
    sz = np.shape()
    # If size is larger than the expected (60,80), resize it now.
    if sz[1]>=160:
        dwnsmpl = 0.5
        raw_wc = worldvid.copy()
        worldvid = np.zeros( (sz[0],
                                           int(sz[1]*dwnsmpl),
                                           int(sz[2]*dwnsmpl)), dtype='uint8')
        for f in range(sz[0]):
            worldvid[f,:,:] = cv2.resize(raw_wc[f,:,:]
                                         (int(sz[2]*dwnsmpl), int(sz[1]*dwnsmpl)))

    ### Clean up the ephys data

    # keep the original names from Kilosort/Phy2)
    ephys_data['Phy2_ind'] = ephys_data.index.copy()

    # Only use cells labeled 'good' in Phy2
    ephys_data = ephys_data.loc[ephys_data['group']=='good']

    # Sort ephys cells by the order of shank and site they were recorded from on the linear
    # ephys probe.
    ephys_data = ephys_data.sort_values(by='ch', axis=0, ascending=True)

    # Reindex the dataframe (cell inds will run 0:N)
    ephys_data = ephys_data.reset_index()
    ephys_data = ephys_data.drop('index', axis=1)

    # Get number of cells
    n_cells = len(ephys_data.index.values)

    ### Eye data

    # Define theta (horizontal) and vertical ()
    theta = np.rad2deg(eye_data['ellipse_parameters']['theta'])
    theta = theta - np.nanmean(theta)
    # Flip phi so that up is positive & down is negative
    phi = np.rad2deg(eye_data['ellipse_parameters']['phi'])
    phi = - (phi - np.nanmean(phi))

    ### Timestamps
    eyeT = eye_data['time']
    worldT = world_data['time']
    imuT = imu_data['time']
    topT = top_data['time']

    ### Some diagnostic plots of the data that has been read in so far

    fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)

    ax0.imshow(np.mean(worldvid, axis=0), cmap='gray')
    ax0.set_title('mean worldcam image')

    # 3600 samples will be the first 1 min of the recording
    ax1.plot(np.diff(worldT)[:3600], color='k')
    ax1.set_xticks(np.linspace(0,60,6).astype(int))
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('worldcam deltaT (sec)')

    ax2.hist(np.diff(worldT), 100, density=True, color='k')
    ax2.set_xlabel('worldcam deltaT (sec)')

    ax3.plot(imu_data['gyro_z'][:3600], color='k')
    ax3.set_xticks(np.linspace(0,60,6).astype(int))
    ax3.set_xlabel('time (sec)')
    ax3.set_ylabel('gyro z (deg)')

    ax4.plot(np.diff(eyeT)[:3600], color='k')
    ax4.set_xticks(np.linspace(0,60,6).astype(int))
    ax4.set_xlabel('time (sec)')
    ax4.set_ylabel('eyecam deltaT (sec)')
   
    ax5.hist(np.diff(eyeT), bins=100, density=True, color='k')
    ax5.set_xlabel('worldcam deltaT (sec)')

    fig.tight_layout()
    pdf.savefig(); plt.close()

    ### More diagnostic figures

    fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)

    ax0.plot(theta[::1000], phi[::1000], 'k.', markersize=2)
    ax0.set_xlabel('theta (deg)')
    ax0.set_xlabel('phi (deg)')

    ax1.plot(eye_data['ellipse_parameters']['long_axis'][:3600])
    ax1.set_xticks(np.linspace(0,60,6).astype(int))
    ax1.set_ylabel('pupil radius (pixels)')
    ax1.set_xlabel('time (sec)')

    ax2.plot(theta[:3600], label='theta')
    ax2.plot(theta[:3600], label='phi')
    ax2.set_xticks(np.linspace(0,60,6).astype(int))
    ax2.set_ylabel('deg')
    ax2.legend()
    ax2.set_xlabel('time (sec)')

    ax3.plot(top_data['movement']['speed'][:7200], 'k-')
    ax3.set_xticks(np.linspace(0,120,6).astype(int))
    ax3.set_xlabel('sec')
    ax3.set_ylabel('running speed (cm/sec)')

    for _ax in [ax3,ax4,ax5]:
        _ax.axis('off')

    fig.tight_layout()
    pdf.savefig(); plt.close()

    # Spike raster

    ### Align the timing of the data inputs

    ephysT0 = ephys_data.iloc[0,12]
    _8hr = 8*60*60 # some data are offset by 8 hours

    eyeT = eyeT - ephysT0
    # Some data have an 8 hour offset
    if eyeT[0] < -600:
        eyeT = eyeT + _8hr

    worldT = worldT - ephysT0
    if worldT[0] < -600:
        worldT = worldT + _8hr

    imuT = imuT - ephysT0

    if (cfg['_stim'] != 'FmDk'):
        # Recordings in the dark do not have topdown camera data
        topT = topT - ephysT0

    # There is an offset between ephys+IMU and other data inputs. There is also a small
    # drift over the recording. Ephys is supposed to be sampled at 30 kHz but it is
    # actually sampled at 29.999...etc. kHz, so we regress this drift out and fix the
    # spike times.
    # This is only done for a freely moving recoridng. For head-fixed recordings, we
    # use a default value (there's no IMU for head-fixed recordings, so this wouldn't
    # be possible to calculate, but it also isn't variable between recordings so it
    # would probably be okay to use the default value for all head-fixed and freely
    # moving recordings).
    gyro_z = imu_data['gyro_z']
    offset, drift = fmEphys.calc_offset_drift(eyeT, dEye_dpf, imuT, gyro_z, pdf)

    # Now apply those values to the IMU timestamps
    imuT = imuT - (offset + imuT * drift)

    # Also need to correct spike times in ephys data
    ephys_data['spikeTraw'] = ephys_data['spikeT'].copy()

    for ind, row in ephys_data.iterrows():
        ephys_data.at[ind,'spikeT'] = np.array(row['spikeTraw']) - (offset + np.array(row['spikeTraw']) * drift)

    ### Calculate eye veloctiy
    dEye_dpf = np.diff(theta) # deg/frame
    dEye = dEye_dpf / np.diff(eyeT) # deg/sec

    ### Drop static worldcamera pixels
    # This lets us define the edges of the monitor, since areas off of the monitor will not
    # be modulated by changes in stimulus. This shouldn't really matter for freely moving
    # recordings, but there is no reason not to check for dead pixels.

    # Get std of worldcam
    std_im = np.std(worldvid, axis=0)
    # normalize video
    img_norm = (worldvid - np.mean(worldvid, axis=0)) / std_im

    # Drop static worldcam pixels
    std_im[std_im < 20] = 0
    img_norm = img_norm * (std_im > 0)
    
    # calculate contrast
    contrast = np.empty(worldT.size)
    for i in range(worldT.size):
        contrast[i] = np.nanstd(img_norm[i,:,:])

    fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)
    
    # contrast over time
    ax0.plot(contrast[:3600])
    ax0.set_xticks(np.linspace(0,3600,5), labels=np.linspace(0,1,5).astype(int))
    ax0.set_xlabel('time (sec)')
    ax0.set_ylabel('worldcam contrast')

    ax1.imshow(std_im)
    ax1.colorbar()
    ax1.set_title('worldcam std img')

    # diagnostic for head and eye movements
    ax2.plot(eyeT[:-1], dEye, label='dEye')
    ax2.plot(imuT, gyro_z, label='dHead')
    ax2.set_xlim(0,10)
    ax2.set_ylim(-30,30)
    ax2.set_xlabel('time (sec)')
    ax2.legend()

    gyro_z_interp = scipy.interpolate.interp1d(imuT, gyro_z, bounds_error=False)

    ax3.plot(eyeT[0:-1], dEye, label='dEye')
    ax3.plot(eyeT, gyro_z_interp(eyeT), label='dHead')
    ax3.set_xlim(37,39)
    ax3.set_ylim(-10,10)
    ax3.set_ylabel('deg')
    ax3.set_xlabel('secs')
    ax3.set_xticks(np.linspace(37,39,8), labels=np.linspace(0,2,8).astype(int))
    ax3.legend()

    ax4.plot(eyeT[:-1], np.nancumsum(gyro_z_interp(eyeT[:-1])), label='head')
    ax4.plot(eyeT[:-1], np.nancumsum(gyro_z_interp(eyeT[:-1]) + dEye), label='gaze')
    ax4.plot(eyeT[:-1], theta[:-1], label='eye')
    ax4.set_xlim(35,40)
    ax4.set_ylim(-30,30)
    ax4.set_ylabel('position (deg)')
    ax4.set_xlabel('time (sec)')
    ax4.legend()
    ax4.set_xticks(np.linspace(35,40,8), labels=np.linspace(0,5,8).astype(int))

    for _ax in [ax5]:
        _ax.axis('off')

    fig.tight_layout()
    pdf.savefig(); plt.close()

    ### Adjust the timebase

    # Recalculate the firing
    modelT = np.arange(0, np.max(worldT), model_dT)

    ephys_data['rate'] = fmEphys.empty_obj_col(n_cells, len(modelT))

    _spike_ser = ephys_data['spikeT'].copy()

    for ind, _spT in _spike_ser.iteritems():
        ephys_data.at[ind,'rate'], _ = np.histogram(_spT, modelT)
    
    ephys_data['rate'] = ephys_data['rate'] / model_dT


    ### Calculate tuning curves

    spike_rates = ephys_data['rate'].copy()

    # Calculate contrast response functions
    contrast_range = np.arange(0,1.2,0.1)

    _bins, _tuning, _err = fmEphys.calc_all_tuning(spike_rates, contrast, contrast_range, worldT)
    save_dict['contrast_tuning_bins'] = _bins
    save_dict['contrast_tuning_curve'] = _tuning
    save_dict['contrast_tuning_error'] = _err

    # Add a plot to the pdf
    fmEphys.plot_all_tuning(_bins, _tuning, _err, pdf, 'contrast')


    ### Some set up to prepare stimulus (from worldcam) to spike rate
    # Used for STA, STV, etc.

    # Create interpolator for movie data so we can evaluate at same timebins are firing rate.
    _sz = np.shape(img_norm)
    small_world_vid = np.zeros((_sz[0],
                                np.int(_sz[1]*dwnsmpl),
                                np.int(_sz[2]*dwnsmpl)))
    
    for f in range(_sz[0]):
        small_world_vid[f,:,:] = cv2.resize(img_norm[f,:,:],
                                            (np.int(_sz[2]*dwnsmpl),
                                            np.int(_sz[1]*dwnsmpl)))
    
    mov_interp = scipy.interpolate.interp1d(worldT, small_world_vid, axis=0, bounds_error=False)

    # Set up model video
    nks = np.shape(small_world_vid[0,:,:])
    nk = nks[0]*nks[1]
    model_vid = np.zeros((len(modelT), nk))
    for i in range(len(modelT)):
        model_vid[i,:] = np.reshape(mov_interp(modelT[i]+model_dT/2), nk)

    model_vid[np.isnan(model_vid)] = 0

    # spike rate setup
    model_nsp = np.zeros((n_cells, len(modelT)))
    # get binned spike rate
    bins = np.append(modelT, modelT[-1] + modelT)
    for i, ind in enumerate(spike_rates.index.values):
        model_nsp[i,:], _ = np.histogram(spike_rates.loc[ind, 'spikeT'], bins)

    # Calculate spike-triggered variance for all cells
    all_sta = fmEphys.calc_all_STA(model_nsp, model_vid)
    fmEphys.plot_all_STA(all_sta)

    # Calculate spike-triggered variance for all cells
    all_stv = fmEphys.calc_all_STV(model_nsp, model_vid)
    fmEphys.plot_all_STV(all_stv)


    ### Interpolate properties from the topcam to align them with the eyecamera
    top_data['aligned_movement'] = {}
    for _prop, _vals in top_data['movement'].items():
        _interp_vals = scipy.interpolate.interp1d(topT, _vals,
                                                  bounds_error=False)(eyeT)
        save_dict[_prop] = _interp_vals


    ### Calculate saccade times

    # Line dHead (speed of horizontal head movement) up with eye timestamps
    dHead = scipy.interpolate.interp1d(imuT, gyro_z, bounds_error=False)(eyeT)[:-1]

    # Gaze speed
    # dGaze = dEye + dHead
    dGaze = dEye + dHead

    # Get timestamp of gaze-shifting and compensatory eye movements
    # This is going to return a dictionary with keys compL, compR, gazeL, and
    # gazeR. Each key will be paired with a dictionary which is an array of
    # timestamps as floats in units of seconds.
    saccade_events = fmEphys.calc_saccade_times(dHead, dGaze, eyeT)

    spike_times = ephys_data['spikeT'].copy()

    all_eyehead_psth = {}

    # Calculate a PSTH for all cells and movementt types
    for _saccade_type, _saccade_times in saccade_events.items():

        all_eyehead_psth[_saccade_type] = fmEphys.calc_all_PSTH(spike_times, eventT)


    # Get active periods (based on gyro)
    _ag = imu_data['gyro_z_raw'].copy()
    model_active_gyro = scipy.interpolate.interp1d(imuT,
                                                   (_ag - np.nanmean(_ag) * 7.5),
                                                   bounds_error=False)(modelT)
    model_active = np.convolve(np.abs(model_active_gyro),
                               np.ones(np.int(1 / model_dT)), 'same')

    active_imu = scipy.interpolate.interp1d(modelT, model_active, bounds_error=False)(imuT)
    use_as_active = np.where(active_imu > 40)
    active_imuT = imuT[use_as_active]

    ### Movement tuning curves

    gyro_range = np.linspace(-400,400,10)

    # gyro x
    active_gx = imu_data['gyro_x'][use_as_active]
    _bins, _tuning, _err = fmEphys.calc_all_tuning(spike_rates, active_gx, gyro_range, active_imuT)
    save_dict['gyroX_tuning_bins'] = _bins
    save_dict['gyroX_tuning_curve'] = _tuning
    save_dict['gyroX_tuning_error'] = _err

    fmEphys.plot_all_tuning(_bins, _tuning, _err, pdf, 'gyroX')

    # gyro y
    active_gy = imu_data['gyro_y'][use_as_active]
    _bins, _tuning, _err = fmEphys.calc_all_tuning(spike_rates, active_gy, gyro_range, active_imuT)
    save_dict['gyroY_tuning_bins'] = _bins
    save_dict['gyroY_tuning_curve'] = _tuning
    save_dict['gyroY_tuning_error'] = _err

    fmEphys.plot_all_tuning(_bins, _tuning, _err, pdf, 'gyroY')

    # gyro z
    active_gz = imu_data['gyro_z'][use_as_active]
    _bins, _tuning, _err = fmEphys.calc_all_tuning(spike_rates, active_gz, gyro_range, active_imuT)
    save_dict['gyroZ_tuning_bins'] = _bins
    save_dict['gyroZ_tuning_curve'] = _tuning
    save_dict['gyroZ_tuning_error'] = _err

    fmEphys.plot_all_tuning(_bins, _tuning, _err, pdf, 'gyroZ')


    # some setup for roll/pitch tuning
    rollpitch_range = np.linspace(-30,30,10)

    # head roll
    active_roll = imu_data['roll'][use_as_active]
    _bins, _tuning, _err = fmEphys.calc_all_tuning(spike_rates, active_roll, rollpitch_range, active_imuT)
    save_dict['roll_tuning_bins'] = _bins
    save_dict['roll_tuning_curve'] = _tuning
    save_dict['roll_tuning_error'] = _err

    fmEphys.plot_all_tuning(_bins, _tuning, _err, pdf, 'roll')

    # head pitch
    active_pitch = imu_data['pitch'][use_as_active]
    _bins, _tuning, _err = fmEphys.calc_all_tuning(spike_rates, active_pitch, rollpitch_range, active_imuT)
    save_dict['pitch_tuning_bins'] = _bins
    save_dict['pitch_tuning_curve'] = _tuning
    save_dict['pitch_tuning_error'] = _err

    fmEphys.plot_all_tuning(_bins, _tuning, _err, pdf, 'pitch')

    # Subtract mean from roll and pitch to center the values around zero
    pitch = imu_data['pitch'].copy()
    roll = imu_data['roll'].copy()
    centered_pitch = pitch - np.mean(pitch)
    centered_roll = roll - np.mean(roll)
    # Interpolate to match eye timing
    pitch_interp = scipy.interpolate.interp1d(imuT, centered_pitch,
                                              bounds_error=False)(eyeT)
    roll_interp = scipy.interpolate.interp1d(imuT, centered_roll,
                                             bounds_error=False)(eyeT)


    ### Some diagnostic figures relating to head/eye movement

    fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)

    ax0.hist(dEye, bins=21, density=True)
    ax0.set_xlabel('dEye (deg/sec)')

    ax1.hist(dGaze, bins=21, density=True)
    ax1.set_xlabel('dGaze (deg/sec)')

    ax2.hist(dHead, bins=21, density=True)
    ax2.set_xlabel('dHead (deg/sec)')
    
    ax3.plot(dEye[::20], dHead[::20], 'k.')
    ax3.plot([-900,900], [900,-900], 'r:')
    ax3.set_xlabel('dEye (deg/sec)')
    ax3.set_ylabel('dHead (deg/sec)')
    ax3.set_xlim((-900,900))
    ax3.set_ylim((-900,900))
    ax3.axes('square')

    ax4.plot(pitch_interp[::100], theta[::100], 'k.')
    ax4.set_xlabel('pitch (deg)')
    ax4.set_ylabel('theta (deg)')
    ax4.set_ylim([-60,60])
    ax4.set_xlim([-60,60])
    ax4.plot([-60,60],[-60,60], 'r:')

    ax5.plot(roll_interp[::100], phi[::100], 'k.')
    ax5.set_xlabel('roll (deg)')
    ax5.set_ylabel('phi (deg)')
    ax5.set_ylim([-60,60])
    ax5.set_xlim([-60,60])
    ax5.plot([-60,60],[60,-60], 'r:')

    plt.tight_layout()
    pdf.savefig(); plt.close()


    ### More diagnostic figures

    fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)

    ax0.plot(roll_interp[::100], theta[::100], 'k.')
    ax0.set_xlabel('roll (deg)')
    ax0.set_ylabel('theta (deg)')
    ax0.set_ylim([-60,60])
    ax0.set_xlim([-60,60])

    ax1.plot(pitch_interp[::100], phi[::100], 'k.')
    ax1.set_xlabel('pitch (deg)')
    ax1.set_ylabel('phi (deg)')
    ax1.set_ylim([-60,60])
    ax1.set_xlim([-60,60])

    ax2.hist(centered_pitch, bins=50, color='k', density=True)
    ax2.set_xlabel('pitch (deg)')

    ax3.hist(centered_roll, bins=50, color='k', density=True)
    ax3.set_xlabel('roll (deg)')

    ax4.hist(theta, bins=50, color='k', density=True)
    ax4.set_xlabel('theta (deg)')

    ax5.hist(phi, bins=50, color='k', density=True)
    ax5.set_xlabel('phi (deg)')

    plt.tight_layout()
    pdf.savefig(); plt.close()

    ### Tuning to eye/pupil properties

    # Calculate pupil radius
    longaxis = eye_data['ellipse_parameters']['longaxis'].copy()
    norm_longaxis = (longaxis - np.mean(longaxis)) / np.std(longaxis)

    # normalize eye position
    norm_theta = (theta - np.nanmean(theta)) / np.nanstd(theta)
    norm_phi = (phi - np.nanmean(phi)) / np.nanstd(phi)

    # pupil radius tuning
    pupilR_range = np.linspace(10,50,10)

    _bins, _tuning, _err = fmEphys.calc_all_tuning(spike_rates, longaxis, pupilR_range, eyeT)
    save_dict['pupilR_tuning_bins'] = _bins
    save_dict['pupilR_tuning_curve'] = _tuning
    save_dict['pupilR_tuning_error'] = _err

    fmEphys.plot_all_tuning(_bins, _tuning, _err, pdf, 'pupilR')

    # tuning to theta and phi
    thphi_range = np.linspace(-30,30,10)

    # theta
    _bins, _tuning, _err = fmEphys.calc_all_tuning(spike_rates, theta, thphi_range, eyeT)
    save_dict['theta_tuning_bins'] = _bins
    save_dict['theta_tuning_curve'] = _tuning
    save_dict['theta_tuning_error'] = _err

    fmEphys.plot_all_tuning(_bins, _tuning, _err, pdf, 'theta')

    # phi
    _bins, _tuning, _err = fmEphys.calc_all_tuning(spike_rates, phi, thphi_range, eyeT)
    save_dict['phi_tuning_bins'] = _bins
    save_dict['phi_tuning_curve'] = _tuning
    save_dict['phi_tuning_error'] = _err

    fmEphys.plot_all_tuning(_bins, _tuning, _err, pdf, 'phi')

    pdf.close()

    ### Add more data to save_dict

    add_dict = {

        # time correction
        'ephys_offset': offset,
        'ephys_drift': drift,

        # timestamps
        'topT': topT,
        'eyeT': eyeT,
        'worldT': worldT,
        'imuT': imuT,
        'modelT': modelT,

        # pupil
        'theta': theta,
        'phi': phi,
        'pupilR': longaxis,

        # head
        'gyroX': imu_data['gyro_x'],
        'gyroY': imu_data['gyro_y'],
        'gyroZ': gyro_z,
        'pitch': pitch,
        'roll': roll,

        # speed
        'dEye': dEye,
        'deye_dpf': dEye_dpf,
        'dGaze': dGaze,
        'dHead': dHead,

        # movement
        'bool_model_active': use_as_active,
        'model_active': model_active,

        # ephys
        'model_nsp': model_nsp,
        'ephysT0': ephysT0,
        'spike_rate': fmEphys.series_to_arr(ephys_data['rate']),
        'spike_times': fmEphys.series_to_list(ephys_data['spikeT']),
        'raw_spike_times': fmEphys.series_to_list(ephys_data['spikeTraw']),

        # exisitng ephys data/props
        'Phy2_ind': fmEphys.series_to_list(ephys_data['Phy2_ind']),
        'inds': ephys_data.index.values,
        'channel': fmEphys.series_to_list(ephys_data['ch']),
        'Phy2_amplitude': fmEphys.series_to_list(ephys_data['amp']),
        'Phy2_label': fmEphys.series_to_list(ephys_data['group']),
        'KS_label': fmEphys.series_to_list(ephys_data['KSLabel']),
        'firing_rate': fmEphys.series_to_list(ephys_data['fr']),
        'Phy2_ContamPct': fmEphys.series_to_list(ephys_data['ContamPct']),

        # saccade times
        'left_gazeshift_times': saccade_events['gazeL'],
        'right_gazeshift_times': saccade_events['gazeR'],
        'left_compensatory_times': saccade_events['compL'],
        'right_compensatory_times': saccade_events['compR'],

        # saccade PSTHs
        'left_gazeshift_PSTH': all_eyehead_psth['gazeL'],
        'right_gazeshift_PSTH': all_eyehead_psth['gazeR'],
        'left_compensatory_PSTH': all_eyehead_psth['compL'],
        'right_compensatory_PSTH': all_eyehead_psth['compR'],

        # video/stim info
        'model_world_vid': model_vid,
        'world_vid': worldvid,
        'stim_contrast': contrast,
        'STA': all_sta,
        'STV': all_stv

    }

    # merge the dict built during analysis (save_dict) with the new
    # one of gathered data (add_dict)
    save_dict = {**save_dict, **add_dict}

    # add a prefix to all of the keys so that we kow which recording
    # they came from
    save_dict = {'{}_{}'.format(cfg['_stim_abbrv'], k): v for k,v in save_dict.items()}

    savepath = os.path.join(cfg['rpath'], '{}_analysis.h5'.format(cfg['rfname']))
    fmEphys.write_h5(savepath, save_dict)


