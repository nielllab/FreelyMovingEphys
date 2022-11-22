""" Neural responses to stimuli

"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys

def freely_moving(cfg):
    """
    
    requires cfg dict entries for
        _rname
        _cam
        _stim
        _rpath
    
    """

    # Write a diagnostic PDF of all figures
    pdf_savepath = os.path.join(cfg['_rpath'],
            '{}_{}_diagnostics.pdf'.format(cfg['rname'], cfg['_cam']))
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
    sz = np.shape(world_data['video_ds'])
    # If size is larger than the expected (60,80), resize it now.
    if sz[1]>=160:
        dwnsmpl = 0.5
        raw_wc = world_data['video_ds'].copy()
        world_data['video_ds'] = np.zeros( (sz[0],
                                           int(sz[1]*dwnsmpl),
                                           int(sz[2]*dwnsmpl)), dtype='uint8')
        for f in range(sz[0]):
            world_data['video_ds'][f,:,:] = cv2.resize(raw_wc[f,:,:]
                                                      (int(sz[2]*dwnsmpl), int(sz[1]*dwnsmpl)))

    ### Clean up the ephys data

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

    # Spike times
    ephys_data['spikeTraw'] = ephys_data['spikeT'].copy()

    ### 

    # Define theta (horizontal) and vertical ()
    theta = np.rad2deg(eye_data['ellipse_parameters']['theta'])
    theta = theta - np.nanmean(theta)
    # Flip phi so that up is positive & down is negative
    phi = np.rad2deg(eye_data['ellipse_parameters']['phi'])
    phi = - (phi - np.nanmean(phi))

    # Timestamps
    eyeT = eye_data['time']
    worldT = world_data['time']
    imuT = imu_data['time']
    topT = top_data['time']

    ### Some diagnostic plots of the data that has been read in so far

    fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)

    ax0.imshow(np.mean(world_data['video_ds'], axis=0), cmap='gray')
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

    ax3.plot(ball_speed)
    ax3.set_xlabel('sec')
    ax3.set_ylabel('running speed (cm/sec)')


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

    imuT_raw = imuT_raw - ephysT0

    if (cfg['_stim'] != 'FmDk'):
        # Recordings in the dark do not have topdown camera data
        topT = topT - ephysT0

    # Calculate eye veloctiy
    self.dEye = np.diff(self.theta) # deg/frame
    self.dEye_dps = self.dEye / np.diff(self.eyeT) # deg/sec

    

    if np.isnan(self.ephys_drift_rate) and np.isnan(self.ephys_offset):
        # plot eye velocity against head movements
        plt.figure
        plt.plot(self.eyeT[0:-1], -self.dEye, label='-dEye')
        plt.plot(self.imuT_raw, self.gyro_z, label='gyro z')
        plt.legend()
        plt.xlim(0,10); plt.xlabel('secs'); plt.ylabel('gyro (deg/s)')
        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()





def head_fixed(cfg):

    offset=0.1, drift_rate=-0.000114

    # treadmill data
    treadmill_data_path = os.path.join(cfg['_rpath'], '{}_treadmill_preprocessing.h5'.format(cfg['_rname']))
    treadmill_data = fmEphys.read_h5(ephys_data_path)