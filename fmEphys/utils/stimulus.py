""" Neural responses to stimuli

"""

import os
import cv2
import numpy as np
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
    ephys_data_path = os.path.join(cfg['_rpath'], '{}_ephys_preprocessing.h5'.format(cfg['_rname']))
    ephys_data = fmEphys.read_h5(ephys_data_path)

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

    # Select cells that were labeled as 'good' in Phy2
    _keep = (ephys_data['group']=='good')
    ephys_data = ephys_data[_keep]

    # Sort ephys cells by the order of shank and site they were recorded from on the linear
    # ephys probe.
    ephys_data = ephys_data.sort_values(by='ch', axis=0, ascending=True)
    ephys_data = ephys_data.reset_index()
    ephys_data = ephys_data.drop('index', axis=1)
    
    # spike times
    ephys_data = ephys_data[:,'spikeT']


    self.ephys_data['spikeTraw'] = self.ephys_data['spikeT'].copy()
    # select good cells from phy2
    self.cells = self.ephys_data.loc[self.ephys_data['group']=='good']
    self.units = self.cells.index.values
    # get number of good units
    self.n_cells = len(self.cells.index)
    # make a raster plot
    self.spike_raster()


    
    
    cell_order = np.argsort(ephys_data['ch'])
    old2new_cell_n = dict(zip(ephys_data.keys(), cell_order))
    # Save the old keys as an entry in each nested dict
    for old, new in cell_order.items():
        ephys_data[old]['preprocessed_cell_n'] = old
    # Now, make a new dict with new keys.
    ephys_data = dict(zip(ephys_data))

    ephys_data[]
    ephys_data

    self.ephys_data = self.ephys_data.sort_values(by='ch', axis=0, ascending=True)
    self.ephys_data = self.ephys_data.reset_index()
    self.ephys_data = self.ephys_data.drop('index', axis=1)

    # spike times
    create 

    ephys_data

    D = {c['spikeT']
            for c, cvals in ephys_data.items()}


    self.ephys_data['spikeTraw'] = self.ephys_data['spikeT'].copy()
    # select good cells from phy2
    self.cells = self.ephys_data.loc[self.ephys_data['group']=='good']
    self.units = self.cells.index.values
    # get number of good units
    self.n_cells = len(self.cells.index)


    # Define theta (horizontal) and vertical ()
    theta = np.rad2deg(eye_data['ellipse_parameters']['theta'])
    theta = theta - np.nanmean(theta)
    # Flip phi so that up is positive & down is negative
    phi = np.rad2deg(eye_data['ellipse_parameters']['phi'])
    phi = - (phi - np.nanmean(phi))

    ### Some diagnostic plots of the data

    fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)

    ax0.imshow(np.mean(world_data['video_ds'], axis=0), cmap='gray')
    ax0.set_title('mean worldcam image')

    # 3600 samples will be the first 1 min of the recording
    ax1.plot(np.diff(world_data['time'])[:3600], color='k')
    ax1.set_xticks(np.linspace(0,60,6).astype(int))
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('worldcam deltaT (sec)')

    ax2.hist(np.diff(world_data), 100, density=True, color='k')
    ax2.set_xlabel('worldcam deltaT (sec)')

    ax3.plot(imu_data['gyro_z'][:3600], color='k')
    ax3.set_xticks(np.linspace(0,60,6).astype(int))
    ax3.set_xlabel('time (sec)')
    ax3.set_ylabel('gyro z (deg)')

    ax4.plot(np.diff(eye_data['time'])[:3600], color='k')
    ax4.set_xticks(np.linspace(0,60,6).astype(int))
    ax4.set_xlabel('time (sec)')
    ax4.set_ylabel('eyecam deltaT (sec)')
   
    ax5.hist(np.diff(eye_data['time']), bins=100, density=True, color='k')
    ax5.set_xlabel('worldcam deltaT (sec)')

    fig.tight_layout()
    pdf.savefig(); plt.close()


    plt.plot(self.ballT, self.ball_speed)
    plt.xlabel('sec'); plt.ylabel('running speed (cm/sec)')

    # Spike raster





def head_fixed(cfg):

    # treadmill data
    treadmill_data_path = os.path.join(cfg['_rpath'], '{}_treadmill_preprocessing.h5'.format(cfg['_rname']))
    treadmill_data = fmEphys.read_h5(ephys_data_path)