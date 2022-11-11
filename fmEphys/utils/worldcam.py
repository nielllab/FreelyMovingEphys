

import os
import numpy as np

import fmEphys

def preprocess_worldcam(cfg=None, rpath=None):

    cfg = fmEphys.get_cfg(cfg)

    if rpath is not None:
        cfg['rpath'] = rpath

    if ('rname' not in cfg.keys()) or (cfg['rname'] is None):
        cfg['rname'] = fmEphys.get_rec_name(cfg['rpath'])

    # Find the video
    raw_vid_path = fmEphys.find('*{}*{}*.avi'.format(cfg['rname'], cfg['camname']), cfg['rpath'], mr=True)
    print('Preprocessing worldcam video {}'.format(raw_vid_path))

    # Deinterlace
    vid_path = fmEphys.deinterlace(raw_vid_path)

    # Fix camera warp
    sp = fmEphys.make_avi_savepath(raw_vid_path, 'deinter')
    vid_path = fmEphys.fix_distortion(vid_path, cfg['camMtx_wc'], savepath=sp)

    # Read in video and format as an array
    vidarr = fmEphys.avi2arr(vid_path, ds=0.25)
    nF = np.shape(vidarr, 0)

    # Read in timestamps
    raw_camT_path = fmEphys.find('*{}*{}*BonsaiTS.csv'.format(cfg['rname'], cfg['camname']), cfg['rpath'], mr=True)
    camT = fmEphys.read_time(raw_camT_path, dlen=nF)

    # Save everything together
    worldcam_data = {
        'video_ds': vidarr,
        'time': camT,
        'options': cfg
    }

    savecamname = cfg['camname'].lower()
    savepath = os.path.join(cfg['rname'], '{}_{}_preprocessing.h5'.format(cfg['rname'], savecamname))

    fmEphys.write_h5(savepath, worldcam_data)