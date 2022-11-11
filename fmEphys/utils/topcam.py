import os
import argparse
import numpy as np

os.environ['DLClight'] = 'True'
import deeplabcut

import fmEphys

def preprocess_topcam(cfg=None, rpath=None):
    
    cfg = fmEphys.get_cfg(cfg)

    # Find the video
    raw_vid_path = fmEphys.find('*{}*{}*.avi'.format(cfg['rname'], cfg['camname']), cfg['rpath'], mr=True)
    print('Preprocessing topdown video {}'.format(raw_vid_path))

    # Pose estimation
    deeplabcut.analyze_videos(cfg['dlcPath_top'], [raw_vid_path])

    # Read in video and format as an array
    vidarr = fmEphys.avi_to_arr(raw_vid_path, ds=0.25)

    # Timestamps
    raw_camT_path = fmEphys.find('*{}*{}*BonsaiTS.csv'.format(cfg['rname'], cfg['camname']), cfg['rpath'], mr=True)
    camT = fmEphys.read_time(raw_camT_path)

    body_data = fmEphys.track_body(cfg)

    # Collect data to save out
    topcam_data = {
        'video_ds': vidarr,
        'time': camT,
        'options': cfg
    }

    topcam_data = {**topcam_data, **body_data}

    savecamname = cfg['camname'].lower()
    savepath = os.path.join(cfg['rpath'], '{}_{}_preprocessing.h5'.format(cfg['rname'], savecamname))

    fmEphys.write_h5(savepath, topcam_data)