""" Eye camera preprocessing.

To do the preprocessing without a yaml file, some files paths will be
needed, that can be added to a dictionary and passed in as an argument.

```
tmp_cfg = {
    '_rpath': '/path/to/recording/directory',
    'rname': '010122_Animal_Rig_Experiment',
    'camname': 'REYE'
}
```

To run it from the terminal, you can do

```
python -m fmEphys.utils.eyecam preprocess -- 

--- 

Niell Lab
Written by DMM, Nov 2022
"""

import os
import argparse
import numpy as np

# os.environ['DLClight'] = 'True'
# import deeplabcut

import fmEphys

def preprocess_eyecam(cfg=None, rpath=None):

    cfg = fmEphys.get_cfg(cfg)
    cfg['_cam'] = cfg['raw_names']['reye']

    # Find the video
    raw_vid_path = fmEphys.find('*{}*{}*.avi'.format(cfg['_rname'], cfg['_cam']),
                                cfg['_rpath'], exclude=['plot', 'deinter'], mr=True)
    print('Preprocessing eyecam video {}'.format(raw_vid_path))

    # Deinterlace
    vid_path = fmEphys.deinterlace(raw_vid_path)

    # Pose estimation
    # deeplabcut.analyze_videos(cfg['dlcPath_reye'], [vid_path])

    # Read in video and format as an array
    vidarr = fmEphys.avi_to_arr(vid_path, ds=0.25)
    nF = np.size(vidarr, axis=0)

    # Read in timestamps
    raw_camT_path = fmEphys.find('*{}*{}*BonsaiTS.csv'.format(cfg['_rname'], cfg['_cam']), cfg['_rpath'], mr=True)
    camT = fmEphys.read_time(raw_camT_path, dlen=nF)

    pupil_data = fmEphys.track_pupil(cfg)

    # Collect data to save out
    eyecam_data = {
        'video_ds': vidarr,
        'time': camT,
        'options': cfg
    }

    eyecam_data = {**eyecam_data, **pupil_data}

    savecamname = cfg['_cam'].lower()
    savepath = os.path.join(cfg['_rpath'], '{}_{}_preprocessing.h5'.format(cfg['_rname'], savecamname))

    fmEphys.write_h5(savepath, eyecam_data)

if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    args = parser.parse_args()

    preprocess_eyecam(args.cfg)