"""
fmEphys/calcWarp.py

Calculate distortion in a camera image.
Assumes the input video will use a board with width=7 and
height=5 checkerboard.

Command line arguments
----------------------
--vidpath
    Path to the video file.
--savedir
    Path to the directory where the warp matrix will be saved.
--cam
    Name of the camera (e.g., 'TOP1').


Written by DMM, 2021
"""


import os
import argparse

import fmEphys as fme


def calcWarp():

    parser = argparse.ArgumentParser()
    parser.add_argument('--vidpath', type=str, default=None) # path
    parser.add_argument('--savedir', type=str, default=None) # directory
    parser.add_argument('--cam', type=str, default=None) # directory
    args = parser.parse_args()

    vidpath = args.vidpath
    savedir = args.savedir
    camname = args.cam

    if vidpath is None:
        print('Select checkerboard video file.')
        vidpath = fme.select_file(
            title='Select checkerboard video file.',
            filetypes=[('AVI','.avi'),]
        )
    if savedir is None:
        print('Select save directory.')
        savedir = fme.select_directory(
            title='Select save directory.'
        )
    if camname is None:
        print('Specify camera name (e.g., TOP1, WORLD).')
        camname = fme.get_string_input(
            title='Specify camera name (e.g., TOP1, WORLD).'
        )

    str_date, _ = fme.fmt_now()
    savedir = os.path.join(savedir, '{}_cam_warp_mtx_{}.h5'.format(camname, str_date))

    fme.calc_distortion(vidpath, savedir)


if __name__ == '__main__':
    
    calcWarp()

