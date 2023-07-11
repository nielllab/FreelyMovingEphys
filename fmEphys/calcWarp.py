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
import PySimpleGUI as sg

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

    sg.theme('Default1')
    if vidpath is None:
        vidpath = sg.popup_get_file('Checkerboard video file')
    if savedir is None:
        savedir = sg.popup_get_folder('Save folder')
    if camname is None:
        camname = sg.popup_get_text('Camera name (e.g. REYE)')

    str_date, _ = fme.fmt_now()
    savedir = os.path.join(savedir, '{}_cam_warp_mtx_{}.h5'.format(camname, str_date))

    fme.calc_distortion(vidpath, savedir)


if __name__ == '__main__':
    
    calcWarp()

