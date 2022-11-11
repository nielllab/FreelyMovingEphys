"""
Calculate distortion in a camera image.

Assumes a board w/ width=7 and height=5 checkerboard.
"""

import os
import argparse
import PySimpleGUI as sg

import fmEphys

def main(vidpath, savedir, camname):
    sg.theme('Default1')
    if vidpath is None:
        vidpath = sg.popup_get_file('Checkerboard video file')
    if savedir is None:
        savedir = sg.popup_get_folder('Save folder')
    if camname is None:
        camname = sg.popup_get_text('Camera name (e.g. REYE)')

    str_date, _ = fmEphys.fmt_now()
    savedir = os.path.join(savedir, '{}_cam_warp_mtx_{}.h5'.format(camname, str_date))

    fmEphys.calc_distortion(vidpath, savedir)

def calcWarp():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=None) # path
    parser.add_argument('--cam', type=str, default=None) # directory
    parser.add_argument('--dir', type=str, default=None) # directory
    args = parser.parse_args()

    main(args.video, args.dir, args.cam)
