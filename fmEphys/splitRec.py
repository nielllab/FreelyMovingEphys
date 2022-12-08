"""
FreelyMovingEphys/modules/split_recordings/split_recordings.py
"""
import argparse
import PySimpleGUI as sg

import fmEphys

def splitRec():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--matfile', type=str, default=None)
    args = parser.parse_args()

    if args.matfile is None:
        # if no path was given as an argument, open a dialog box
        sg.theme('Default1')
        matfile = sg.popup_get_file('Choose .mat file.')
    else:
        matfile = args.matfile

    rephys = fmEphys.RawEphys(matfile)
    rephys.format_spikes()

if __name__ == '__main__':
    splitRec()
    
