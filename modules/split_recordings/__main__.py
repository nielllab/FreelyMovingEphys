"""
FreelyMovingEphys/modules/split_recordings/split_recordings.py
"""
import argparse
import PySimpleGUI as sg

from src.utils.auxiliary import str_to_bool
from src.prelim import RawEphys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--matfile', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_args()

    if args.matfile is None:
        # if no path was given as an argument, open a dialog box
        matfile = sg.popup_get_file('Choose .mat file.')
    else:
        matfile = args.matfile

    rephys = RawEphys(matfile)
    rephys.format_spikes()
