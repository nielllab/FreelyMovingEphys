"""
fmEphys/splitRec.py

Split recordings into individual ephys files. Files for all stimuli
were merged before spike sorting.

Command line arguments
----------------------
-f, --matfile
    Path to .mat file containing metadata for merged ephys data. If
    no path is given, a dialog box will open to select a file.

Example use
-----------
From a terminal:
    $ python -m fmEphys.splitRec -f T:/Path/to/mergedEphys.mat
Or choosing the file in a popup window:
    $ python -m fmEphys.splitRec

    
Written by DMM, 2021
"""


import argparse
import PySimpleGUI as sg

import fmEphys as fme


def splitRec():
    """ Split the merged ephys data into individual recordings.

    This function is called from the command line. It takes a .mat file
    containing the metadata for merged ephys data as an argument. It
    then reads the merged binary file, splits the data into individual
    recordings using timing information in the .mat, and saves each
    recording as a .json file containing spike data.
    """

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--matfile', type=str, default=None)
    args = parser.parse_args()

    if args.matfile is None:
        # If no path was given as an argument, open a dialog box
        sg.theme('Default1')
        matfile = sg.popup_get_file('Choose .mat file.')
    else:
        matfile = args.matfile

    # Create the RawEphys object
    rephys = fme.RawEphys(matfile)
    
    # Write a .json file of spike data for each stimulus
    rephys.format_spikes()


if __name__ == '__main__':

    splitRec()

