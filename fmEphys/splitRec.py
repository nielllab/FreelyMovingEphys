""" Recording split.

Electrophysiology data from all stimuli (e.g. head-fixed white noise, freely moving, etc.),
before spike sorting, was merged into a single binary file. Before preprocessing can be run
on the data for each recording, this script must be run to split the ephys data back into
separate files. The boundries for each recording were saved into the .mat file, which you
will select from a dialogue box or give as an argument when running this script.

Example
-------
This is run using
  $ python -m fmEphys.recsplit
Once this is run, a window will open in which you will select the merge .mat file. Alternatively,
you can run it with
  $ python -m fmEphys.recsplit -f /path/to/merge.mat
to avoid selecting the .mat file in a window.

Notes
-----
  * If the data has moved since the .mat file was written, the file paths saved in the .mat
    will be wrong and point the code to the wrong place. You can make a new .mat file with
    correct paths using the Matlab script /matlab/updateBasePath.m

"""

import argparse
import PySimpleGUI as sg

import fmEphys

def splitRec():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--matfile', type=str, default=None)
    args = parser.parse_args()

    # If there isn't a path as an argument, open a window so one can be selected
    if args.matfile is None:
        sg.theme('Default1')
        matfile = sg.popup_get_file('Choose merge .mat file')
    else:
        matfile = args.matfile

    fmEphys.do_data_split(matfile)

if __name__ == '__main__':
    splitRec()
    