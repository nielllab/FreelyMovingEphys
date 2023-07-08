"""
fmEphys/pipelineBatch.py

Run the pipeline for a sequence of multiple sessions.

Command line arguments
----------------------
-c, --cfg
    Path to the config file, a copy of pipeline_cfg.yml
    with the appropriate metadata.
-j, --recJson
    Path to the .json file containing path data for
    each session.
-i, --iter
    Which keys of the dictionary in the JSON should be
    analyzed (it should be a specific key from the JSON).
    If not specified, the default value is 'all', and all
    keys in the JSON will be analyzed.
-l, --log
    Bool, whether to log the terminal outputs to a file.
    Default is True.

Example use
-----------
    $ python -m fmEphys.pipelineBatch -c T:/Path/to/pipeline_cfg.yml
        -j T:/Path/to/recording.json -i all -l True

Notes
-----
The recording JSON file should, once loaded as a dictionary,
contain an item for each session. Keys should be be letters
numbers which, when sorted, give the desired order that the
sessions should be analyzed in. The value for each session
should be dictionary with four keys:
    'date': The date of the session, in the format YYYYMMDD
    'animal': The animal ID
    'probe': The probe ID (e.g., 'NN_H16')
    'path': The path to the session directory.


Written by DMM, 2022
"""


import os
import sys
import json
import yaml
import argparse
import warnings
import PySimpleGUI as sg

import fmEphys as fme

warnings.filterwarnings('ignore')


def batch_cycle(cfg, cycle_dict):
    """ Run the pipeline for a single session.

    Parameters
    ----------
    cfg : dict
        The config dictionary.
    cycle_dict : dict
        A dictionary containing the metadata for a this
        session. This will be read in from the recording
        JSON file, and is the value specific to this
        session's key.
    
    """

    print(' ---> Starting {} {}'.format(cycle_dict['date'], cycle_dict['animal']))

    # Load the metadata for this session
    cfg['probe'] = cycle_dict['probe']
    cfg['animal_directory'] = cycle_dict['path']

    # Create the session object and run the analysis
    sess = fme.Session(cfg)
    sess.run_main()


def pipelineBatch():
    """ Run the pipeline for a sequence of multiple sessions.
    """

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('-j', '--recJson', type=str)
    parser.add_argument('-i', '--iter', type=str, default='all')
    parser.add_argument('-l', '--log', type=fme.str_to_bool, default=True)
    args = parser.parse_args()

    # Create the log file
    if args.log is True:
        sys.stdout = fme.Logger(os.path.split(args.cfg)[0])

    # Load the recording metadata JSON file
    with open(args.recJson, 'r') as fp:
        rec_dict = json.load(fp)

    # Load the config file for analysis parameters
    with open(args.cfg, 'r') as infile:
        cfg = yaml.load(infile, Loader=yaml.FullLoader)

    # If the user wants to analyze all sessions in the JSON
    # file, then iterate over all keys in the dictionary.
    if args.iter == 'all':

        # Sort the keys alphabetically
        for key in sorted(rec_dict.keys()):

            val = rec_dict[key]
            # Use the same config parameters for all sessions.
            # Only the path and probe ID will be changed between
            # sessions, and this is done inside of the batch_cycle
            # function.
            _use_cfg = cfg.copy()

            # Analyze this session.
            batch_cycle(_use_cfg, val)

    # Otherwise, just analyze the session specified by the
    # user.
    else:
        batch_cycle(cfg, rec_dict[args.iter])


if __name__ == '__main__':

    pipelineBatch()
