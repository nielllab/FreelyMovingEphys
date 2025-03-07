"""
fmEphys/makeCfg.py

Make a pipeline configuration file for ephys data.
Uses the default options in the fiel pipeline_cfg.yml.

Command line arguments
----------------------
-dir, --savedir
    Path to the directory where the config file will be saved.

Example use
-----------
Running from a terminal:
    $ python -m fmEphys.makeCfg --savedir T:/Path/to/animal
Or, choosing the parameters in a popup window:
    $ python -m fmEphys.makeCfg


Written by DMM, 2021
"""


import os
import shutil
import argparse

import fmEphys as fme


def makeCfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--savedir', type=str, default=None)
    args = parser.parse_args()

    if args.savedir is None:
        # if no path was given as an argument, open a dialog box
        savedir = fme.select_directory('Choose animal directory.')
    else:
        savedir = args.savedir

    readpath = os.path.join(fme.up_dir(__file__, 2), 'pipeline_cfg.yml')
    savepath = os.path.join(savedir, 'pipeline_cfg.yml')

    shutil.copyfile(readpath, savepath)


if __name__ == '__main__':

    makeCfg()

