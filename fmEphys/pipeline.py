"""
fmEphys/pipeline.py

Run the pipeline for a single session.

Functions
---------
pipeline: Run the preprocessing and analysis pipeline.

Command line arguments
----------------------
-c, --cfg
    Path to the config file. If no path is given, a dialog box
    will open to select a file.
-l, --log
    True or False, whether to log the output to a file.

Example use
-----------
Running from a terminal:
    $ python -m fmEphys.pipeline -c T:/Path/to/ephys_cfg.yaml -l True
Or choosing the file in a popup window:
    $ python -m fmEphys.pipeline


Written by DMM, 2022
"""


import os
import sys
import argparse
import warnings

import fmEphys as fme

warnings.filterwarnings("ignore")


def pipeline():
    """Run the preprocessing and analysis pipeline.
    """

    # Get user arguments
    parser = argparse.ArgumentParser()

    # Path to the config file
    parser.add_argument('-c', '--cfg', type=str)

    # True or False, whether to log the output to a file
    parser.add_argument('-l', '--log', type=fme.str_to_bool,
                        default=True)
    args = parser.parse_args()

    # If no config file path is specified, ask the user to choose
    # one in a popup window
    if args.cfg is None:
        print('Choose animal ephys_cfg.yaml file.')
        cfg_path = fme.select_file(
            title='Choose animal ephys_cfg.yaml',
            filetypes=[('YAML','.yaml'),('YML','.yml'),]
        )
    # Otherwise, use the path specified by the user as an argument
    else:
        cfg_path = args.cfg

    # If the user wanted to log the output, set the filepath for that
    # to be the same as the config path.
    if args.log is True:
        sys.stdout = fme.Logger(os.path.split(cfg_path)[0])

    # Create the session analysis object and run the pipeline.
    sess = fme.Session(cfg_path)
    sess.run_main()


if __name__ == '__main__':

    pipeline()
