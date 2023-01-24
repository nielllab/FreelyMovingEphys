"""
.pipeline.py

"""
import os
import sys
import argparse
import warnings
import PySimpleGUI as sg

import fmEphys

warnings.filterwarnings("ignore")

def pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('-l', '--log', type=fmEphys.str_to_bool, default=False)
    args = parser.parse_args()

    if args.cfg is None:
        # if no path was given as an argument, open a dialog box
        sg.theme('Default1')
        cfg_path = sg.popup_get_file('Choose animal ephys_cfg.yaml')
    else:
        cfg_path = args.cfg

    if args.log is True:
        sys.stdout = fmEphys.Logger(os.path.split(cfg_path)[0])

    sess = fmEphys.Session(cfg_path)
    sess.run_main()


if __name__ == '__main__':

    pipeline()
