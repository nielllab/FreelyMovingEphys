"""
.pipeline.py

"""
import os
import argparse
import warnings
import PySimpleGUI as sg

import fmEphys

warnings.filterwarnings("ignore")

def pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-l', '--log', type=fmEphys.str_to_bool, default=False)
    args = parser.parse_args()

    if args.config is None:
        # if no path was given as an argument, open a dialog box
        sg.theme('Default1')
        config_path = sg.popup_get_file('Choose animal ephys_cfg.yaml')
    else:
        config_path = args.config

    if args.log is True:
        fmEphys.start_log(os.path.split(config_path)[0])

    sess = fmEphys.Session(config_path)
    sess.run_main()


if __name__ == '__main__':

    pipeline()
