"""
FreelyMovingEphys/modules/session_analysis/__main__.py
"""
import argparse
import PySimpleGUI as sg

from core.utils.aux import str_to_bool
from core.run import Session

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()

    if args.config is None:
        # if no path was given as an argument, open a dialog box
        config_path = sg.popup_get_file('Choose animal config.yaml')
    else:
        config_path = args.config

    sess = Session(config_path)
    sess.run_main()
