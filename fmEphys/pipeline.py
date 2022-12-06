import argparse
import warnings
import PySimpleGUI as sg

import fmEphys

warnings.filterwarnings("ignore")

def pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    args = parser.parse_args()

    if args.config is None:
        # if no path was given as an argument, open a dialog box
        config_path = sg.popup_get_file('Choose animal config.yaml')
    else:
        config_path = args.config

    sess = fmEphys.Session(config_path)
    sess.run_main()


if __name__ == '__main__':

    pipeline()
