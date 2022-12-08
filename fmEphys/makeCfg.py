import os
import shutil
import argparse
import PySimpleGUI as sg

import fmEphys

def makeCfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--savedir', type=str, default=None)
    args = parser.parse_args()

    if args.savedir is None:
        # if no path was given as an argument, open a dialog box
        sg.theme('Default1')
        savedir = sg.popup_get_folder('Choose animal directory.')
        # usetemplate = sg.popup_get_text('Use template', default_text='ephys_cfg')
    else:
        savedir = args.savedir

    readpath = os.path.join(fmEphys.up_dir(__file__, 2), 'pipeline_cfg.yml')
    savepath = os.path.join(savedir, 'pipeline_cfg.yml')

    shutil.copyfile(readpath, savepath)

if __name__ == '__main__':

    makeCfg()

