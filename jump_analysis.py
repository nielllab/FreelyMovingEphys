"""
jump_analysis.py

analysis and figures for jumping experiments
preprocessing.py must be run first

Dec. 09, 2020
"""
# package imports
import argparse, json, sys, os
import xarray as xr
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
# module imports
from util.paths import find
from util.analyze_jump import jump_cc, jump_gaze_trace

def main(json_config_path):
    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    # find all the text files that contain recording metadata
    text_file_list = find('*.txt', config['data_path'])
    # remove vidclip files from the metadata list
    vidclip_file_list = find('*vidclip*.txt', config['data_path'])
    for x in vidclip_file_list:
        text_file_list.remove(x)
    # iterate through the text files
    for trial_path in text_file_list:
        # read the trial metadata data in
        with open(trial_path) as f:
            trial_contents = f.read()
        trial_metadata = json.loads(trial_contents)
        # get the name of the file
        trial_path_noext = os.path.splitext(trial_path)[0]
        head, trial_name_long = os.path.split(trial_path_noext)
        trial_name = '_'.join(trial_name_long.split('_')[:-1])
        config['recording_name'] = trial_name; config['trial_head'] = head
        # find the matching sets of .nc files produced during preprocessing
        leye = xr.open_dataset(find((trial_name + '*Leye.nc'), head)[0])
        reye = xr.open_dataset(find((trial_name + '*Reye.nc'), head)[0])
        side = xr.open_dataset(find((trial_name + '*side.nc'), head)[0])
        top = xr.open_dataset(find((trial_name + '*top.nc'), head)[0])
        side_vid = find((trial_name + '*Side*.avi'), head)[0]
        # correlation figures
        trial_cc_data = jump_cc(reye, leye, top, side, config)
        # plot over video
        trial_gaze_data = jump_gaze_trace(reye, leye, top, side, side_vid, config)

        # merge all data for this trial here... should be an xarray...
        # this will depend on the format of the trial data

        # if trial_path == text_file_list[0]:
        #     pooled_data = trial_data_xr.copy()
        # else:
        #     pooled_data = xr.merge([pooled_data, trial_data)

if __name__ == '__main__':
    # args = pars_args()
    
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    
    main(file_path)