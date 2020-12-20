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
from util.analyze_jump import jump_cc, jump_gaze_trace, pooled_jump_analysis, animated_gaze_plot

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
        
    trial_count = 0
    # iterate through the text files
    for trial_path in text_file_list:
        trial_count = trial_count + 1
        # read the trial metadata data in
        with open(trial_path) as f:
            trial_contents = f.read()
        trial_metadata = json.loads(trial_contents)
        # get the name of the file
        trial_path_noext = os.path.splitext(trial_path)[0]
        head, trial_name_long = os.path.split(trial_path_noext)
        trial_name = '_'.join(trial_name_long.split('_')[:-1])
        config['recording_name'] = trial_name; config['trial_head'] = head
        print('analyzing '+config['recording_name'])
        # get the metadata out of vidclip text file
        for time_text_path in vidclip_file_list:
            if trial_name in time_text_path:
                with open(time_text_path) as f:
                    time_txt = f.read()
        time_dict = json.loads(time_txt)
        # find the matching sets of .nc files produced during preprocessing
        leye = xr.open_dataset(find((trial_name + '*Leye.nc'), head)[0])
        reye = xr.open_dataset(find((trial_name + '*Reye.nc'), head)[0])
        side = xr.open_dataset(find((trial_name + '*side.nc'), head)[0])
        top = xr.open_dataset(find((trial_name + '*top.nc'), head)[0])
        side_vid = find((trial_name + '*Side*.avi'), head)
        top_vid = find((trial_name + '*Top*.avi'), head)
        leye_vid = find((trial_name + '*LEYE*.avi'), head)
        reye_vid = find((trial_name + '*REYE*.avi'), head)
        for x in side_vid:
            if 'plot' in x:
                side_vid.remove(x)
        for x in top_vid:
            if 'plot' in x:
                top_vid.remove(x)
        for x in leye_vid:
            if 'plot' in leye_vid or 'unflipped' in leye_vid:
                leye_vid.remove(x)
        for x in reye_vid:
            if 'plot' in reye_vid or 'unflipped' in reye_vid:
                reye_vid.remove(x)               
        side_vid = side_vid[0]
        top_vid = top_vid[0]
        leye_vid = leye_vid[0]
        reye_vid = reye_vid[0]

        # correlation figures
        trial_cc_data = jump_cc(reye, leye, top, side, time_dict, trial_metadata, config)
        trial_cc_data.name = config['recording_name']
        # plot over video
        if config['plot_avi_vids'] is True:
            print('plotting jump gaze for side view of ' + config['recording_name'])
            jump_gaze_trace(reye, leye, top, side, side_vid, config)
            print('plotting videos with animated plots for ' + config['recording_name'])
            animated_gaze_plot(reye, leye, top, side, side_vid, leye_vid, reye_vid, top_vid, config)

        if trial_path == text_file_list[0]:
            pooled_data = trial_cc_data.copy()
        else:
            pooled_data = xr.merge([pooled_data, trial_cc_data])
        print('done with trial '+str(trial_count)+' of '+str(len(text_file_list)))
    
    print('saving pooled data at ' + config['data_path'])
    # save out an xarray of pooled data
    pooled_data.to_netcdf(os.path.join(config['data_path'], 'pooled_jump_data.nc'))

    print('making plots of pooled data for all trials')
    # make a pdf of pooled data
    pooled_jump_analysis(pooled_data, config)

    print('done analyzing ' + str(len(text_file_list)) + ' trials')

if __name__ == '__main__':
    # args = pars_args()
    
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    
    main(file_path)