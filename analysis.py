"""
analysis.py

ephys analysis and figures

Oct. 19, 2020
"""

import argparse, json, sys, os

from util.read_data import find
from util.analyze_ephys import ephys_figures
from util.analyze_jump import jump_cc, jump_gaze_trace

# get user inputs
def pars_args():
    parser = argparse.ArgumentParser(description='run analysis on preprocessed data')
    parser.add_argument('-c', '--json_config_path', 
        default='~/Desktop/analysis_config.json',
        help='path to analysis config file')
    args = parser.parse_args()
    
    return args

def main(args):
    json_config_path = os.path.expanduser(args.json_config_path)

    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    exp_type = config['exp_type']
    
    # analyze if jumping experiment
    if exp_type['jumping'] is True:
        print('starting jumping analysis for ' + config['trial_name'])
        # find the .nc files
        REYE = find((config['trial_name'] + '*Reye*.nc'), config['data_path'])
        LEYE = find((config['trial_name'] + '*Leye*.nc'), config['data_path'])
        TOP = find((config['trial_name'] + '*Top*.nc'), config['data_path'])
        SIDE = find((config['trial_name'] + '*Side*.nc'), config['data_path'])
        # find the .avi video for the side camera
        SIDE_VID = find((config['trial_name'] + '*Side*.avi'), config['data_path'])
        # get figures of cross correlation in gaze vs head angle
        print('making correlation figures')
        jump_cc(REYE, LEYE, TOP, SIDE, config)
        # plot a video of head angle vs combined gaze during jump
        print('plotting video of gaze and head angle')
        jump_gaze_trace(REYE, LEYE, TOP, SIDE, SIDE_VID, config)

    elif exp_type['freelymovingephys'] is True:
        print('starting freely moving ephys analysis for ' + config['trial_name'])
        

if __name__ == '__main__':
    args = pars_args()
    main(args)