"""
jump_eyecam_analysis.py
"""
import argparse, os, json, sys

sys.path.insert(0, '/home/niell_lab/Documents/github/FreelyMovingEphys')

from session_analysis.session_analysis import main as preprocessing_main
from project_analysis.jumping.jump_utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()
    return args

def main(args):
    # read config of jump options
    with open(args.config_path, 'r') as fp:
        j_config = json.load(fp)
    # read config of preprocessing options
    preprocessing_config_path = j_config['preprocessing_config_path']
    # get steps to run
    steps = j_config['steps']
    # collect data into recoding directories
    if steps['organize_dirs']:
        organize_dirs(j_config)
    # run preprocessing
    if steps['preprocessing']:
        preprocessing_main(preprocessing_config_path)
    if steps['split_timebins']:
        split_nc_into_timebins(j_config)
    # run jump analysis
    if steps['analysis']:
        jump_analysis(j_config)

if __name__ == '__main__':
    args = get_args()
    main(args)