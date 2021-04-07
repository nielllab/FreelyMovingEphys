"""
jump_eyecam_analysis.py
"""
import argpase, os, json, sys

sys.path.insert(0, '/home/niell_lab/Documents/github/FreelyMovingEphys')

from manual_preprocessing import main as preprocessing_main
from project_analysis.jumping.jump_utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jump_config_path', type=str)
    args = parser.parse_args()
    return args

def main(args):
    # read config of jump options
    jump_config_path = args.jump_config
    with open(jump_config_path, 'r') as fp:
        j_config = json.load(fp)
    # read config of preprocessing options
    preprocessing_config_path = jump_config['preprocessing_config_path']
    with open(preprocessing_config_path, 'r') as fp:
        p_config = json.load(fp)
    # collect data into recoding directories
    organize_dirs(j_config)
    # run preprocessing
    preprocessing_main(p_config)
    # run jump analysis
    jump_analysis(j_config)

if __name__ == '__main__':
    args = get_args()
    main(args)