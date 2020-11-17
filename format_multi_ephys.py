"""
format_mulit_ephys.py

loop through each ephys recording, split the recordings from one another, and save the
ephys data as seperate recordings
searches desktop for config, opens dialogue box for file to split

Nov. 17, 2020
"""

import argparse, json, sys, os, subprocess, shutil
from util.ephys import format_spikes_multi
import tkinter as tk
from tkinter import filedialog

# get user inputs
def pars_args():
    parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
    parser.add_argument('-c', '--json_config_path', 
        default='~/Desktop/preprocessing_config.json',
        help='path to video analysis config file')
    # parser.add_argument('-f', '--merge_file_name', help='location of the .mat merge file')
    args = parser.parse_args()
    
    return args

def main(args):
    json_config_path = os.path.expanduser(args.json_config_path)

    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    mat_path = os.path.join(os.getcwd(), file_path)

    format_spikes_multi(mat_path, config)

if __name__ == '__main__':
    args = pars_args()
    main(args)

    