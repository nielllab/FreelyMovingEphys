"""
format_mulit_ephys.py

loop through each ephys recording, split the recordings from one another,
and save the ephys data as seperate recordings
searches desktop for config, opens dialog box for file to split

Jan. 15, 2021
"""
# package imports
import argparse, json, sys, os, subprocess, shutil
import tkinter as tk
from tkinter import filedialog
# module imports
from util.ephys import format_spikes_multi


def split_recordings():
    try:
        default_json_path = '/'.join(os.path.abspath(__file__).split('\\')[:-2]) + '/example_configs/preprocessing_config.json'
        # read in the json
        with open(default_json_path, 'r') as fp:
            default_config = json.load(fp)
    except FileNotFoundError:
        default_json_path = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/example_configs/preprocessing_config.json'
        with open(default_json_path, 'r') as fp:
            default_config = json.load(fp)

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    mat_path = os.path.join(os.getcwd(), file_path)

    format_spikes_multi(mat_path, config)
