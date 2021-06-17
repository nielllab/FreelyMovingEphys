"""
format_mulit_ephys.py
"""
import argparse, yaml, sys, os, subprocess, shutil
import tkinter as tk
from tkinter import filedialog

from util.ephys import format_spikes_multi


def split_recordings():
    try:
        config_path = '/'.join(os.path.abspath(__file__).split('\\')[:-2]) + '/example_configs/config.yaml'
        # read in the json
        with open(config_path, 'r') as infile:
            config = yaml.load(infile, Loader=yaml.FullLoader)
    except FileNotFoundError:
        config_path = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/example_configs/config.yaml'
        with open(config_path, 'r') as infile:
            config = yaml.load(infile, Loader=yaml.FullLoader)

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    mat_path = os.path.join(os.getcwd(), file_path)

    format_spikes_multi(mat_path, config)
