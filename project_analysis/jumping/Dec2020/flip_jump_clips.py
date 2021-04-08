"""
flip_jump_clips.py

flip jumping eye videos vertically

Dec. 14, 2020
"""
# package imports
import argparse, json, sys, os, shutil, subprocess
import xarray as xr
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
# module imports
from util.paths import find

def main(json_config):
    # open config file
    with open(json_config, 'r') as fp:
        config = json.load(fp)

    path_in = config['path_in']
    path_out = config['path_out']

    eye_vid_list = find('*EYE*.avi', path_in)

    for vid in eye_vid_list:
        head, tail = os.path.split(vid)
        new_tail = tail.split('.')[0] + '_unflipped.avi'
        new_path = os.path.join(head, new_tail)
        shutil.copy(vid, new_path)
        os.remove(vid)
        subprocess.call(['ffmpeg', '-i', new_path, '-vf', 'vflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', vid])

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    
    main(file_path)