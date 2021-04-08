"""
get_jump_metadata.py

find and copy jump metadata to data directory

Dec. 14, 2020
"""
# package imports
import argparse, json, sys, os, shutil
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

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    txt_list = find('*.txt', path_in)
    for item in txt_list:
        head, tail = os.path.split(item)
        shutil.copy(item, os.path.join(path_out, tail))

    print('copied ' + str(len(txt_list)) + ' items')

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    
    main(file_path)