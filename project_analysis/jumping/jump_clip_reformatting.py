"""
jump_clip_reformatting.py

organize sliced jumping recordings into directories, create metadata .txt files, and rename files

Dec. 11, 2020
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

def main(json_config_path):
    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    path_in = config['path_in']
    path_out = config['path_out']

    file_list = find('*.avi', path_in) + find('*.csv', path_in)
    # find all the text files that contain recording metadata
    text_file_list = find('*.txt', path_in)

    for file in file_list:
        print('copying ' + file)
        file_name = os.path.split(file)[-1]
        if 'BonsaiTS' not in file:
            base = '_'.join(file_name.split('_')[:-2])
            cam = file_name.split('_')[-2:-1][0]
            num = (file_name.split('_')[-1]).split('.')[0]
            
            new_dir = '_'.join([base, num])
            new_name = '_'.join([base, num, cam+'.avi'])
            save_path = os.path.join(path_out, new_dir)
            
            if os.path.exists(save_path) is False:
                os.makedirs(save_path)

            for meta in text_file_list:
                if base in meta:
                    old_meta_name = os.path.split(meta)[-1]
                    old_meta_noext = os.path.splitext(old_meta_name)[0]
                    if 'vidclip_ts' in old_meta_noext:
                        base_meta = '_'.join(old_meta_noext.split('_')[:-3])
                        ready_meta = '_'.join([base_meta, num, '_'.join(old_meta_noext.split('_')[-3:-1])])
                    elif 'vidclip_ts' not in old_meta_noext:
                        base_meta = '_'.join(old_meta_noext.split('_')[:-1])
                        ready_meta = '_'.join([base_meta, num, ''.join(old_meta_noext.split('_')[-1])])
                    new_meta_name = ready_meta + '.txt'
                    metadata_save_full = os.path.join(save_path, new_meta_name)
                    shutil.copyfile(meta, metadata_save_full)
                    print('saved ' + meta)
                
            save_path_full = os.path.join(save_path, new_name)
            
            # copy the file and rename
            shutil.copyfile(file, save_path_full)
            print('saved ' + file)
            
        if 'BonsaiTS' in file:
            base = '_'.join(file_name.split('_')[:-3])
            cam = file_name.split('_')[-3:-2][0]
            num = (file_name.split('_')[-2])
            bon = (file_name.split('_')[-1]).split('.')[0]
            
            new_dir = '_'.join([base, num])
            new_name = '_'.join([base, num, cam, bon+'.csv'])
            save_path = os.path.join(path_out, new_dir, new_name)
                
            # copy the file and rename
            shutil.copyfile(file, save_path)
            print('saved ' + file)

if __name__ == '__main__':
    
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    
    main(file_path)

