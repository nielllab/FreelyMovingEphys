"""
modify_dir_for_anipose.py

Create a copy of an experiment directory with files prepared for Anipose analysis

Last modified August 29, 2020
"""
import argparse
import fnmatch
import numpy as np
import os
import shutil

from util.read_data.py import find

# get user inputs
parser = argparse.ArgumentParser(description='create a copy of an experiment directory with files prepared for Anipose analysis')
parser.add_argument('-d', '--data_path')
parser.add_argument('-s', '--save_path')
args = parser.parse_args()

file_list = find('*TOP*TS.csv', args.data_path) + find('*TOP?.avi', args.data_path)

for this_file in file_list:
    if 'checkerboard' not in this_file:
        this_file_name = os.path.split(this_file)[-1]
        date_name = this_file_name.split('_')[0]; mouse_name = this_file_name.split('_')[1]; trial_name = this_file_name.split('_')[2] + '_' + this_file_name.split('_')[3]
        new_save_path = os.path.join(args.save_path, str(date_name + '/' + mouse_name + '/' + trial_name + '/videos-raw/'))
        file_out = new_save_path + this_file_name
        print(this_file_name + ' moving to ' + file_out)
        if not os.path.exists(new_save_path):
            os.makedirs(new_save_path)
        # print('new save path is ' + new_save_path)
        shutil.copyfile(this_file, file_out)

print('done with ' + str(len(file_list)) + ' .csv and .avi items')
