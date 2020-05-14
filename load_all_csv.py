#####################################################################################
"""
load_all_csv.py of FreelyMovingEphys

Loads in top-down camera and right or left eye from DLC outputs and data are aligned.

Requires alignment_from_DLC.py
Adapted from /niell-lab-analysis/freely moving/loadAllCsv.m

last modified: May 14, 2020
"""
#####################################################################################
from glob import glob
import pandas as pd
import os.path

from alignment_from_DLC import align_head_from_DLC

def check_and_read(file_input):
    if file_input == 'none':
        no_data = 0
        return no_data
    else:
        read_data = pd.read_csv(file_input, skiprows=2)
        return(read_data)

def read_data(topdown_input='none', acc_input='none', time_input='none', lefteye_input='none', righteye_input='none'):
    topdown_data = check_and_read(topdown_input)
    points = align_head_from_DLC(topdown_data)

    # ... other things it does that aren't filled out yet...

    return points

# find list of all data
main_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort?/*/*/Approach/'

topdown_file_list = glob(main_path + '*top*DeepCut*.csv')
acc_file_list = glob(main_path + '*acc*DeepCut*.dat')
time_file_list = glob(main_path + '*topTS*DeepCut*.csv')
righteye_file_list = glob(main_path + '*eye1r*DeepCut*.csv')
lefteye_file_list = glob(main_path + '*eye2l*DeepCut*.csv')

loop_count = 0
limit_of_loops = 1 # for testing purposes, limit to first file
for file in topdown_file_list:
    if loop_count < limit_of_loops:
        split_path = os.path.split(file)
        file_name = split_path[1]
        mouse_key = file_name[0:5]
        trial_key = file_name[17:27]
        acc_file = ', '.join([i for i in acc_file_list if mouse_key and trial_key in i])
        time_file = ', '.join([i for i in time_file_list if mouse_key and trial_key in i])
        righteye_file = ', '.join([i for i in righteye_file_list if mouse_key and trial_key in i])
        lefteye_file = ', '.join([i for i in lefteye_file_list if mouse_key and trial_key in i])
        points = read_data(file, acc_file, time_file, righteye_file, lefteye_file)
    loop_count = loop_count + 1