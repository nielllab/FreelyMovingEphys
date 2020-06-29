#####################################################################################
"""
rename_files.py

Rename prey-capture DeepLabCut outputs and camera videos so that there is a preceding
zero before a single digit number in the trial-identifying sections of file names.
An improved version of corral_files.py.

last modified: June 24, 2020
"""
#####################################################################################

import os, glob
import re

main_path = '/home/eabe/Desktop/Test_Data_PreyCapture/Approach'
save_path1 = '/home/eabe/Desktop/Test_Data_PreyCapture/Approach/Renamed2/'
vid_path = '/home/eabe/Desktop/Test_Data_PreyCapture/Approach/'
vid_path_save = '/home/eabe/Desktop/Test_Data_PreyCapture/Approach/Renamed2/'

topdown_file_list = sorted(glob.glob(os.path.join(main_path, '*top*DeepCut*.h5')))
righteye_file_list = sorted(glob.glob(os.path.join(main_path, '*eye1r*DeInter2*.h5')))
lefteye_file_list = sorted(glob.glob(os.path.join(main_path, '*eye2l*DeInter2*.h5')))
right_TS = sorted(glob.glob(os.path.join(main_path, '*eye1rTS*.csv')))
left_TS = sorted(glob.glob(os.path.join(main_path, '*eye2lTS*.csv')))
top_TS = sorted(glob.glob(os.path.join(main_path, '*topTS*.csv')))
topdown_vid_list = sorted(glob.glob(os.path.join(main_path, '*top*.avi')))
righteye_vid_list = sorted(glob.glob(os.path.join(main_path, '*eye1r*DeInter2*.avi')))
lefteye_vid_list = sorted(glob.glob(os.path.join(main_path, '*eye2l*DeInter2*.avi')))

# topdown=27, right/lefteye=29 right/lefttime=31 # topdowntime=29

# rename_files(topdown_file_list, 27, save_path)
# rename_files(righteye_file_list, 29, save_path)
# rename_files(lefteye_file_list, 29, save_path)
# rename_files(right_TS, 31, save_path)
# rename_files(left_TS, 31, save_path)
# rename_files(top_TS, 29, save_path)

# get a list of all numbers separated by lower case characters

def rename(vid_list, pos, save_path):
    for filename in vid_list:
        paths = filename.split('/')
        prefix = paths[-1].split('_')
        numbers = re.findall('\d+',prefix[pos])
        num = numbers[0].zfill(2)
        prefix[pos] = num + prefix[pos][(len(numbers[0])):]
        new_filename = '_'.join(prefix)
        os.rename(os.path.join(main_path, paths[-1]), os.path.join(save_path, new_filename))
        print(os.path.join(save_path, new_filename))


rename(topdown_file_list,4,save_path1)
rename(righteye_file_list,4,save_path1)
rename(lefteye_file_list,4,save_path1)
rename(right_TS,4,save_path1)
rename(left_TS,4,save_path1)
rename(top_TS,4,save_path1)
rename(topdown_vid_list,4,vid_path_save)
rename(righteye_vid_list,4,vid_path_save)
rename(lefteye_vid_list,4,vid_path_save)