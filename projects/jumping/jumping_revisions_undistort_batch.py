# jumping_revisions_undistort_batch.py

import sys, os
sys.path.insert(0, r'C:\Users\nlab\Desktop\GitHub Code\FreelyMovingEphys')
from src.run import Session
from src.sidecam import Sidecam
import os,fnmatch
import deeplabcut as dlc

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

#a function to find the files we want
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files: 
            if fnmatch.fnmatch(name,pattern): 
                result.append(os.path.join(root,name))
    if len(result)==1:
        result = result[0]
    return result

config_path = r'T:\jumping_revisions_training\jump_cfg.yaml'
expt_path = r'T:\jumping_revisions'
dates = get_immediate_subdirectories(expt_path)
for date in dates[20:]:
    date_directory = os.path.join(r'T:\jumping_revisions',date)
    print('doing %s' % date)
    animal_directories = [os.path.join(date_directory, x) for x in next(os.walk(date_directory))[1]]
    for animal_directory in animal_directories:
        sess = Session(config_path)
        sess.config['animal_directory'] = animal_directory
        sc = Sidecam(sess.config, None, None, 'SIDE')
        sc.undistort(mtxkey='sidecam_mtx', readcamkey='SIDE', savecamkey='_SIDEcalib.avi', checkervid='sidecam_checkerboard')