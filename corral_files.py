#####################################################################################
"""
corral_files.py of FreelyMovingEphys

Renames files so that there is a preceding 1 before single digits so that glob
function can tell apart _1 from _11

last modified: June 11, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################

#

from glob import glob
import pandas as pd
import os.path
import h5py
import numpy as np
import xarray as xr
import h5netcdf

from topdown_preening import preen_topdown_data
from eye_tracking import eye_angles
from time_management import read_time
from data_reading_utilities import read_data

main_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort?/J463c(blue)/*/Approach/'
save_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort3/J463c(blue)/110719/CorraledApproach/'

topdown_file_list = set(glob(main_path + '*top*DeepCut*.h5')) - set(glob(main_path + '*DeInter*.h5'))
righteye_file_list = set(glob(main_path + '*eye1r*DeepCut*.h5')) - set(glob(main_path + '*DeInter*.h5'))
lefteye_file_list = set(glob(main_path + '*eye2l*DeepCut*.h5')) - set(glob(main_path + '*DeInter*.h5'))
acc_file_list = glob(main_path + '*acc*.dat')
# camera time files
righteye_time_file_list = glob(main_path + '*eye1r*TS*.csv')
lefteye_time_file_list = glob(main_path + '*eye2l*TS*.csv')
topdown_time_file_list = glob(main_path + '*topTS*.csv')

for file in topdown_time_file_list:
    split_path = os.path.split(file)
    file_name = split_path[1]
    path_mising_pos = 29 # topdown=27, right/lefteye=29 right/lefttime=31 # topdowntime=29
    if file_name[path_mising_pos].isalpha():
        output_name1 = file_name[:(path_mising_pos-1)] + '0' + file_name[(path_mising_pos-1):]
        output_name = save_path + '/' + output_name1
    elif file_name[path_mising_pos].isdigit():
        output_name = save_path + file_name
    else:
        output_name1 = file_name[:(path_mising_pos - 1)] + '0' + file_name[(path_mising_pos - 1):]
        output_name = save_path + '/' + output_name1

    print(output_name)

    if 'TS' not in output_name:
        open_file = pd.read_hdf(file)
        open_file.to_hdf(output_name, key='data', mode='w')
    elif 'TS' in output_name:
        open_file = pd.read_csv(file)
        open_file.to_csv(output_name)