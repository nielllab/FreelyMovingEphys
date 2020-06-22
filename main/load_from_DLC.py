#####################################################################################
"""
load_from_DLC.py of FreelyMovingEphys/main/
(formerly: load_all_csv.py)

Loads in top-down camera and right and/or left eye from DeepLabCut .h5 file outputs.
The user provides to this script with a file path for all of the data, collected by a
glob function, and an xarray DataArray is built from each of these camera viewpoints
and combined into one DataArray for all trials. Topdown data are processing and
ellipse parameters are collected for each of the eyes (runs with 0, 1, or 2 sets of
eye data. Then, camera videos are played back with points and ellipse parameters plotted
over the video feeds and saved out as a combined aligned video with feeds stitched
together in a grid. Worldcam data are included in this, but if worldcam or any other
video is not provided, topdown (the only required video input) will be copied and printed
out in the place of any missing videos. Finally, the right or left ellipse parameters and
preened topdown points are saved into .nc files. Topdown is saved in a format of both
preened and preened with y-1200 values.

Before running, do the following:
- change savepath_input, the location where .nc and .avi outputs will be saved to
- change main_path and vid_path, the locations of dlc .h5 and camera .avi files to be read in, respectively
- change limit of loops, which cuts off the glob list of files at a given number of trials
- change figures parameter of preening and ellipse-finding functions at bottom of this file

Requires the functions in: topdown_preening.py
                           eye_tracking.py
                           time_management.py
                           data_reading.py
                           check_tracking.py
other files that are used by these above functions include:
                           data_cleaning.py
                           find.py
to preview camera feeds side-by-side, there is the file:
                           play_videos.py
to systematically rename files in a more readable format, there is the script:
                           corral_files.py
and data can be reopened and previewed after being saved out from this script with
                           read_nc_files.py

Adapted from GitHub repository /niell-lab-analysis/freely moving/loadAllCsv.m

TO DO:
- transform worldcam with eye direction (need a usable set of data for this)
- start using find() from find.py instead of glob()
- change figure=True/False parameter to two seperate ones: showfig=True to print it out in a window and makefig=True to
save it out without running the line plt.show()
- get saving out point and ellipse data working--currently does not work because of dask-related issue

last modified: June 18, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################
from glob import glob
import os.path
import numpy as np
import xarray as xr
import pandas as pd

from utilities.topdown_preening import preen_topdown_data
from utilities.eye_tracking import eye_angles
from utilities.time_management import read_time
from utilities.data_reading import read_data
from utilities.check_tracking import parse_data_for_playback

####################################
##          USER INPUTS           ##
####################################
# set savepath for ellipses, points, and videos
savepath_input = '/home/dylan/data/Niell/PreyCapture/Cohort3Outputs/J463c(blue)_110719/analysis_test_U09/'

# save out multiple figures for each trial into folders with the trial's name
savefig_input = True
# display multiple figures in separate window for every trial
# it can only show figures that are going to be saved
showfig_input = False

# find list of all data for DLC points as .h5 files and videos as .avi files
main_path = '/home/dylan/data/Niell/PreyCapture/Cohort3/J463c(blue)/110719/CorralledApproachData/'
vid_path = '/home/dylan/data/Niell/PreyCapture/Cohort3/J463c(blue)/110719/CorralledApproachVids/'

####################################
##        END USER INPUTS         ##
####################################

# find the files wanted from the given main_path
# DeepLabCut point locations
topdown_file_list = set(glob(main_path + '*top*DeepCut*.h5')) - set(glob(main_path + '*DeInter*.h5'))
righteye_file_list = set(glob(main_path + '*eye1r*DeepCut*.h5')) - set(glob(main_path + '*DeInter*.h5'))
lefteye_file_list = set(glob(main_path + '*eye2l*DeepCut*.h5')) - set(glob(main_path + '*DeInter*.h5'))
# video files that those points come from
righteye_vid_list = set(glob(vid_path + '*eye1r*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
lefteye_vid_list = set(glob(vid_path + '*eye2l*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
topdown_vid_list = set(glob(vid_path + '*top*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
worldcam_vid_list = set(glob(vid_path + '*world*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
# accelerometer files
acc_file_list = glob(main_path + '*acc*.dat')
# camera time files
righteye_time_file_list = glob(main_path + '*eye1r*TS*.csv')
lefteye_time_file_list = glob(main_path + '*eye2l*TS*.csv')
topdown_time_file_list = glob(main_path + '*topTS*.csv')

# for testing purposes, limit number of topdown files read in. error will be raised if limit_of_loops is not 2 or
# greater; align_head_from_DLC wants to index through trials and it can only do that if there is more than one
limit_of_loops = 2

# loop through each topdown file and find the associated files
# then, read the data in for each set, and build from it an xarray DataArray
loop_count = 0

trial_id_list = []

topdown_file_list = [i for i in topdown_file_list if '1_110719_01' not in i]

for file in topdown_file_list:
    if loop_count < limit_of_loops:
        split_path = os.path.split(file)
        file_name = split_path[1]
        mouse_key = file_name[0:5]
        # trial_key was [17:27], that only caught the first of two previously possible ending digits
        # now set to end at 28 because of corralled data
        trial_key = file_name[17:28]

        # get a the other recorded files that are associated with the topdown file currently being read
        # find the accelerometer file (these aren't yet used)
        acc_files = [i for i in acc_file_list if mouse_key and trial_key in i]
        # find the right/left eye DLC files
        righteye_files = [i for i in righteye_file_list if mouse_key and trial_key in i]
        lefteye_files = [i for i in lefteye_file_list if mouse_key and trial_key in i]
        # find the camera time files
        topdown_time_files = [i for i in topdown_time_file_list if mouse_key and trial_key in i]
        lefteye_time_files = [i for i in lefteye_time_file_list if mouse_key and trial_key in i]
        righteye_time_files = [i for i in righteye_time_file_list if mouse_key and trial_key in i]

        # this turns it into a string from a list of only one string item
        lefteye_file = lefteye_files[0]
        righteye_file = righteye_files[0]
        topdown_time_file = topdown_time_files[0]
        lefteye_time_file = lefteye_time_files[0]
        righteye_time_file = righteye_time_files[0]

        # read in the data from file locations
        topdown_pts, topdown_names, lefteye_pts, lefteye_names, righteye_pts, righteye_names = read_data(file, lefteye_file, righteye_file)

        # make a unique name for the mouse and the recording trial
        trial_id = 'mouse_' + str(mouse_key) + '_trial_' + str(trial_key)
        trial_id_list.append(trial_id)

        # read in the time stamp data of each camera for this trial
        # also: extrapolate to what the last timepoint shoudl be, since time files always have one fewer length than point data
        # TO DO: move the timestep correction into time_management.py
        if topdown_time_file is not None:
            topdown_time, topdown_start, topdown_end = read_time(topdown_time_file)
        elif topdown_time_file is None:
            topdown_time = None

        if lefteye_time_file is not None:
            lefteye_time, lefteye_start, lefteye_end = read_time(lefteye_time_file)
        elif lefteye_time_file is None:
            lefteye_time = None

        if righteye_time_file is not None:
            righteye_time, righteye_start, righteye_end = read_time(righteye_time_file)
        elif righteye_time_file is None:
            righteye_time = None

        print('building xarrays for trial ' + trial_id)
        # build one DataArray that stacks up all topdown trials in separate dimensions
        if topdown_time is not None:
            if loop_count == 0:
                topdown = xr.DataArray(topdown_pts, coords=[topdown_time, topdown_pts.columns], dims=['time', 'point_loc'])
                topdown['trial'] = trial_id
                topdown['time_start'] = topdown_start
                topdown['time_end'] = topdown_end
            elif loop_count > 0:
                topdown_trial = xr.DataArray(topdown_pts, coords=[topdown_time, topdown_pts.columns], dims=['time', 'point_loc'])
                topdown_trial['trial'] = trial_id
                topdown_trial['time_start'] = topdown_start
                topdown_trial['time_end'] = topdown_end
                topdown = xr.concat([topdown, topdown_trial], dim='trial', fill_value=np.nan)
        elif topdown_time is None:
            print('trial ' + trial_id + ' has no topdown time data')

        # build one DataArray that stacks up all trials in separate dimensions for each of two possible eyes
        if lefteye_pts is not None and lefteye_time is not None:
            if loop_count == 0:
                lefteye = xr.DataArray(lefteye_pts, coords=[lefteye_time, lefteye_pts.columns], dims=['time', 'point_loc'])
                lefteye['trial'] = trial_id
                lefteye['time_start'] = lefteye_start
                lefteye['time_end'] = lefteye_end
                # create a DaraFrame of all trial's timestamps for the left and right eye so that they can be added
                # back to the theta, phi, etc. ellipse parameter DataArrays
                lefteye_time_df = pd.DataFrame(lefteye_time,columns=[trial_id])
            elif loop_count > 0:
                lefteye_trial = xr.DataArray(lefteye_pts, coords=[lefteye_time, lefteye_pts.columns], dims=['time', 'point_loc'])
                lefteye_trial['trial'] = trial_id
                lefteye_trial['time_start'] = lefteye_start
                lefteye_trial['time_end'] = lefteye_end
                lefteye = xr.concat([lefteye, lefteye_trial], dim='trial', fill_value=np.nan)
                lefteye_time_df_to_append = pd.DataFrame(lefteye_time,columns=[trial_id])
                lefteye_time_df = lefteye_time_df.join(lefteye_time_df_to_append)
        elif lefteye_pts is None or lefteye_time is None:
            print('trial ' + trial_id + ' has no left eye camera data')

        if righteye_pts is not None and righteye_time is not None:
            if loop_count == 0:
                righteye = xr.DataArray(righteye_pts, coords=[righteye_time, righteye_pts.columns], dims=['time', 'point_loc'])
                righteye['trial'] = trial_id
                righteye['time_start'] = righteye_start
                righteye['time_end'] = righteye_end
                righteye_time_df = pd.DataFrame(righteye_time,columns=[trial_id])
            elif loop_count > 0:
                righteye_trial = xr.DataArray(righteye_pts, coords=[righteye_time, righteye_pts.columns], dims=['time', 'point_loc'])
                righteye_trial['trial'] = trial_id
                righteye_trial['time_start'] = righteye_start
                righteye_trial['time_end'] = righteye_end
                righteye = xr.concat([righteye, righteye_trial], dim='trial', fill_value=np.nan)
                righteye_time_df_to_append = pd.DataFrame(righteye_time,columns=[trial_id])
                righteye_time_df = righteye_time_df.join(righteye_time_df_to_append)
        elif righteye_pts is None or righteye_time is None:
            print('trial ' + trial_id + ' has no right eye camera data')

        loop_count = loop_count + 1

# run through each topdown trial, correct y-coordinates, and threshold point liklihoods
print('preening top-down points')
preened_topdown = preen_topdown_data(topdown, trial_id_list, topdown_names, savepath_input, showfig=showfig_input, savefig=savefig_input, coord_correction_val=0)
# preened_topdown_y1200 = preen_topdown_data(topdown, trial_id_list, topdown_names, savepath_input, showfig=showfig_input, savefig=savefig_input, coord_correction_val=1200)

# print('getting left eye angles')
left_ellipse = eye_angles(lefteye, lefteye_names, trial_id_list, savepath_input, lefteye_time_df, showfig=showfig_input, savefig=savefig_input, side='left')
print('getting right eye angles')
right_ellipse = eye_angles(righteye, righteye_names, trial_id_list, savepath_input, righteye_time_df, showfig=showfig_input, savefig=savefig_input, side='right')

# save out the xarrays as .nc files
# print('saving out xarray data')
# preened_topdown.to_netcdf(savepath_input + 'all_topdown_positions.nc')
# preened_topdown_y1200.to_netcdf(savepath_input + 'all_topdown_positions_yminus1200.nc')
# left_ellipse.to_netcdf(savepath_input + 'all_leftellipse_params.nc')
# right_ellipse.to_netcdf(savepath_input + 'all_rightellipse_params.nc')

# playback the videos and save out a combined alinged video with feeds stitched side-by-side
print('parsing data and video files for plotting and playback')
parse_data_for_playback(savepath_input, trial_id_list, preened_topdown, left_ellipse, right_ellipse,
                        topdown_vid_list, lefteye_vid_list, righteye_vid_list, worldcam_vid_list,
                        topdown_names, lefteye_names, righteye_names, left_pts=lefteye, right_pts=righteye)

