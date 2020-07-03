#####################################################################################
"""
load_from_DLC.py

NOTES ON USE
Through terminal, the user provides (1) a folder path for DeepLabCut .h5 files and
camera timestamp .csv files, (2) a folder path for .avi videos from camera feeds, (3)
a folder path into which outputs of load_from_DLC.py can be saved including plots,
videos, and point position and eye ellipse data, (4) whether or not to save figures
into the save path with the argument in y/n format, and (5) whether or not to save out
point and ellipse data as .nc files with the argument in bool y/n format.
Optional arguments include: -ll is how many trial's to add into the cohort before preening
the data and finding ellipses, etc.; -lt is the likelihood threshold to use for DeepLabCut found
point positions; -pt is the maximum number of pixels for radius of the pupil for eye tracking;
-cc is the values with which topdown view y-coordinates can be corrected.

Example terminal line to run the script:
python3 load_from_DLC.py /dlc/h5/file/path/ /video/avi/file/path/ /path/to/save/data/and/videos/ y y -ll 2

NOTES ON FUNCTIONALITY
DeepLabCut (DLC) outputs are read into xarray DataArray format for the topdown camera, and
the camera for each eye side. It runs just fine if one or both eyes are missing for a
trial. Time is kept in nanoseconds since the first timestamp for a given trial. The
topdown camera data are thresholded by DLC likelihood values and (optionally) corrected by
a coordinate value in topdown_preening.py. Eye points are passed to eye_tracking.py to
threshold by likelihood and get out ellipse parameters, which are returned also in an
xarray DataArray format. Once these DataArrays are all created, they're passed to
check_all_tracking.py which plots the DLC points and ellipse parameters on the .avi videos,
and saves out the videos in .mp4 format. Data are stored in xarray DataArrays during
use, and saved out as a .nc file right after being converted to one all-encompassing
Dataset which contains all points and ellipse parameters for all trials.

last modified: July 03, 2020
"""
#####################################################################################
# package imports
from glob import glob
import os.path
import numpy as np
import xarray as xr
import pandas as pd
import argparse
import warnings

# function imports
from utilities.topdown_preening import preen_topdown_data
from utilities.eye_tracking import eye_angles
from utilities.time_management import read_time
from utilities.data_reading import read_data
from utilities.check_all_tracking import parse_data_for_playback
from utilities.eye_calibration import plot_check_eye_calibration
from utilities.check_ind_tracking import plot_pts_on_vid

# get user inputs
parser = argparse.ArgumentParser(description='Process DeepLabCut data and corresponding videos.', epilog='The DeepLabCut data must include a set of topdown points, and can contain up to two eyes. The videos must include a topdown video, and may optionallly include one or both eyes and/or a worldcam. Timestamp files in .csv file format must be present in the DeepLabCut folder for every camera that has video files. If a timestamp file is missing, the corresponding camera cannot be read in.')
parser.add_argument('dlcpath', help='a file path where the dlc .h5 files can be found for the trial(s) of interest')
parser.add_argument('vidpath', help='a file path where corrosponding eye, topdown, and worldcam videos can be found as .avi files')
parser.add_argument('savepath', help='a file path into which figures, videos, and dlc points can be saved')
parser.add_argument('savefig', help='save out multiple figures and videos for every trial into save directory? (y/n)')
parser.add_argument('savenc', help='save out .nc file of all ddlc data for all trials in the cohort passed in? (y/n)')
parser.add_argument('-ll', '--looplim', help='number of unique trials to loop through, type=int, if nothing passed, looplim=100', type=int)
parser.add_argument('-lt', '--likthresh', help='number of unique trials to loop through, type=int, if nothing passed, likthresh=0.99', type=int)
parser.add_argument('-pt', '--pxlthresh', help='max number of pixels for radius of pupil, type=int, if nothing passed, pxlthresh=50', type=int)
parser.add_argument('-cc', '--coordcor', help='values with which to correct y-coordinates i topdown view, type=int, if nothing passed, coordcor=0', type=int)

args = parser.parse_args()

# create values from user inputs
if args.savefig == 'y':
    savefig = True
elif args.savefig == 'n':
    savefig = False

if args.savenc == 'y':
    savenc = True
elif args.savenc == 'n':
    savenc = False

if args.looplim:
    limit_of_loops = args.looplim
else:
    limit_of_loops = 100

if args.likthresh:
    likelihood_thresh  = args.likthresh
else:
    likelihood_thresh = 0.99

if args.pxlthresh:
    pixel_thresh = args.pxlthresh
else:
    pixel_thresh = 50

if args.coordcor:
    coordcor = args.coordcor
else:
    coordcor = 0

# find the files wanted from the given args.dlcpath and args.vidpath
# DeepLabCut point locations
topdown_file_list = glob(os.path.join(args.dlcpath, '*top*DeepCut*.h5'))
righteye_file_list = glob(os.path.join(args.dlcpath, '*eye1r*DeInter2*.h5'))
lefteye_file_list = glob(os.path.join(args.dlcpath, '*eye2l*DeInter2*.h5'))
# video files that those points come from
righteye_vid_list = glob(os.path.join(args.vidpath, '*eye1r*.avi'))
lefteye_vid_list = glob(os.path.join(args.vidpath, '*eye2l*.avi'))
topdown_vid_list = glob(os.path.join(args.vidpath, '*top*.avi'))
worldcam_vid_list = glob(os.path.join(args.vidpath, '*world*.avi'))
# camera time files
righteye_time_file_list = glob(os.path.join(args.dlcpath, '*eye1r*TS*.csv'))
lefteye_time_file_list = glob(os.path.join(args.dlcpath, '*eye2l*TS*.csv'))
topdown_time_file_list = glob(os.path.join(args.dlcpath, '*topTS*.csv'))

# exclude some of the sets of data that cause issues
topdown_file_list = [i for i in topdown_file_list if '1_110719_01' not in i]
topdown_file_list = [i for i in topdown_file_list if '2_110719_08' not in i]
topdown_file_list = [i for i in topdown_file_list if '1_110719_11' not in i]

# sort the files that are used to find all other files
topdown_file_list = sorted(topdown_file_list)

# loop through each topdown DLC point .h5 file name
loop_count = 0
trial_id_list = []
for file in topdown_file_list:
    if loop_count < limit_of_loops:
        # get unique sections of filename out so that they can be used to find the associated files
        split_path = os.path.split(file)
        file_name = split_path[1]
        mouse_key = file_name[0:5]
        trial_key = file_name[17:28]

        # find the right/left eye DLC files that match the topdown DLC file
        righteye_files = [i for i in righteye_file_list if mouse_key and trial_key in i]
        lefteye_files = [i for i in lefteye_file_list if mouse_key and trial_key in i]
        # find the camera time files that match the topdown DLC file
        topdown_time_files = [i for i in topdown_time_file_list if mouse_key and trial_key in i]
        lefteye_time_files = [i for i in lefteye_time_file_list if mouse_key and trial_key in i]
        righteye_time_files = [i for i in righteye_time_file_list if mouse_key and trial_key in i]

        # the above lines return lists of one string, this converts them into just a string
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
        if topdown_time_file is not None:
            topdown_time = read_time(topdown_time_file, len(topdown_pts))
        elif topdown_time_file is None:
            topdown_time = None

        if lefteye_time_file is not None:
            lefteye_time = read_time(lefteye_time_file, len(lefteye_pts))
        elif lefteye_time_file is None:
            lefteye_time = None

        if righteye_time_file is not None:
            righteye_time = read_time(righteye_time_file, len(righteye_pts))
        elif righteye_time_file is None:
            righteye_time = None

        print('building xarrays for trial ' + trial_id)
        # build one DataArray that stacks up all topdown trials in separate dimensions
        if topdown_time is not None:
            if loop_count == 0:
                topdown = xr.DataArray(topdown_pts, dims=['frame', 'point_loc'])
                topdown['trial'] = trial_id
                topdown_time_df = pd.DataFrame(topdown_time, columns=[trial_id])
            elif loop_count > 0:
                topdown_trial = xr.DataArray(topdown_pts, dims=['frame', 'point_loc'])
                topdown_trial['trial'] = trial_id
                topdown = xr.concat([topdown, topdown_trial], dim='trial', fill_value=np.nan)
                topdown_time_df_to_append = pd.DataFrame(topdown_time, columns=[trial_id])
                topdown_time_df = topdown_time_df.join(topdown_time_df_to_append)
        elif topdown_time is None:
            print('trial ' + trial_id + ' has no topdown time data')

        # build one DataArray that stacks up all trials in separate dimensions for each of two possible eyes
        if lefteye_pts is not None and lefteye_time is not None:
            if loop_count == 0:
                # create a DataArray of left eye point positions
                lefteye = xr.DataArray(lefteye_pts, dims=['frame', 'point_loc'])
                lefteye['trial'] = trial_id
                lefteye_time_df = pd.DataFrame(lefteye_time,columns=[trial_id])
            elif loop_count > 0:
                # point positions (concat new to full)
                lefteye_trial = xr.DataArray(lefteye_pts, dims=['frame', 'point_loc'])
                lefteye_trial['trial'] = trial_id
                lefteye = xr.concat([lefteye, lefteye_trial], dim='trial', fill_value=np.nan)
                lefteye_time_df_to_append = pd.DataFrame(lefteye_time,columns=[trial_id])
                lefteye_time_df = lefteye_time_df.join(lefteye_time_df_to_append)
        elif lefteye_pts is None or lefteye_time is None:
            print('trial ' + trial_id + ' has no left eye camera data')

        if righteye_pts is not None and righteye_time is not None:
            if loop_count == 0:
                righteye = xr.DataArray(righteye_pts, dims=['time', 'point_loc'])
                righteye['trial'] = trial_id
                righteye_time_df = pd.DataFrame(righteye_time,columns=[trial_id])
            elif loop_count > 0:
                righteye_trial = xr.DataArray(righteye_pts, dims=['time', 'point_loc'])
                righteye_trial['trial'] = trial_id
                righteye = xr.concat([righteye, righteye_trial], dim='trial', fill_value=np.nan)
                righteye_time_df_to_append = pd.DataFrame(righteye_time,columns=[trial_id])
                righteye_time_df = righteye_time_df.join(righteye_time_df_to_append)
        elif righteye_pts is None or righteye_time is None:
            print('trial ' + trial_id + ' has no right eye camera data')

        # turn time pandas objects into xarrays
        if topdown_time is not None:
            if loop_count == 0:
                all_topdownTS = xr.DataArray(topdown_time_df)
                all_topdownTS['trial'] = trial_id
            elif loop_count > 0:
                topdownTS = xr.DataArray(topdown_time_df)
                topdownTS['trial'] = trial_id
                all_topdownTS = xr.concat([all_topdownTS, topdownTS], dim='trial')

        if lefteye_time is not None:
            if loop_count == 0:
                all_lefteyeTS = xr.DataArray(lefteye_time_df)
                all_lefteyeTS['trial'] = trial_id
            elif loop_count > 0:
                lefteyeTS = xr.DataArray(lefteye_time_df)
                lefteyeTS['trial'] = trial_id
                all_lefteyeTS = xr.concat([all_lefteyeTS, lefteyeTS], dim='trial')

        if righteye_time is not None:
            if loop_count == 0:
                all_righteyeTS = xr.DataArray(righteye_time_df)
                all_righteyeTS['trial'] = trial_id
            elif loop_count > 0:
                righteyeTS = xr.DataArray(righteye_time_df)
                righteyeTS['trial'] = trial_id
                all_righteyeTS = xr.concat([all_righteyeTS, righteyeTS], dim='trial')

        loop_count = loop_count + 1

# process the topdown data
print('preening top-down points')
preened_topdown = preen_topdown_data(topdown, trial_id_list, topdown_names, args.savepath, savefig=savefig, coord_correction_val=coordcor, thresh=likelihood_thresh)
preened_topdown = xr.DataArray.rename(preened_topdown, 'topdown')

# get the ellipse parameters out from the DLC points of each eye
with warnings.catch_warnings():
    # ignore a reoccurring runtime error while running the ellipse parameter functions
    warnings.simplefilter("ignore")

    print('getting left eye angles')
    left_ellipse = eye_angles(lefteye, lefteye_names, trial_id_list, args.savepath, lefteye_time_df, savefig=savefig, side='left', pxl_thresh=pixel_thresh)
    left_ellipse = xr.DataArray.rename(left_ellipse, 'left_ellipse')
    print('getting right eye angles')
    right_ellipse = eye_angles(righteye, righteye_names, trial_id_list, args.savepath, righteye_time_df, savefig=savefig, side='right', pxl_thresh=pixel_thresh)
    right_ellipse = xr.DataArray.rename(right_ellipse, 'right_ellipse')

# confirm that the eye tracking has done an alright job
# print('checking calibration of eyes') # THIS DOES NOT WORK YET
# plot_check_eye_calibration(left_ellipse, lefteye, trial_id_list, 'left', args.savepath)
# plot_check_eye_calibration(right_ellipse, righteye, trial_id_list, 'right', args.savepath)

if savefig is True:
    for current_trial in trial_id_list:
        print('writing video data')
        # go through the trial key and pull out unique sections that appear in video names
        mouse_key = current_trial[6:11]
        trial_key = current_trial[18:]

        # get out the video associated with this trial for every camera viewpoint
        righteye_vids = [i for i in righteye_vid_list if mouse_key and trial_key in i]
        lefteye_vids = [i for i in lefteye_vid_list if mouse_key and trial_key in i]
        topdown_vids = [i for i in topdown_vid_list if mouse_key and trial_key in i]
        worldcam_vids = [i for i in worldcam_vid_list if mouse_key and trial_key in i]

        # plot the points on each camera view and save them out separately
        topdown_vid = topdown_vids[0]
        td_pt_data = preened_topdown.sel(trial=current_trial)
        plot_pts_on_vid(current_trial, 't', topdown_vid, args.savepath, td_pt_data)

        try:
            righteye_vid = righteye_vids[0]
            re_pt_data = righteye.sel(trial=current_trial)
            re_pt_ell = right_ellipse.sel(trial=current_trial)
            plot_pts_on_vid(current_trial, 'r', righteye_vid, args.savepath, re_pt_data, re_pt_ell)
        except IndexError:
            pass

        try:
            worldcam_vid = worldcam_vids[0]
            plot_pts_on_vid(current_trial, 'w', worldcam_vid, args.savepath)
        except IndexError:
            pass

        try:
            lefteye_vid = lefteye_vids[0]
            le_pt_data = lefteye.sel(trial=current_trial)
            le_pt_ell = left_ellipse.sel(trial=current_trial)
            plot_pts_on_vid(current_trial, 'l', lefteye_vid, args.savepath, le_pt_data, le_pt_ell)
        except IndexError:
            pass

# save out the xarrays as .nc files
if savenc is True:
    print('saving out xarray data')
    ds_topdown = preened_topdown.to_dataset(name='topdown')
    ds_leftellipse = left_ellipse.to_dataset(name='left_ellipse')
    ds_rightellipse = right_ellipse.to_dataset(name='right_ellipse')

    ds_leftellipse = ds_leftellipse.assign({'cam_center_x': ds_leftellipse['cam_center_x'].values})
    ds_leftellipse = ds_leftellipse.assign({'cam_center_y': ds_leftellipse['cam_center_y'].values})
    ds_rightellipse = ds_rightellipse.assign({'cam_center_x': ds_rightellipse['cam_center_x'].values})
    ds_rightellipse = ds_rightellipse.assign({'cam_center_y': ds_rightellipse['cam_center_y'].values})
    ds_leftellipse = xr.Dataset.drop_vars(ds_leftellipse, names=['eye_side', 'cam_center_x', 'cam_center_y'])
    ds_rightellipse = xr.Dataset.drop_vars(ds_rightellipse, names=['eye_side', 'cam_center_x', 'cam_center_y'])

    gathered = xr.merge([ds_topdown, ds_leftellipse, ds_rightellipse])
    gathered_path = args.savepath + 'params.nc'
    gathered.to_netcdf(gathered_path)

    ds_topdown_raw = topdown.to_dataset(name='topdown')
    ds_lefteye_raw = lefteye.to_dataset(name='lefteye')
    ds_righteye_raw = righteye.to_dataset(name='righteye')

    gathered_raw = xr.merge([ds_topdown_raw, ds_lefteye_raw, ds_righteye_raw])
    gathered_raw_path = args.savepath + 'raw_points.nc'
    gathered_raw.to_netcdf(gathered_raw_path)

    ds_topdown_time = all_topdownTS.to_dataset(name='topdown_time')
    ds_lefteye_time = all_lefteyeTS.to_dataset(name='lefteye_time')
    ds_righteye_time = all_righteyeTS.to_dataset(name='righteye_time')

    gathered_time = xr.merge([ds_topdown_time, ds_lefteye_time, ds_righteye_time])
    gathered_time_path = args.savepath + 'timestamps.nc'
    gathered_time.to_netcdf(gathered_time_path)

