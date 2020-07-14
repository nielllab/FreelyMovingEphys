"""
FreelyMovingEphys terminal-facing DeepLabCut data intake
dlc_intake.py

Terminal-facing script to reach dlc- and video-handling functions

last modified: July 14, 2020
"""

# package imports
from glob import glob
import os.path
import numpy as np
import xarray as xr
import pandas as pd
import argparse
import warnings

# module imports
from util.read_data import find_paths, read_paths
from util.track_topdown import topdown_tracking #, head_angle
from util.track_eye import eye_tracking #, check_eye_calibration
from util.plot_video import check_tracking
from util.save_data import savecomplete

# get user inputs
parser = argparse.ArgumentParser(description='Process DeepLabCut data and corresponding videos.', epilog='The global data path may include between one and three topdown views, between zero and two eye views, and between zero and two world views. Timestamp files should be provided in .csv format, DLC points in .h5 format, and videos in .avi format. Saved outputs will include one .nc file with all point, ellipse, and angle data for all trials and one .nc file with all starting DLC data for all trials. Additionally, a folder will be created for each trial and videos with points and/or ellipse paramaters plotted over camera video feeds will be saved out as .avi formats for each view input, and saftey-check plots will be saved as .png formats showing how well DLC and the intake pipeline have done.')
parser.add_argument('global_data_path', help='source for .avi videos, DLC .h5 files, and .csv timestamp files')
parser.add_argument('global_save_path', help='path into which outputs will be saved')
parser.add_argument('-lt', '--lik_thresh', help='DLC likelihood threshold, default=0.99', default=0.99, type=int)
parser.add_argument('-cc', '--coord_cor', help='value with which to correct y-coordinates in topdown view, default=0', default=0, type=int)
parser.add_argument('-tp', '--topdown_pt_num', help='number of labeled topdown DeepLabCut points (including points on cricket, if there is one) default=10', default=10, type=int)
parser.add_argument('-ck', '--cricket', help='is there a cricket in the tank as the last two DeepLabCut points? default=True', default=True, type=bool)
parser.add_argument('-pt', '--pxl_thresh', help='max number of pixels for radius of pupil, default=50', default=50, type=int)
parser.add_argument('-et', '--ell_thresh', help='maximum ratio of ellipse shortaxis to longaxis, default=0.90', default=0.90, type=int)
parser.add_argument('-ep', '--eye_pt_num', help='number of labeled eye camera points (including tear duct and back of eye, if those exist); default=8', default=8, type=int)
parser.add_argument('-tr', '--tear', help='are there eye points labeled for the tear duct and back of the eye? default=False', default=False, type=bool)
parser.add_argument('-sv', '--save_vids', help='should videos be plotted upon and saved out? default=True', default=True, type=bool)
parser.add_argument('-bt', '--bonsaiTS', help='should bonsai timestamps be used? If False, Flir timestamps will be used. default=True (Bonsai)', default=True, type=bool)
args = parser.parse_args()

print('getting DLC, video, and  time files')
# find DLC files in global_data_path
dlc_glob_keys = ['*TOP1*.h5', '*TOP2*.h5', '*TOP3*.h5', '*REye*.h5', '*LEye*.h5']
dlc_paths = find_paths(args.global_data_path, dlc_glob_keys)
topdown1_dlc_files = dlc_paths[0]; topdown2_dlc_files = dlc_paths[1]; topdown3_dlc_files = dlc_paths[2]
righteye_dlc_files = dlc_paths[3]; lefteye_dlc_files = dlc_paths[4]

# find video files in global_data_path
vid_glob_keys = ['*TOP1*.avi', '*TOP2*.avi', '*TOP3*.avi', '*REye*.avi', '*LEye*.avi', '*RWorld*.avi', '*LWorld*.avi']
vid_paths = find_paths(args.global_data_path, vid_glob_keys)
topdown1_vid_files = vid_paths[0]; topdown2_vid_files = vid_paths[1]; topdown3_vid_files = vid_paths[2]
righteye_vid_files = vid_paths[3]; lefteye_vid_files = vid_paths[4]
rightworld_vid_files = vid_paths[5]; leftworld_vid_files = vid_paths[6]

# find time files in global_data_path
time_glob_keys = ['*TOP1*BonsaiTS.csv', '*TOP2*BonsaiTS.csv', '*TOP3*BonsaiTS.csv', '*REye*BonsaiTS.csv', '*LEye*BonsaiTS.csv', '*RWorld*BonsaiTS.csv', '*LWorld*BonsaiTS.csv']
time_paths = find_paths(args.global_data_path, time_glob_keys)
topdown1_time_files = time_paths[0]; topdown2_time_files = time_paths[1]; topdown3_time_files = time_paths[2]
righteye_time_files = time_paths[3]; lefteye_time_files = time_paths[4]
rightworld_time_files = time_paths[5]; leftworld_time_files = time_paths[6]

for top1dlcpath in topdown1_dlc_files:
    top1fullname = os.path.split(top1dlcpath)[1]
    key_pieces = top1fullname.split('_')[:-1]
    key = '_'.join(key_pieces)

    print('starting on ' + str(key))

    # in the case where there's only one DLC file for each camera type (i.e. one trial only), run this...
    if len(topdown1_dlc_files) == 1:
        top2dlcpath = topdown2_dlc_files; top3dlcpath = topdown3_dlc_files
        eyeLdlcpath = lefteye_dlc_files; eyeRdlcpath = righteye_dlc_files
        top1vidpath = topdown1_vid_files; top2vidpath = topdown2_vid_files; top3vidpath = topdown3_vid_files
        eyeLvidpath = lefteye_vid_files; eyeRvidpath = righteye_vid_files
        worldLvidpath = leftworld_vid_files; worldRvidpath = rightworld_vid_files
        top1timepath = topdown1_time_files; top2timepath = topdown2_time_files; top3timepath = topdown3_time_files
        eyeLtimepath = lefteye_time_files; eyeRtimepath = righteye_time_files
        worldLtimepath = leftworld_time_files; worldRtimepath = rightworld_time_files

    elif len(topdown1_dlc_files) > 1:
        # get associated DLC files for top1dlcpath
        top2dlcpath = [i for i in topdown2_dlc_files if key in i]
        top3dlcpath = [i for i in topdown3_dlc_files if key in i]
        eyeLdlcpath = [i for i in lefteye_dlc_files if key in i]
        eyeRdlcpath = [i for i in righteye_dlc_files if key in i]

        # get associated video files for top1dlcpath
        top1vidpath = [i for i in topdown1_vid_files if key in i]
        top2vidpath = [i for i in topdown2_vid_files if key in i]
        top3vidpath = [i for i in topdown3_vid_files if key in i]
        eyeLvidpath = [i for i in lefteye_vid_files if key in i]
        eyeRvidpath = [i for i in righteye_vid_files if key in i]
        worldLvidpath = [i for i in leftworld_vid_files if key in i]
        worldRvidpath = [i for i in rightworld_vid_files if key in i]

        # get associated time files for top1dlcpath
        top1timepath = [i for i in topdown1_time_files if key in i]
        top2timepath = [i for i in topdown2_time_files if key in i]
        top3timepath = [i for i in topdown3_time_files if key in i]
        eyeLtimepath = [i for i in lefteye_time_files if key in i]
        eyeRtimepath = [i for i in righteye_time_files if key in i]
        worldLtimepath = [i for i in leftworld_time_files if key in i]
        worldRtimepath = [i for i in rightworld_time_files if key in i]

        print(top1timepath)

    # build xarray DataArrays for each camera type which contain data for each view of that type
    # also make DataArray from the timestamp list for each view of the camera type
    topdlc, toptime, topnames = read_paths(top1dlcpath, top1timepath, top2dlcpath, top2timepath, top3dlcpath, top3timepath)
    eyedlc, eyetime, eyenames = read_paths(eyeLdlcpath, eyeLtimepath, eyeRdlcpath, eyeRtimepath)

    topdlc['trial'] = key
    toptime['trial'] = key
    eyedlc['trial'] = key
    eyetime['trial'] = key

    # topdown tracking, plotting, and data saving
    top_vlist = ['v1', 'v2', 'v3']
    for v in top_vlist:
        try:
            print('trying to track topdown camera view ' + str(v) + ' for ' + str(key))
            vpts = topdlc[v]
            if v == 'v1':
                vid = top1vidpath
                viewext = 'TOP1'
            elif v == 'v2':
                vid = top2vidpath
                viewext = 'TOP2'
            elif v == 'v3':
                vid = top3vidpath
                viewext = 'TOP3'
            vcleanpts = topdown_tracking(vpts, topnames, args.global_save_path, key, args.lik_thresh, args.coord_cor, args.topdown_pt_num, args.cricket)
            # vtheta = head_angle(vcleanpts, topnames, args.lik_thresh)
            if isinstance(vid, list):
                check_tracking(key, 't', vid[0], args.global_save_path, dlc_data=vcleanpts, vext=viewext) #, head_ang=vtheta)
            else:
                check_tracking(key, 't', vid, args.global_save_path, dlc_data=vcleanpts, vext=viewext) #, head_ang=vtheta)
            vpts.name = 'raw_pt_values'
            vcleanpts.name = 'output_pt_values'
            if v == 'v1':
                gatheredtop = xr.merge([vpts, vcleanpts])
            elif v != 'v1':
                concattop = xr.merge([vpts, vcleanpts])
                gatheredtop = xr.concat([gatheredtop, concattop], dim='view', fill_value=np.nan)
            print('tracking sucessful for ' + str(v))
        except KeyError: # in case not all three views exist
            print('failed to find view ' + str(v))
            pass
    if top1dlcpath == topdown1_dlc_files[0]:
        gatheredtop['trial'] = key
        topout = gatheredtop
    elif top1dlcpath != topdown1_dlc_files[0]:
        gatheredtop['trial'] = key
        topout = xr.concat([topout, gatheredtop], dim='trial', fill_value=np.nan)

    eye_vlist = ['v1', 'v2'] # v1 is always the left eye, and v2 is the right eye
    for v in eye_vlist:
        try:
            print('tracking eye camera view ' + str(v) + ' for ' + str(key))
            vpts = eyedlc[v]
            if v == 'v1':
                vid = eyeLvidpath
                viewext = 'LEye'
            elif v == 'v2':
                vid = eyeRvidpath
                viewext = 'REye'
            vparams = eye_tracking(vpts, eyenames, args.global_save_path, key, args.lik_thresh, args.pxl_thresh, args.eye_pt_num, args.tear)
            # check_eye_calibration(vparams, vpts, args.global_save_path, key, args.ell_thresh)
            if isinstance(vid, list):
                check_tracking(key, 'e', vid[0], args.global_save_path, dlc_data=vpts, ell_data=vparams, vext=viewext)
            else:
                check_tracking(key, 'e', vid, args.global_save_path, dlc_data=vpts, ell_data=vparams, vext=viewext)
            vpts.name = 'raw_pt_values'
            vparams.name = 'ellipse_param_values'
            if v == 'v1':
                gatheredeye = xr.merge([vpts, vparams])
            elif v != 'v1':
                concateye = xr.merge([vpts, vparams])
                gatheredeye = xr.concat([gatheredeye, concateye])
            print('tracking sucessful for ' + str(v))
        except KeyError:
            print('failed to find view ' + str(v))
            pass
    if top1dlcpath == topdown1_dlc_files[0]:
        gatheredeye['trial'] = key
        eyeout = gatheredeye
    elif top1dlcpath != topdown1_dlc_files[0]:
        gatheredeye['trial'] = key
        eyeout = xr.concat([topout, gatheredeye], dim='trial', fill_value=np.nan)

    print('processing of trial ' + key + ' is complete... outputs wont be saved until the end of all trials')

try:
    savecomplete(topout, args.global_save_path, 'tops')
except NameError:
    print('no top .nc file saved because no data was passed')
try:
    savecomplete(eyeout, args.global_save_path, 'eyes')
except NameError:
    print('no eye .nc file saved because no data was passed')
