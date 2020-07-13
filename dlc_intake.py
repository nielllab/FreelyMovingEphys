"""
FreelyMovingEphys terminal-facing DeepLabCut data intake
dlc_intake.py

Terminal-facing script to reach dlc- and video-handling functions in repository's directory /util/

last modified: July 12, 2020
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
from util.track_topdown import topdown_tracking, head_angle
from util.track_eye import eye_tracking, check_eye_calibration
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
dlc_glob_keys = ['*Top1*.h5', '*Top2*.h5', '*Top3*.h5', '*REye*.h5', '*LEye*.h5']
dlc_paths = find_paths(args.global_data_path, dlc_glob_keys)
topdown1_dlc_files = dlc_paths[0]; topdown2_dlc_files = dlc_paths[1]; topdown3_dlc_files = dlc_paths[2]
righteye_dlc_files = dlc_paths[3]; lefteye_dlc_files = dlc_paths[4]

# find video files in global_data_path
vid_glob_keys = ['*Top1*.avi', '*Top2*.avi', '*Top3*.avi', '*REye*.avi', '*LEye*.avi', '*RWorld*.avi', '*LWorld*.avi']
vid_paths = find_paths(args.global_data_path, vid_glob_keys)
topdown1_vid_files = vid_paths[0]; topdown2_vid_files = vid_paths[1]; topdown3_vid_files = vid_paths[2]
righteye_vid_files = vid_paths[3]; lefteye_vid_files = vid_paths[4]
rightworld_vid_files = vid_paths[5]; leftworld_vid_files = vid_paths[6]

# find time files in global_data_path
time_glob_keys = ['*Top1*BonsaiTS.csv', '*Top2*BonsaiTS.csv', '*Top3*BonsaiTS.csv', '*REye*BonsaiTS.csv', '*LEye*BonsaiTS.csv', '*RWorld*BonsaiTS.csv', '*LWorld*BonsaiTS.csv']
time_paths = find_paths(args.global_data_path, time_glob_keys)
topdown1_time_files = time_paths[0]; topdown2_time_files = time_paths[1]; topdown3_time_files = time_paths[2]
righteye_time_files = time_paths[3]; lefteye_time_files = time_paths[4]
rightworld_time_files = time_paths[5]; leftworld_time_files = time_paths[6]

for top1dlcpath in topdown1_dlc_files:
    top1fullname = os.path.split(top1dlcpath)[1]
    key = top1fullname.split('_')[:-1]
    print('starting on ' + str(key))

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

    # build xarray DataArrays for each camera type which contain data for each view of that type
    # also make DataArray from the  timestamp list for each view of the camera type
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
            print('tracking topdown camera view ' + str(v) + ' for ' + str(key))
            vpts = topdlc.sel(view=v)
            vcleanpts = topdown_tracking(vpts, topnames, args.savepath, key, args.lik_thresh, args.coord_cor, args.topdown_pt_num, args.cricket)
            vtheta = head_angle(vcleanpts, topnames, args.lik_thresh)
            check_tracking(key, 't', top1vidpath, args.savepath, dlc_data=vcleanpts, head_ang=vtheta)
            if v == 'v1':
                gatheredtop = xr.merge([vpts, vcleanpts, vtheta])
            elif v != 'v1':
                concattop = xr.merge([vpts, vcleanpts, vtheta])
                gatheredtop = xr.concat([gatheredtop, concattop])
        except IndexError:
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
            vpts = eyedlc.sel(view=v)
            vparams = eye_tracking(vpts, eyenames, args.global_save_path, key, args.lik_thresh, args.pxl_thresh, args.eye_pt_num, args.tear)
            check_eye_calibration(vparams, vpts, args.global_save_path, key, args.ell_thresh)
            check_tracking(key, 't', top1vidpath, args.savepath, dlc_data=vpts, ell_data=vparams)
            if v == 'v1':
                gatheredeye = xr.merge([vpts, vcleanpts, vtheta])
            elif v != 'v1':
                concattop = xr.merge([vpts, vcleanpts, vtheta])
                gatheredeye = xr.concat([gatheredeye, concattop])
        except IndexError:
            pass
    if top1dlcpath == topdown1_dlc_files[0]:
        gatheredeye['trial'] = key
        eyeout = gatheredeye
    elif top1dlcpath != topdown1_dlc_files[0]:
        gatheredeye['trial'] = key
        eyeout = xr.concat([topout, gatheredeye], dim='trial', fill_value=np.nan)

savecomplete(topout, args.global_save_path, 'tops')
savecomplete(eyeout, args.global_save_path, 'eyes')



