"""
FreelyMovingEphys terminal-facing DeepLabCut data intake
dlc_intake.py

last modified: August 24, 2020
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
from util.track_topdown import topdown_tracking , head_angle, check_topdown_tracking
from util.track_eye import eye_tracking, check_eye_tracking
from util.save_data import savecomplete
from util.track_world import adjust_world, find_pupil_rotation
from util.track_cricket import get_cricket_props
from util.analyze_jump import jump_gaze_trace

# get user inputs
parser = argparse.ArgumentParser(description='Process DeepLabCut data and corresponding videos.', epilog='The global data path may include between zero and three topdown views, between one and two eye views, and between zero and two world views. Timestamp files should be provided in .csv format, DLC points in .h5 format, and videos in .avi format. Saved outputs will include one .nc file with all point, ellipse, and angle data for all trials and one .nc file with all starting DLC data for all trials. Additionally, a folder will be created for each trial and videos with points and/or ellipse paramaters plotted over camera video feeds will be saved out as .avi formats for each view input, and saftey-check plots will be saved as .png formats showing how well DLC and the intake pipeline have done. A right eye .avi video is necessary to find all other files.')
parser.add_argument('global_data_path', help='source for .avi videos, DLC .h5 files, and .csv timestamp files')
parser.add_argument('global_save_path', help='path into which outputs will be saved')
parser.add_argument('experiment_type', help='type of experiment to run, options are freelymoving or jumping')
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

print('starting on data set of type=' + str(args.experiment_type))
if args.experiment_type == 'freelymoving':
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

    for eyeRvidpath in righteye_vid_files:
        eyeRfullname = os.path.split(eyeRvidpath)[1]
        key_pieces = eyeRfullname.split('_')[:-1]
        key = '_'.join(key_pieces)

        print('starting on ' + str(key))

        # in the case where there's only one DLC file for each camera type (i.e. one trial only), run this...
        if len(righteye_vid_files) == 1:
            top1dlcpath = topdown1_dlc_files; top2dlcpath = topdown2_dlc_files; top3dlcpath = topdown3_dlc_files
            eyeLdlcpath = lefteye_dlc_files; eyeRdlcpath = righteye_dlc_files
            top1vidpath = topdown1_vid_files; top2vidpath = topdown2_vid_files; top3vidpath = topdown3_vid_files
            eyeLvidpath = lefteye_vid_files;
            worldLvidpath = leftworld_vid_files; worldRvidpath = rightworld_vid_files
            top1timepath = topdown1_time_files; top2timepath = topdown2_time_files; top3timepath = topdown3_time_files
            eyeLtimepath = lefteye_time_files; eyeRtimepath = righteye_time_files
            worldLtimepath = leftworld_time_files; worldRtimepath = rightworld_time_files

        elif len(righteye_vid_files) > 1:
            # get associated DLC files for top1dlcpath
            top1dlcpath = [i for i in topdown1_dlc_files if key in i]
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
        # also make DataArray from the timestamp list for each view of the camera type
        topdlc, toptime, topnames = read_paths(top1dlcpath, top1timepath, top2dlcpath, top2timepath, top3dlcpath, top3timepath)
        eyedlc, eyetime, eyenames = read_paths(eyeRdlcpath, eyeRtimepath, eyeLdlcpath, eyeLtimepath)

        # TOPDOWN VIEW PROCESSING
        if topdlc is not None:
            topdlc['trial'] = key; toptime['trial'] = key
            top_vlist = ['v1', 'v2', 'v3']
            for v in top_vlist:
                try:
                    print('trying to track topdown camera view ' + str(v) + ' for ' + str(key))
                    vpts = topdlc[v]
                    if v == 'v1':
                        vid = top1vidpath
                        time = top1timepath
                        viewext = 'TOP1'
                    elif v == 'v2':
                        vid = top2vidpath
                        time = top2timepath
                        viewext = 'TOP2'
                    elif v == 'v3':
                        vid = top3vidpath
                        time = top3timepath
                        viewext = 'TOP3'
                    vcleanpts, nose_x, nose_y = topdown_tracking(vpts, topnames, args.global_save_path, key, args.lik_thresh, args.coord_cor, args.topdown_pt_num, args.cricket, time[0])
                    vthetas = head_angle(vcleanpts, topnames, args.lik_thresh, args.global_save_path, args.cricket, key, nose_x, nose_y, time)
                    if args.cricket is True:
                        cricket_props = get_cricket_props(vcleanpts, vthetas, args.global_save_path, key)
                    if args.cricket is False:
                        cricket_props = None

                    if isinstance(vid, list):
                        check_topdown_tracking(key, vid[0], args.global_save_path, dlc_data=vcleanpts, vext=viewext, head_ang=vtheta)
                    else:
                        check_topdown_tracking(key, vid, args.global_save_path, dlc_data=vcleanpts, vext=viewext, head_ang=vtheta)
                    vpts.name = 'raw_pt_values'
                    vcleanpts.name = 'output_pt_values'
                    vthetas.name = 'head_angle_values'
                    if cricket_props is not None:
                        if v == 'v1':
                            gatheredtop = xr.merge([vcleanpts, vthetas, cricket_props])
                        elif v != 'v1':
                            concattop = xr.merge([vcleanpts, vthetas, cricket_props])
                            gatheredtop = xr.concat([gatheredtop, concattop], dim='view', fill_value=np.nan)
                    if cricket_props is None:
                        if v == 'v1':
                            gatheredtop = xr.merge([vcleanpts, vthetas])
                        elif v != 'v1':
                            concattop = xr.merge([vcleanpts, vthetas])
                            gatheredtop = xr.concat([gatheredtop, concattop], dim='view', fill_value=np.nan)
                    print('tracking sucessful for ' + str(v))
                except KeyError: # in case not all three views exist
                    print('failed to find view ' + str(v))
                    pass
            if eyeRvidpath == righteye_vid_files[0]:
                gatheredtop['trial'] = key + viewext
                topout = gatheredtop
            elif eyeRvidpath != righteye_vid_files[0]:
                gatheredtop['trial'] = key + viewext
                topout = xr.concat([topout, gatheredtop], dim='trial', fill_value=np.nan)

        # EYE VIEW PROCESSING
        if eyedlc is not None:
            eyedlc['trial'] = key; eyetime['trial'] = key
            eye_vlist = ['v1', 'v2'] # v1 is always the left eye, and v2 is the right eye
            for v in eye_vlist:
                try:
                    vpts = eyedlc[v]
                    foundv = True
                except KeyError:
                    print('failed to find view ' + str(v))
                    foundv = False
                    pass
                if foundv is True:
                    print('tracking eye camera view ' + str(v) + ' for ' + str(key))
                    if v == 'v1':
                        vid = eyeLvidpath
                        viewext = 'LEye'
                    elif v == 'v2':
                        vid = eyeRvidpath
                        viewext = 'REye'
                    vparams = eye_tracking(vpts, eyenames, args.global_save_path, key, args.lik_thresh, args.pxl_thresh, args.eye_pt_num, args.tear)
                    # check_eye_calibration(vparams, vpts, args.global_save_path, key, args.ell_thresh)
                    if vid != []:
                        if isinstance(vid, list):
                            check_eye_tracking(key, vid[0], args.global_save_path, dlc_data=vpts, ell_data=vparams, vext=viewext)
                        else:
                            check_eye_tracking(key, vid, args.global_save_path, dlc_data=vpts, ell_data=vparams, vext=viewext)
                    vpts.name = 'raw_pt_values'
                    vparams.name = 'ellipse_param_values'
                    if v == 'v1':
                        gatheredeye = xr.merge([vpts, vparams])
                    elif v != 'v1':
                        concateye = xr.merge([vpts, vparams])
                        gatheredeye = xr.concat([gatheredeye, concateye], dim='view', fill_value=np.nan)
                    print('tracking sucessful for ' + str(v))
                    if v == 'v1':
                        world_vid_path = worldLvidpath; world_time_path = worldLvidpath
                        eye_time_path = eyeLtimepath; worldext = 'LWorld'
                    elif v == 'v2':
                        world_vid_path = worldRvidpath; world_time_path = worldRvidpath
                        eye_time_path = eyeRtimepath; worldext = 'RWorld'
                    if world_vid_path != [] and world_time_path != []:
                        # for now, adjust_world() will only use the first topview because it's only plotted for context, not actual analysis
                        # should eliminate dependence on topdown data... topdown might not always be there
                        print('finding pupil rotation for ' + viewext + ' world view')
                        find_pupil_rotation(args.global_data_path, key, viewext, 'TOP1', worldext, vparams, args.global_save_path, 'linear')
                    else:
                        print('no world camera found for ' + key)
            # if there's only one trial being passed in, make the first trial the thing to be saved out
            if eyeRvidpath == righteye_vid_files[0]:
                gatheredeye['trial'] = key + viewext
                eyeout = gatheredeye
            # if there's more than one trial, concat them together and make a single large Dataset of eye trials
            elif eyeRvidpath != righteye_vid_files[0]:
                gatheredeye['trial'] = key + viewext
                eyeout = xr.concat([topout, gatheredeye], dim='trial', fill_value=np.nan)

        print('processing of trial ' + key + ' is complete... outputs wont be saved until the end of all queued trials')

    try:
        savecomplete(topout, args.global_save_path, 'tops')
    except NameError:
        print('no top .nc file saved because no data was passed')
    try:
        savecomplete(eyeout, args.global_save_path, 'eyes')
    except NameError:
        print('no eye .nc file saved because no data was passed')

    ### end of freely moving case ###


############


elif args.experiment_type == 'jumping':

    print('getting DLC, video, and  time files')
    # find DLC files in global_data_path
    dlc_glob_keys = ['*TOP*.h5', '*REYE*.h5', '*LEYE*.h5', '*SIDE*.h5']
    dlc_paths = find_paths(args.global_data_path, dlc_glob_keys)
    topdown_dlc_files = dlc_paths[0]; righteye_dlc_files = dlc_paths[1]; lefteye_dlc_files = dlc_paths[2]; side_dlc_files = dlc_paths[3]

    # find video files in global_data_path
    vid_glob_keys = ['*TOP*.avi', '*REYE*.avi', '*LEYE*.avi', '*SIDE*.avi']
    vid_paths = find_paths(args.global_data_path, vid_glob_keys)
    topdown_vid_files = vid_paths[0]; righteye_vid_files = vid_paths[1]; lefteye_vid_files = vid_paths[2]; side_vid_files = vid_paths[3]

    # find time files in global_data_path
    time_glob_keys = ['*TOP*BonsaiTS.csv', '*REYE*BonsaiTS.csv', '*LEYE*BonsaiTS.csv', '*SIDE*BonsaiTS.avi']
    time_paths = find_paths(args.global_data_path, time_glob_keys)
    topdown_time_files = time_paths[0]; righteye_time_files = time_paths[1]; lefteye_time_files = time_paths[2]; side_time_files = time_paths[3]

    for eyeRvidpath in righteye_vid_files:
        eyeRfullname = os.path.split(eyeRvidpath)[1]
        key_pieces = eyeRfullname.split('_')[:-1]
        key = '_'.join(key_pieces)

        print('starting on ' + str(key))

        # in the case where there's only one DLC file for each camera type (i.e. one trial only), run this...
        if len(righteye_vid_files) == 1:
            top1dlcpath = topdown1_dlc_files; top2dlcpath = topdown2_dlc_files; top3dlcpath = topdown3_dlc_files
            eyeLdlcpath = lefteye_dlc_files; eyeRdlcpath = righteye_dlc_files
            top1vidpath = topdown1_vid_files; top2vidpath = topdown2_vid_files; top3vidpath = topdown3_vid_files
            eyeLvidpath = lefteye_vid_files;
            worldLvidpath = leftworld_vid_files; worldRvidpath = rightworld_vid_files
            top1timepath = topdown1_time_files; top2timepath = topdown2_time_files; top3timepath = topdown3_time_files
            eyeLtimepath = lefteye_time_files; eyeRtimepath = righteye_time_files
            worldLtimepath = leftworld_time_files; worldRtimepath = rightworld_time_files

        elif len(righteye_vid_files) > 1:
            # get associated DLC files for top1dlcpath
            topdlcpath = [i for i in topdown_dlc_files if key in i]
            eyeLdlcpath = [i for i in lefteye_dlc_files if key in i]
            eyeRdlcpath = [i for i in righteye_dlc_files if key in i]
            sidedlcpath = [i for i in side_dlc_files if key in i]

            # get associated video files for top1dlcpath
            topvidpath = [i for i in topdown_vid_files if key in i]
            eyeLvidpath = [i for i in lefteye_vid_files if key in i]
            eyeRvidpath = [i for i in righteye_vid_files if key in i]
            sidevidpath = [i for i in side_vid_files if key in i]

            # get associated time files for top1dlcpath
            toptimepath = [i for i in topdown_time_files if key in i]
            eyeLtimepath = [i for i in lefteye_time_files if key in i]
            eyeRtimepath = [i for i in righteye_time_files if key in i]
            sidetimepath = [i for i in side_time_files if key in i]

        # build xarray DataArrays for each camera type which contain data for each view of that type
        # also make DataArray from the timestamp list for each view of the camera type
        topdlc, toptime, topnames = read_paths(topdlcpath, toptimepath)
        eyedlc, eyetime, eyenames = read_paths(eyeRdlcpath, eyeRtimepath, eyeLdlcpath, eyeLtimepath)
        sidedlc, sidetime, sidenames = read_paths(sidedlcpath, sidetimepath)

        # TOPDOWN VIEW PROCESSING
        if topdlc is not None:
            vpts = topdlc['v1']
            print('trying to track topdown camera view for ' + str(key))
            vcleanpts, nose_x, nose_y = topdown_tracking(vpts, topnames, args.global_save_path, key, args.lik_thresh, args.coord_cor, args.topdown_pt_num, cricket=False, time_path=time[0])
            vthetas = head_angle(vcleanpts, topnames, args.lik_thresh, args.global_save_path, args.cricket, key, nose_x, nose_y, time)
            if isinstance(vid, list):
                check_topdown_tracking(key, vid[0], args.global_save_path, dlc_data=vcleanpts, vext=viewext, head_ang=vtheta)
            else:
                check_topdown_tracking(key, vid, args.global_save_path, dlc_data=vcleanpts, vext=viewext, head_ang=vtheta)
            vcleanpts.name = 'output_pt_values'
            vthetas.name = 'head_angle_values'
            gatheredtop = xr.merge([vcleanpts, vthetas])
            print('tracking sucessful for ' + str(v))
            if eyeRvidpath == righteye_vid_files[0]:
                gatheredtop['trial'] = key + viewext
                topout = gatheredtop
            elif eyeRvidpath != righteye_vid_files[0]:
                gatheredtop['trial'] = key + viewext
                topout = xr.concat([topout, gatheredtop], dim='trial', fill_value=np.nan)

        # EYE VIEW PROCESSING
        if eyedlc is not None:
            eyedlc['trial'] = key; eyetime['trial'] = key
            eye_vlist = ['v1', 'v2'] # v1 is always the left eye, and v2 is the right eye
            for v in eye_vlist:
                try:
                    vpts = eyedlc[v]
                    foundv = True
                except KeyError:
                    print('failed to find view ' + str(v))
                    foundv = False
                    pass
                if foundv is True:
                    print('tracking eye camera view ' + str(v) + ' for ' + str(key))
                    if v == 'v1':
                        vid = eyeLvidpath
                        viewext = 'LEye'
                    elif v == 'v2':
                        vid = eyeRvidpath
                        viewext = 'REye'
                    vparams = eye_tracking(vpts, eyenames, args.global_save_path, key, args.lik_thresh, args.pxl_thresh, args.eye_pt_num, args.tear)
                    # check_eye_calibration(vparams, vpts, args.global_save_path, key, args.ell_thresh)
                    if vid != []:
                        if isinstance(vid, list):
                            check_eye_tracking(key, vid[0], args.global_save_path, dlc_data=vpts, ell_data=vparams, vext=viewext)
                        else:
                            check_eye_tracking(key, vid, args.global_save_path, dlc_data=vpts, ell_data=vparams, vext=viewext)
                    vpts.name = 'raw_pt_values'
                    vparams.name = 'ellipse_param_values'
                    if v == 'v1':
                        gatheredeye = xr.merge([vpts, vparams])
                    elif v != 'v1':
                        concateye = xr.merge([vpts, vparams])
                        gatheredeye = xr.concat([gatheredeye, concateye], dim='view', fill_value=np.nan)
                    print('tracking sucessful for ' + str(v))
            # if there's only one trial being passed in, make the first trial the thing to be saved out
            if eyeRvidpath == righteye_vid_files[0]:
                gatheredeye['trial'] = key + viewext
                eyeout = gatheredeye
            # if there's more than one trial, concat them together and make a single large Dataset of eye trials
            elif eyeRvidpath != righteye_vid_files[0]:
                gatheredeye['trial'] = key + viewext
                eyeout = xr.concat([topout, gatheredeye], dim='trial', fill_value=np.nan)

        # SIDE VIEW PROCESSING
        if sidedlc is not None and gatheredeye is not None:
            print('preparing eye and topdown data for jumping gaze trace of ' + viewext)
            LEye_params = eyeout.sel(variable='ellipse_param_values', trial=key, view='LEye')
            REye_params = eyeout.sel(variable='ellipse_param_values', trial=key, view='REye')
            top_params = topout.sel(variable='head_angle_values', trial=key)

            print('finding trace of gaze while jumping for ' + viewext + ' side view')
            jump_gaze_trace(args.global_data_path, args.global_save_path, key, REye_params, LEye_params, top_params, sidedlc, eyeRvidpath, eyeLvidpath, sidevidpath, topvidpath)
            jump_cc(args.global_data_path, args.global_save_path, key, REye_params, LEye_params, top_params, sidedlc)
        elif foundv is False or foundworld is False:
            print('failed to find ' + viewext + ' side view')


        print('processing of trial ' + key + ' is complete... outputs wont be saved until the end of all queued trials')

    try:
        savecomplete(topout, args.global_save_path, 'tops')
    except NameError:
        print('no top .nc file saved because no data was passed')
    try:
        savecomplete(eyeout, args.global_save_path, 'eyes')
    except NameError:
        print('no eye .nc file saved because no data was passed')

    ### end of jumping case ###

elif args.experiment_type != 'jumping' & args.experiment_type != 'freelymoving':
    print(str(args.experiment_type) + ' is not an experiment type; should be freelymoving or jumping')
