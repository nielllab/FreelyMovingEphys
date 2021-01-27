"""
format_data.py

functions for manipulating the format of data

Dec. 02, 2020
"""
# package imports
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
import os
import fnmatch
import dateutil
import cv2
from tqdm import tqdm
from datetime import datetime
import time
import argparse
# module imports
from util.open_data import open_h5, open_ma_h5
from util.time import open_time

# add videos to xarray
# will downsample by ratio in config file and convert to black and white uint8
def format_frames(vid_path, config):
    print('formatting video into DataArray')
    vidread = cv2.VideoCapture(vid_path)
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*config['dwnsmpl']),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*config['dwnsmpl'])], dtype=np.uint8)
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = vidread.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sframe = cv2.resize(frame, (0,0), fx=config['dwnsmpl'], fy=config['dwnsmpl'], interpolation=cv2.INTER_NEAREST)
        all_frames[frame_num,:,:] = sframe.astype(np.int8)

    formatted_frames = xr.DataArray(all_frames.astype(np.int8), dims=['frame', 'height', 'width'])
    formatted_frames.assign_coords({'frame':range(0,len(formatted_frames))})
    del all_frames

    return formatted_frames

# build an xarray DataArray of the a single camera's dlc point .h5 files and .csv timestamp corral_files
# function is used for any camera view regardless of type, though extension must be specified in 'view' argument
def h5_to_xr(pt_path, time_path, view, config):
    if pt_path is not None and pt_path != []:
        if 'TOP' in view and config['multianimal_TOP'] is True:
            # add a step to convert pickle files here?
            pts = open_ma_h5(pt_path)
        else:
            pts, names = open_h5(pt_path)
        time = open_time(time_path, len(pts))
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
        xrpts.name = view
        try:
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:])) # indexing [1:] into time because first row is the empty header, 0
        except ValueError:
            # this is both messy and temporary -- trying to fix issue: ValueError: conflicting sizes for dimension 'frame': length 71521 on 'timestamps' and length 71522 on 'frame'
            timestep = time[1] - time[0]
            last_value = time[-1] + timestep
            new_time = np.append(time, pd.Series(last_value))
            xrpts = xrpts.assign_coords(timestamps=('frame', new_time[1:]))
    elif pt_path is None or pt_path == []:
        if time_path is not None and time_path != []:
            time = open_time(time_path)
            xrpts = xr.DataArray(np.zeros([len(time)-1]), dims=['frame'])
            xrpts = xrpts.assign_coords({'frame':range(0,len(xrpts))})
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
            names = None
        elif time_path is None or time_path == []:
            xrpts = None; names = None

    return xrpts

# convert xarray DataArray of DLC x and y positions and likelihood values into separate pandas data structures
def split_xyl(eye_names, eye_data, thresh):
    x_locs = []
    y_locs = []
    likeli_locs = []
    for loc_num in range(0, len(eye_names)):
        loc = eye_names[loc_num]
        if '_x' in loc:
            x_locs.append(loc)
        elif '_y' in loc:
            y_locs.append(loc)
        elif 'likeli' in loc:
            likeli_locs.append(loc)
    # get the xarray, split up into x, y,and likelihood
    for loc_num in range(0, len(likeli_locs)):
        pt_loc = likeli_locs[loc_num]
        if loc_num == 0:
            likeli_pts = eye_data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            likeli_pts = xr.concat([likeli_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    for loc_num in range(0, len(x_locs)):
        pt_loc = x_locs[loc_num]
        # threshold from likelihood
        eye_data.sel(point_loc=pt_loc)[eye_data.sel(point_loc=pt_loc) < thresh] = np.nan
        if loc_num == 0:
            x_pts = eye_data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            x_pts = xr.concat([x_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    for loc_num in range(0, len(y_locs)):
        pt_loc = y_locs[loc_num]
        # threshold from likelihood
        eye_data.sel(point_loc=pt_loc)[eye_data.sel(point_loc=pt_loc) < thresh] = np.nan
        if loc_num == 0:
            y_pts = eye_data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            y_pts = xr.concat([y_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    x_pts = xr.DataArray.squeeze(x_pts)
    y_pts = xr.DataArray.squeeze(y_pts)
    likeli_pts = xr.DataArray.squeeze(likeli_pts)
    # convert to dataframe, transpose so points are columns
    x_vals = xr.DataArray.to_pandas(x_pts).T
    y_vals = xr.DataArray.to_pandas(y_pts).T
    likeli_pts = xr.DataArray.to_pandas(likeli_pts).T

    return x_vals, y_vals, likeli_pts

# safely merge list of xarray dataarrays, even when their lengths do not match
# always does it along dim 'frame'
def safe_xr_merge(obj_list):
    max_lens = []
    for obj in obj_list:
        max_lens.append(dict(obj.frame.sizes)['frame'])
    set_len = np.min(max_lens)

    out_objs = []
    for obj in obj_list:
        obj_len = dict(obj.frame.sizes)['frame']
        if obj_len > set_len:
            diff = obj_len - set_len
            obj = obj.sel(frame=slice(:-diff)]
            out_objs.append(obj)
        else:
            out_objs.append(obj)
    
    merge_objs = xr.merge(out_objs)

    return merge_objs