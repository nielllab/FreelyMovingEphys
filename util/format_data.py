"""
format_data.py
"""
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

from util.open_data import open_h5, open_ma_h5
from util.time import open_time

def format_frames(vid_path, config):
    """
    add videos to xarray
    INPUTS
        vid_path:path to an avi
        config:options dict
    OUTPUTS
        formatted_frames:xarray DataArray of video as b/w int8
    """
    # open the .avi file
    vidread = cv2.VideoCapture(vid_path)
    # empty array that is the target shape
    # should be number of frames x downsampled height x downsampled width
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*config['parameters']['outputs_and_visualization']['dwnsmpl']),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*config['parameters']['outputs_and_visualization']['dwnsmpl'])], dtype=np.uint8)
    # iterate through each frame
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # read the frame in and make sure it is read in correctly
        ret, frame = vidread.read()
        if not ret:
            break
        # convert to grayyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # downsample the frame by an amount specified in the config file
        sframe = cv2.resize(frame, (0,0), fx=config['parameters']['outputs_and_visualization']['dwnsmpl'], fy=config['parameters']['outputs_and_visualization']['dwnsmpl'], interpolation=cv2.INTER_NEAREST)
        # add the downsampled frame to all_frames as int8
        all_frames[frame_num,:,:] = sframe.astype(np.int8)
    # store the combined video frames in an xarray
    formatted_frames = xr.DataArray(all_frames.astype(np.int8), dims=['frame', 'height', 'width'])
    # label frame numbers in the xarray
    formatted_frames.assign_coords({'frame':range(0,len(formatted_frames))})
    # delete all frames, since it's somewhat large in memory
    del all_frames

    return formatted_frames

def h5_to_xr(pt_path, time_path, view, config):
    """
    build an xarray DataArray of the a single camera's dlc point .h5 files and .csv timestamp
    works for any camera type
    INPUTS
        pt_path:filepath to the .h5
        time_path:filepath to a .csv
        view:str of camera name (i.e. REYE)
    OUTPUTS
        xrpts:pose estimate xarray
    """
    # check that pt_path exists
    if pt_path is not None and pt_path != [] and time_path is not None:
        # open multianimal project with a different function than single animal h5 files
        if 'TOP' in view and config['pose_estimation']['multianimal_top_project'] is True:
            # add a step to convert pickle files here?
            pts = open_ma_h5(pt_path)
        # otherwise, use regular h5 file read-in
        else:
            pts, names = open_h5(pt_path)
        # read time file, pass length of points so that it will know if that length matches the length of the timestamps
        # if they don't match because time was not interpolated to match deinterlacing, the timestamps will be interpolated
        time = open_time(time_path, len(pts))
        # label dimensions of the points dataarray
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
        # label the camera view
        xrpts.name = view
        # assign timestamps as a coordinate to the 
        try:
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:])) # indexing [1:] into time because first row is the empty header, 0
        # correcting for issue caused by small differences in number of frames
        except ValueError:
            diff = len(time[1:]) - len(xrpts['frame'])
            if diff > 0: # time is longer
                diff = abs(diff)
                new_time = time.copy()
                new_time = new_time[0:-diff]
                xrpts = xrpts.assign_coords(timestamps=('frame', new_time[1:]))
            elif diff < 0: # frame is longer
                diff = abs(diff)
                timestep = time[1] - time[0]
                new_time = time.copy()
                for i in range(1,diff+1):
                    last_value = new_time[-1] + timestep
                    new_time = np.append(new_time, pd.Series(last_value))
                xrpts = xrpts.assign_coords(timestamps=('frame', new_time[1:]))
            else: # equal (probably won't happen because ValueError should have been caused by unequal lengths)
                xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
    # pt_path will have no data in it for world cam data, so it will make an xarray with just timestamps
    elif pt_path is None or pt_path == [] and time_path is not None:
        if time_path is not None and time_path != []:
            # read in the time
            time = open_time(time_path)
            # setup frame indices
            xrpts = xr.DataArray(np.zeros([len(time)-1]), dims=['frame'])
            # assign frame coordinates, then timestamps
            xrpts = xrpts.assign_coords({'frame':range(0,len(xrpts))})
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
            names = None
        elif time_path is None or time_path == []:
            xrpts = None; names = None
    # if timestamps are missing, still read in and format as xarray
    elif pt_path is not None and pt_path != [] and time_path is None:
        # open multianimal project with a different function than single animal h5 files
        if 'TOP' in view and config['pose_estimation']['multianimal_top_project'] is True:
            # add a step to convert pickle files here?
            pts = open_ma_h5(pt_path)
        # otherwise, use regular h5 file read-in
        else:
            pts, names = open_h5(pt_path)
        # label dimensions of the points dataarray
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
        # label the camera view
        xrpts.name = view

    return xrpts

def split_xyl(names, data, thresh):
    """
    convert xarray DataArray of DLC x and y positions and likelihood values into separate pandas data structures
    INPUTS
        names: list of names of points
        data: xarray DataArray of data
        thresh: likelihood threshold
    OUTPUTS
        x_vals: pandas dataframe of x positions
        y_vals: pandas dataframe of y positions
        likeli_pts: pandas dataframe of likelihoods
    """
    x_locs = []
    y_locs = []
    likeli_locs = []
    # seperate the lists of point names into x, y, and likelihood
    for loc_num in range(0, len(names)):
        loc = names[loc_num]
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
            likeli_pts = data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            likeli_pts = xr.concat([likeli_pts, data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    for loc_num in range(0, len(x_locs)):
        pt_loc = x_locs[loc_num]
        # threshold from likelihood
        data.sel(point_loc=pt_loc)[data.sel(point_loc=pt_loc) < thresh] = np.nan
        if loc_num == 0:
            x_pts = data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            x_pts = xr.concat([x_pts, data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    for loc_num in range(0, len(y_locs)):
        pt_loc = y_locs[loc_num]
        # threshold from likelihood
        data.sel(point_loc=pt_loc)[data.sel(point_loc=pt_loc) < thresh] = np.nan
        if loc_num == 0:
            y_pts = data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            y_pts = xr.concat([y_pts, data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    x_pts = xr.DataArray.squeeze(x_pts)
    y_pts = xr.DataArray.squeeze(y_pts)
    likeli_pts = xr.DataArray.squeeze(likeli_pts)
    # convert to dataframe, transpose so points are columns
    x_vals = xr.DataArray.to_pandas(x_pts).T
    y_vals = xr.DataArray.to_pandas(y_pts).T
    likeli_pts = xr.DataArray.to_pandas(likeli_pts).T

    return x_vals, y_vals, likeli_pts

# 
def safe_xr_merge(obj_list, dim_name='frame'):
    """
    safely merge list of xarray dataarrays, even when their lengths do not match
    INPUTS
        obj_list: xarray DataArrays to merge as a list (objects should all have a shared dim)
        dim_name: name of xr dimension to merge along, default='frame'
    OUTPUTS
        merge_objs: merged xarray of all objects in input list, even if lengths do not match
    this is only a good idea if expected length differences will be minimal
    """
    max_lens = []
    # iterate through objects
    for obj in obj_list:
        # get the sizes of the dim, dim_name
        max_lens.append(dict(obj.frame.sizes)[dim_name])
    # get the smallest of the object's length's
    set_len = np.min(max_lens)
    # shorten everything to the shortest length found
    out_objs = []
    for obj in obj_list:
        # get the length of the current object
        obj_len = dict(obj.frame.sizes)[dim_name]
        # if the size of dim is longer
        if obj_len > set_len:
            # how much does it need to be shortened by?
            diff = obj_len - set_len
            # what indeces should be kept?
            good_inds = range(0,obj_len-diff)
            # index to remove what would be jagged ends
            obj = obj.sel(frame=good_inds)
            # add to the list of objects to merge
            out_objs.append(obj)
        # if it is the smallest length or all objects have the same length
        else:
            # just append it to the list of objects to merge
            out_objs.append(obj)
    # do the merge with the lengths all matching along provided dimension
    merge_objs = xr.merge(out_objs)

    return merge_objs