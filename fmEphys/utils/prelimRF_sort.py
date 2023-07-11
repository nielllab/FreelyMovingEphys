"""
fmEphys/utils/prelimRF_sort.py

Preliminary receptive field mapping  with spike-sorted ephys data.


Written by DMM, 2021
"""


import os
import subprocess
from tqdm import tqdm
from glob import glob
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import scipy.interpolate
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys as fme


def open_time(path, dlc_len=None, force_shift=False):
    """ Open a timestamp file.

    Parameters
    ----------
    path : str
        Path to the timestamp .csv file.
    dlc_len : int
        Number of frames in the DLC video.
    force_shift : bool
        Force the timestamps to be shifted?
    
    Returns
    -------
    time_out : np.array
        Timestamps in seconds.

    """

    # read in the timestamps if they've come directly from cameras
    read_time = pd.read_csv(path,
                            encoding='utf-8', engine='c',
                            header=None).squeeze()

    if read_time[0] == 0:
        # in case header == 0, which is true of some files, drop that
        # header which will have been read in as the first entry  
        read_time = read_time[1:]

    time_in = []
    fmt = '%H:%M:%S.%f'

    if read_time.dtype!=np.float64:

        for current_time in read_time:

            currentT = str(current_time).strip()

            try:
                t = datetime.strptime(currentT,fmt)
            
            except ValueError as v:
                
                ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                if ulr:
                    currentT = currentT[:-ulr]
            
            try:
                time_in.append((datetime.strptime(currentT, '%H:%M:%S.%f') -            \
                                datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')).total_seconds())
            
            except ValueError:
                time_in.append(np.nan)

        time_in = np.array(time_in)
        
    else:
        time_in = read_time.values

    # auto check if vids were deinterlaced
    if dlc_len is not None:

        # test length of the time just read in as it compares to the length of the data,
        # correct for deinterlacing if needed
        timestep = np.nanmedian(np.diff(time_in, axis=0))
        
        if dlc_len > len(time_in):
            
            time_out = np.zeros(np.size(time_in, 0)*2)
            
            # shift each deinterlaced frame by 0.5 frame period forward/backwards relative
            # to timestamp
            time_out[::2] = time_in - 0.25 * timestep
            time_out[1::2] = time_in + 0.25 * timestep
        
        elif dlc_len == len(time_in):
            time_out = time_in
        
        elif dlc_len < len(time_in):
            time_out = time_in

    elif dlc_len is None:
        time_out = time_in

    # force the times to be shifted if the user is sure it should be done
    if force_shift is True:
        
        # test length of the time just read in as it compares to the length of the
        # data, correct for deinterlacing
        timestep = np.nanmedian(np.diff(time_in, axis=0))
        time_out = np.zeros(np.size(time_in, 0)*2)
        
        # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
        time_out[::2] = time_in - 0.25 * timestep
        time_out[1::2] = time_in + 0.25 * timestep

    return time_out


def safe_xr_merge(obj_list, dim_name='frame'):
    """ Merge list of DataArrays without matching dims.
    
    Only use if expected length differences will be minimal.
    This eliminates data in the time dimension.

    Parameters
    ----------
    obj_list : list
        List of DataArrays to merge.
    dim_name : str
        Name of the dimension to merge along. Default
        is 'frame'.
    
    Returns
    -------
    merge_objs : xr.DataArray
        Merged xarray of all objects in input list. They will
        be the same size in the dimension `dim_name`.

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


def open_h5(path):
    """ Read in .h5 DLC files and manage column names
    Parameters:
    path (str): filepath to .h5 file outputs by DLC
    Returns:
    pts (pd.DataFrame): values for position
    pt_loc_names (list): column names
    """

    try:
        # read the .hf file when there is no key
        pts = pd.read_hdf(path)

    except ValueError:
        # read in .h5 file when there is a key set in corral_files.py
        pts = pd.read_hdf(path, key='data')

    # organize columns
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]

    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ',
                                        '_') for n in range(len(pts.columns))})
    
    pt_loc_names = pts.columns.values

    return pts, pt_loc_names


def open_ma_h5(path):
    """ Open .h5 file of a multianimal DLC project
    Parameters:
    path (str): filepath to .h5 file outputs by DLC
    Returns:
    pts (Pd.DataFrame): pandas dataframe of points
    """

    pts = pd.read_hdf(path)

    # flatten columns from MultiIndex 
    pts.columns = ['_'.join(col[:][1:]).strip() for col in pts.columns.values]

    return pts


def format_frames(vid_path, cfg):
    """ Add videos to xarray
    Parameters:
    vid_path (str): path to an avi
    cfg (dict): options
    Returns:
    formatted_frames (xr.DataArray): of video as b/w int8
    """

    # open the .avi file
    vidread = cv2.VideoCapture(vid_path)

    # empty array that is the target shape
    # should be number of frames x downsampled height x downsampled width
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
            (vidread.get(cv2.CAP_PROP_FRAME_HEIGHT) * cfg['parameters']['outputs_and_visualization']['dwnsmpl']),
            int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH) * cfg['parameters']['outputs_and_visualization']['dwnsmpl'])],
            dtype=np.uint8)
    
    # iterate through each frame
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        
        # read the frame in and make sure it is read in correctly
        ret, frame = vidread.read()
        if not ret:
            break
        
        # convert to grayyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # downsample the frame by an amount specified in the cfg file
        sframe = cv2.resize(frame, (0,0),
                            fx=cfg['parameters']['outputs_and_visualization']['dwnsmpl'],
                            fy=cfg['parameters']['outputs_and_visualization']['dwnsmpl'],
                            interpolation=cv2.INTER_NEAREST
                            )
        
        # add the downsampled frame to all_frames as int8
        all_frames[frame_num,:,:] = sframe.astype(np.int8)
    
    # store the combined video frames in an xarray
    formatted_frames = xr.DataArray(all_frames.astype(np.int8),
                                    dims=['frame', 'height', 'width'])
    
    # label frame numbers in the xarray
    formatted_frames.assign_coords({'frame':range(0,len(formatted_frames))})

    # delete all frames, since it's somewhat large in memory
    del all_frames

    return formatted_frames


def h5_to_xr(pt_path, time_path, view, cfg):
    """ Build an xarray DataArray of the a single camera's dlc point .h5 files and .csv timestamp
    Parameters:
    pt_path (str): filepath to the .h5
    time_path (str): filepath to a .csv
    view (str): camera name (i.e. REYE)
    
    Returns:
    xrpts (xr.DataArray): pose estimate
    """

    # check that pt_path exists
    if pt_path is not None and pt_path != [] and time_path is not None:

        # open multianimal project with a different function than single animal h5 files
        if 'TOP' in view and cfg['pose_estimation']['DLC_topMA'] is True:

            # add a step to convert pickle files here?
            pts = open_ma_h5(pt_path)

        # otherwise, use regular h5 file read-in
        else:
            pts, names = open_h5(pt_path)

        # read time file, pass length of points so that it will know if that length
        # matches the length of the timestamps
        # if they don't match because time was not interpolated to match
        # deinterlacing, the timestamps will be interpolated
        time = open_time(time_path, len(pts))

        # label dimensions of the points dataarray
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])

        # label the camera view
        xrpts.name = view

        # assign timestamps as a coordinate to the 
        try:
            # indexing [1:] into time because first row is the empty header, 0
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
            
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

            else:
                # equal (probably won't happen because ValueError should have been
                # caused by unequal lengths)
                xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))

    # pt_path will have no data in it for world cam data, so it will make an xarray with
    # just timestamps
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
            xrpts = None
            names = None

    # if timestamps are missing, still read in and format as xarray
    elif pt_path is not None and pt_path != [] and time_path is None:

        
        if 'TOP' in view and cfg['pose_estimation']['DLC_topMA'] is True:
            # open multianimal project with a different function than single animal h5 files

            # add a step to convert pickle files here?
            pts = open_ma_h5(pt_path)

        else:
            # otherwise, use regular h5 file read-in
            pts, names = open_h5(pt_path)

        # label dimensions of the points dataarray
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])

        # label the camera view
        xrpts.name = view

    return xrpts


def deinterlace_data(cfg, vid_list=None, time_list=None):
    """ Deinterlace videos, shift times to match the new video frame count.
    Searches subdirectories if vid_list and time_list are both None.
    If lists of files are provided, it will not search subdirectories and instead
    analyze items in those lists.

    Parameters:
    cfg (dict): options dict
    vid_list (list): .avi file paths for videos to deinterlace (optional)
    time_list (list): .csv file paths of timestamps matching videos to deinterlace (optional)

    """

    # get paths out of the cfg dictionary
    data_path = cfg['animal_dir']

    # find all the files assuming no specific files are listed
    if vid_list is None:
        avi_list = fme.find('*.avi', data_path)
        csv_list = fme.find('*.csv', data_path)

    # if a specific list of videos is provided, ignore the cfg file's data path
    elif vid_list is not None:
        avi_list = vid_list.copy()
        csv_list = time_list.copy()

    # iterate through each video
    for this_avi in avi_list:
        current_path = os.path.split(this_avi)[0]

        # make a save path that keeps the subdirectories
        # get out an key from the name of the video that will be shared with all
        # other data of this trial
        vid_name = os.path.split(this_avi)[1]
        key_pieces = vid_name.split('.')[:-1]
        key = '.'.join(key_pieces)

        # then, find those other pieces of the trial using the key
        try:
            this_csv = [i for i in csv_list if key in i][0]
            csv_present = True

        except IndexError:
            csv_present = False

        # open the video
        cap = cv2.VideoCapture(this_avi)

        # get some info about the video
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # number of total frames
        fps = cap.get(cv2.CAP_PROP_FPS) # frame rate

        # make sure the save directory exists
        if not os.path.exists(current_path):
            os.makedirs(current_path)

        # files that will need to be deinterlaced will be read in with a frame rate of 30 frames/sec
        elif fps == 30:

            print('starting to deinterlace and interpolate on ' + key)
            
            # create save path
            avi_out_path = os.path.join(current_path, (key + 'deinter.avi'))

            # flip the eye video horizonally and vertically and deinterlace, if this is specified in the cfg
            if cfg['deinterlace']['flip_eye_during_deinter'] is True and ('EYE' in this_avi or 'WORLD' in this_avi):
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0, vflip, hflip, scale=640:480',
                                 '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac',
                                 '-b:a', '256k', '-y', avi_out_path])
            
            # or, deinterlace without flipping
            elif cfg['deinterlace']['flip_eye_during_deinter'] is False and ('EYE' in this_avi or 'WORLD' in this_avi):
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0, scale=640:480',
                                 '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a',
                                 'aac', '-b:a', '256k', '-y', avi_out_path])
            
            # correct the frame count of the video
            # now that it's deinterlaced, the video has 2x the number of frames as before
            # this will be used to correct the timestamps associated with this video
            frame_count_deinter = frame_count * 2

            if csv_present is True:

                # get the save path for new timestamps
                csv_out_path = os.path.join(current_path, (key + '_BonsaiTSformatted.csv'))

                # read in the exiting timestamps, interpolate to match the new number of steps, and format as dataframe
                csv_out = pd.DataFrame(open_time(this_csv, int(frame_count_deinter)))
                
                # save new timestamps
                csv_out.to_csv(csv_out_path, index=False)

        else:
            print('frame rate not 30 or 60 for ' + key)


def undistort_vid(vidpath, savepath, mtx, dist, rvecs, tvecs):
    """
    undistort novel videos using provided camera calibration properties
    INPUTS
        vidpath: path to the video file
        savepath: file path (not a directory) into which the undistorted video will be saved
        mtx: camera matrix
        dist: distortion coefficients
        rvecs: rotation vectors
        tvecs: translation vectors
    OUTPUTS
        None
    if vidpath and savepath are the same filename, the file will be overwritten
    saves a new copy of the video, after it has been undistorted
    """


    # open the video
    cap = cv2.VideoCapture(vidpath)

    # setup the file writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc,
                              60.0,
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    # iterate through all frames
    print('undistorting video')

    for step in tqdm(range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):

        # open frame and check that it opens correctly
        ret, frame = cap.read()

        if not ret:
            break

        # run opencv undistortion function
        undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)

        # write the frame to the video
        out_vid.write(undist_frame)

    out_vid.release()


def calibrate_new_world_vids(cfg):
    """
    calibrate novel world videos using previously genreated .npy of parameters
    INPUTS
        cfg: options dictionary
    OUTPUTS
        None
    """

    # load the parameters
    checker_in = np.load(cfg['calibration']['world_checker_npz'])

    # unpack camera properties
    mtx = checker_in['mtx']
    dist = checker_in['dist']
    rvecs = checker_in['rvecs']
    tvecs = checker_in['tvecs']

    # iterate through eye videos and save out a copy which has had distortions removed
    world_list = fme.find('*WORLDdeinter*.avi', cfg['animal_dir'])

    for world_vid in world_list:

        if 'plot' not in world_vid:

            savepath = '_'.join(world_vid.split('_')[:-1])+'_WORLDcalib.avi'

            undistort_vid(world_vid, savepath, mtx, dist, rvecs, tvecs)


def plot_spike_raster(goodcells):
    """
    plot spike raster so that superficial channels are at the top of the panel
    INPUTS
        goodcells: ephys dataframe
    OUTPUTS
        fig: figure
    """

    fig, ax = plt.subplots()
    ax.fontsize = 20
    n_units = len(goodcells)

    # iterate through units
    for i, ind in enumerate(goodcells.index):

        # array of spike times
        sp = np.array(goodcells.at[ind,'spikeT'])

        # make vertical line for each time the unit fires
        plt.vlines(sp[sp<10],i-0.25,i+0.25)

        # plot only ten seconds
        plt.xlim(0, 10)

        # turn off ticks
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    plt.xlabel('secs', fontsize=20)
    plt.ylabel('unit number',fontsize=20)
    plt.ylim([n_units,0])

    return fig


def plot_spike_rate_vs_var(use, var_range, goodcells, useT, t, var_label):
    """
    plot spike rate vs a given variable (e.g. pupil radius, worldcam contrast, etc.)
    INPUTS
        use: varaible to plot (can be filtered e.g. only active times)
        var_range: range of bins to calculate response over
        goodcells: ephys dataframe
        useT: timestamps that match the vairable (use)
        t: timebase
        var_label: label for last panels xlabel
    OUTPUTS
        var_cent: x axis bins
        tuning: tuning curve across bins
        tuning_err: stderror of variable at each bin
        fig: figure
    """

    n_units = len(goodcells)

    scatter = np.zeros((n_units,len(use)))

    tuning = np.zeros((n_units,len(var_range)-1))
    tuning_err = tuning.copy()

    var_cent = np.zeros(len(var_range)-1)
    
    for j in range(len(var_range)-1):

        var_cent[j] = 0.5*(var_range[j] + var_range[j+1])

    for i, ind in enumerate(goodcells.index):

        rateInterp = scipy.interpolate.interp1d(t[0:-1],
                                                goodcells.at[ind,'rate'],
                                                bounds_error=False)
        
        scatter[i,:] = rateInterp(useT)

        for j in range(len(var_range)-1):

            usePts = (use>=var_range[j]) & (use<var_range[j+1])

            tuning[i,j] = np.nanmean(scatter[i, usePts])

            tuning_err[i,j] = np.nanstd(scatter[i, usePts]) / np.sqrt(np.count_nonzero(usePts))

    fig = plt.subplots(int(np.ceil(n_units/7)),7,
                       figsize=(35,np.int(np.ceil(n_units/3))), dpi=50)
    
    for i, ind in enumerate(goodcells.index):

        plt.subplot(int(np.ceil(n_units/7)), 7, i+1)
        plt.errorbar(var_cent, tuning[i,:], yerr=tuning_err[i,:])

        try:
            plt.ylim(0, np.nanmax(tuning[i,:]*1.2))
        except ValueError:
            plt.ylim(0, 1)

        plt.xlim([var_range[0], var_range[-1]])
        plt.title(ind,fontsize=5)
        plt.xlabel(var_label,fontsize=5)
        plt.ylabel('sp/sec',fontsize=5)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)

    plt.tight_layout()

    return var_cent, tuning, tuning_err, fig


def plot_STA(goodcells, img_norm, worldT, movInterp, ch_count, lag=2, show_title=True):
    """
    plot spike-triggered average for either a single lag or a range of lags
    INPUTS
        goodcells: dataframe of ephys data
        img_norm: normalized worldcam video
        worldT: worldcam timestamps
        movInterp: interpolator for worldcam movie
        ch_count: number of probe channels
        lag: time lag, should be np.arange(-2,8,2) for range of lags, or 2 for single lag
        show_title: bool, whether or not to show title above each panel
    OUTPUTS
        staAll: STA receptive field of each unit
        fig: figure
    """

    n_units = len(goodcells)

    # model setup
    model_dt = 0.025
    model_t = np.arange(0, np.max(worldT), model_dt)
    model_nsp = np.zeros((n_units, len(model_t)))
    
    # get binned spike rate
    bins = np.append(model_t, model_t[-1]+model_dt)
    
    for i, ind in enumerate(goodcells.index):
        model_nsp[i,:], bins = np.histogram(goodcells.at[ind,'spikeT'], bins)
    
    # settting up video
    nks = np.shape(img_norm[0,:,:])
    nk = nks[0] * nks[1]
    model_vid = np.zeros((len(model_t),nk))
    
    for i in range(len(model_t)):

        model_vid[i,:] = np.reshape(movInterp(model_t[i]+model_dt/2), nk)
    
    # spike-triggered average
    staAll = np.zeros((n_units,
                       np.shape(img_norm)[1],
                       np.shape(img_norm)[2]))
    model_vid[np.isnan(model_vid)] = 0
    
    if type(lag) == int:

        fig = plt.subplots(int(np.ceil(n_units/10)), 10,
                           figsize=(20,np.int(np.ceil(n_units/3))), dpi=50)
        
        for c, ind in enumerate(goodcells.index):

            sp = model_nsp[c,:].copy()
            sp = np.roll(sp, -lag)
            sta = model_vid.T @ sp
            sta = np.reshape(sta, nks)
            nsp = np.sum(sp)

            plt.subplot(int(np.ceil(n_units/10)), 10, c+1)

            ch = int(goodcells.at[ind,'ch'])

            if ch_count == 64 or ch_count == 128:
                shank = np.floor(ch/32)
                site = np.mod(ch,32)

            else:
                shank = 0
                site = ch

            if show_title:
                plt.title(f'ind={ind!s} nsp={nsp!s}\n ch={ch!s} shank={shank!s}\n site={site!s}',
                          fontsize=5)
            plt.axis('off')

            if nsp > 0:
                sta = sta/nsp
            else:
                sta = np.nan

            if pd.isna(sta) is True:
                plt.imshow(np.zeros([120,160]))

            else:
                plt.imshow((sta - np.mean(sta)),
                           vmin=-0.3, vmax=0.3, cmap='jet')
                
                staAll[c,:,:] = sta

        plt.tight_layout()

        return staAll, fig
    
    else:

        lagRange = lag
        fig = plt.subplots(n_units, 5,
                           figsize=(6,np.int(np.ceil(n_units/2))), dpi=300)
        
        for c, ind in enumerate(goodcells.index):

            for lagInd, lag in enumerate(lagRange):

                sp = model_nsp[c,:].copy()
                sp = np.roll(sp,-lag)
                sta = model_vid.T @ sp
                sta = np.reshape(sta, nks)
                nsp = np.sum(sp)

                plt.subplot(n_units,5,(c*5)+lagInd + 1)

                if nsp > 0:
                    sta = sta/nsp
                else:
                    sta = np.nan

                if pd.isna(sta) is True:
                    plt.imshow(np.zeros([120,160]))

                else:
                    plt.imshow((sta-np.mean(sta)),vmin=-0.3,vmax=0.3,cmap = 'jet')
                
                if c == 0:
                    plt.title(str(np.round(lag*model_dt*1000)) + 'msec',fontsize=5)
                
                plt.axis('off')

            plt.tight_layout()

        return fig


def plot_STV(goodcells, movInterp, img_norm, worldT):
    """
    plot spike-triggererd varaince
    INPUTS
        goodcells: ephys dataframe
        movInterp: interpolator for worldcam movie
        img_norm: normalized worldcam video
        worldT: world timestamps
    OUTPUTS
        stvAll: spike triggered variance for all units
        fig: figure
    """

    n_units = len(goodcells)

    # model setup
    model_dt = 0.025
    model_t = np.arange(0, np.max(worldT), model_dt)
    model_nsp = np.zeros((n_units, len(model_t)))
    
    # get binned spike rate
    bins = np.append(model_t, model_t[-1]+model_dt)
    for i, ind in enumerate(goodcells.index):
        model_nsp[i,:], bins = np.histogram(goodcells.at[ind,'spikeT'], bins)
    
    # settting up video
    nks = np.shape(img_norm[0,:,:])
    nk = nks[0]*nks[1]
    model_vid = np.zeros((len(model_t),nk))
    
    for i in range(len(model_t)):
        model_vid[i,:] = np.reshape(movInterp(model_t[i]+model_dt/2), nk)

    model_vid = model_vid**2
    lag = 2
    stvAll = np.zeros((n_units, np.shape(img_norm)[1], np.shape(img_norm)[2]))
    
    fig = plt.subplots(int(np.ceil(n_units/10)), 10,
                       figsize=(20,np.int(np.ceil(n_units/3))), dpi=50)
    
    for c, ind in enumerate(goodcells.index):

        sp = model_nsp[c,:].copy()
        sp = np.roll(sp, -lag)
        sta = np.nan_to_num(model_vid,0).T @ sp
        sta = np.reshape(sta, nks)
        nsp = np.sum(sp)

        plt.subplot(int(np.ceil(n_units/10)), 10, c+1)

        if nsp > 0:
            sta = sta / nsp
        else:
            sta = np.nan

        if pd.isna(sta) is True:
            plt.imshow(np.zeros([120,160]))
        else:
            plt.imshow(sta - np.mean(img_norm**2,axis=0),
                       vmin=-1, vmax=1)

        stvAll[c,:,:] = sta - np.mean(img_norm**2, axis=0)

        plt.axis('off')

    plt.tight_layout()

    return stvAll, fig


def prelimRF_sort(whitenoise_directory, probe):

    temp_cfg = {
        'animal_dir': whitenoise_directory,
        'deinterlace':{
            'flip_eye_during_deinter': True,
            'flip_world_during_deinter': True
        },
        'calibration': {
            'world_checker_npz': 'E:/freely_moving_ephys/camera_calibration_params/world_checkerboard_calib.npz'
        },
        'parameters':{
            'strict_name': True,
            'outputs_and_visualization':{
                'save_nc_vids': True,
                'dwnsmpl': 0.25
            },
            'ephys':{
                'ephys_sample_rate': 30000
            }
        }
    }

    # find world files
    world_vids = glob(os.path.join(whitenoise_directory, '*WORLD.avi'))
    world_times = glob(os.path.join(whitenoise_directory, '*WORLD_BonsaiTS.csv'))
    
    # deinterlace world video
    deinterlace_data(temp_cfg, world_vids, world_times)
    
    # apply calibration parameters to world video
    calibrate_new_world_vids(temp_cfg)
    
    # organize nomenclature
    trial_units = []
    name_check = []
    path_check = []

    for avi in fme.find('*.avi', temp_cfg['animal_dir']):

        # don't use trials that have these strings in their path
        bad_list = ['plot','IR','rep11','betafpv','side_gaze']

        if temp_cfg['parameters']['strict_name'] is True:

            if all(bad not in avi for bad in bad_list):

                split_name = avi.split('_')[:-1]
                trial = '_'.join(split_name)

                path_to_trial = os.path.join(os.path.split(trial)[0])

                trial_name = os.path.split(trial)[1]
        
        elif temp_cfg['parameters']['strict_name'] is False:

            if all(bad not in avi for bad in bad_list):

                trial_path_noext = os.path.splitext(avi)[0]
                path_to_trial, trial_name_long = os.path.split(trial_path_noext)
                trial_name = '_'.join(trial_name_long.split('_')[:3])
        
        try:

            if trial_name not in name_check:
                trial_units.append([path_to_trial, trial_name])
                path_check.append(path_to_trial)
                name_check.append(trial_name)

        except UnboundLocalError:
            pass

    # there should only be one item in trial_units in this case
    # iterate into that
    for trial_unit in trial_units:

        temp_cfg['trial_path'] = trial_unit[0]
        t_name = trial_unit[1]

        # find the timestamps and video for all camera inputs
        trial_cam_csv = fme.find(('*BonsaiTS*.csv'), temp_cfg['trial_path'])
        trial_cam_avi = fme.find(('*.avi'), temp_cfg['trial_path'])

        trial_cam_csv = [x for x in trial_cam_csv if x != []]
        trial_cam_avi = [x for x in trial_cam_avi if x != []]

        # filter the list of files for the current trial to get the world view of this side
        world_csv = [i for i in trial_cam_csv if 'WORLD' in i and 'formatted' in i][0]
        world_avi = [i for i in trial_cam_avi if 'WORLD' in i and 'calib' in i][0]

        # make an xarray of timestamps without dlc points, since there aren't any for world camera
        worlddlc = h5_to_xr(pt_path=None, time_path=world_csv, view=('WORLD'), cfg=temp_cfg)
        worlddlc.name = 'WORLD_times'

        # make xarray of video frames
        if temp_cfg['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
            xr_world_frames = format_frames(world_avi, temp_cfg)
            xr_world_frames.name = 'WORLD_video'

        # merge but make sure they're not off in lenght by one value, which happens occasionally
        print('Saving nc file of world view...')

        if temp_cfg['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
            trial_world_data = safe_xr_merge([worlddlc, xr_world_frames])
            trial_world_data.to_netcdf(os.path.join(temp_cfg['trial_path'],
                                                    str(t_name+'_world.nc')),
                                                    engine='netcdf4',
                                                    encoding={'WORLD_video':{"zlib": True,
                                                                             "complevel": 4}})

        elif temp_cfg['parameters']['outputs_and_visualization']['save_nc_vids'] is False:
            worlddlc.to_netcdf(os.path.join(temp_cfg['trial_path'], str(t_name+'_world.nc')))
        
        # now start minimal ephys analysis
        print('generating ephys plots')

        pdf = PdfPages(os.path.join(whitenoise_directory, (t_name + '_prelim_wn_figures.pdf')))
        ephys_file_path = glob(os.path.join(whitenoise_directory, '*_ephys_merge.json'))[0]
        world_file_path = glob(os.path.join(whitenoise_directory, '*_world.nc'))[0]

        world_data = xr.open_dataset(world_file_path)
        world_vid_raw = np.uint8(world_data['WORLD_video'])

        # ephys data
        if '16' in probe:
            ch_count = 16
        elif '64' in probe:
            ch_count = 64
        elif '128' in probe:
            ch_count = 128

        ephys_data = pd.read_json(ephys_file_path)

        ephysT0 = ephys_data.iloc[0,12]
        worldT = world_data.timestamps - ephysT0

        ephys_data['spikeTraw'] = ephys_data['spikeT'].copy()

        # sort ephys units by channel
        ephys_data = ephys_data.sort_values(by='ch', axis=0, ascending=True)
        ephys_data = ephys_data.reset_index()
        ephys_data = ephys_data.drop('index', axis=1)

        # correct offset between ephys and other data inputs
        offset0 = 0.1
        drift_rate = -0.1/1000

        for i in ephys_data.index:
            ephys_data.at[i,'spikeT'] = np.array(ephys_data.at[i,'spikeTraw']) -            \
                                    (offset0 + np.array(ephys_data.at[i,'spikeTraw']) * drift_rate)
        
        
        # get cells labeled as good
        goodcells = ephys_data.loc[ephys_data['group']=='good']

        # occasional problem with worldcam timestamps
        if worldT[0]<-600:
            worldT = worldT + 8*60*60

        # resize worldcam to make more manageable
        world_vid = world_vid_raw.copy()

        # img correction applied to worldcam
        cam_gamma = 2

        world_norm = (world_vid / 255) ** cam_gamma

        std_im = np.std(world_norm, axis=0)

        std_im[std_im < 10/255] = 10 / 255

        img_norm = (world_norm - np.mean(world_norm, axis=0)) / std_im
        img_norm = img_norm * (std_im > 20/255)

        contrast = np.empty(worldT.size)

        for i in range(worldT.size):

            contrast[i] = np.std(img_norm[i,:,:])

        newc = scipy.interpolate.interp1d(worldT, contrast, fill_value='extrapolate')

        # bin ephys spike times as spike rate / s
        dt = 0.025
        t = np.arange(0, np.max(worldT),dt)

        ephys_data['rate'] = np.nan
        ephys_data['rate'] = ephys_data['rate'].astype(object)

        for i, ind in enumerate(ephys_data.index):

            ephys_data.at[ind,'rate'], bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
        
        ephys_data['rate'] = ephys_data['rate'] / dt

        goodcells = ephys_data.loc[ephys_data['group']=='good']

        n_units = len(goodcells)
        contrast_interp = newc(t[0:-1])

        # worldcam interp and set floor to values
        img_norm[img_norm<-2] = -2

        # added extrapolate for cases where x_new is below interpolation range
        movInterp = scipy.interpolate.interp1d(worldT, img_norm,
                                axis=0, bounds_error=False)
        
        # raster
        raster_fig = plot_spike_raster(goodcells)
        pdf.savefig()
        plt.close()
        print('making diagnostic plots')

        # plot contrast over entire video
        plt.figure()
        plt.plot(worldT[0:12000], contrast[0:12000])
        plt.xlabel('time')
        plt.ylabel('contrast')
        pdf.savefig()
        plt.close()

        # plot contrast over ~2min
        plt.figure()
        plt.plot(t[0:600], contrast_interp[0:600])
        plt.xlabel('secs')
        plt.ylabel('contrast')
        pdf.savefig()
        plt.close()

        # worldcam timing diff
        plt.figure()
        plt.plot(np.diff(worldT))
        plt.xlabel('frame')
        plt.ylabel('deltaT')
        plt.title('world cam')
        pdf.savefig()
        plt.close()

        print('getting contrast response function')
        crange = np.arange(0,1.2,0.1)
        crf_cent, crf_tuning, crf_err, crf_fig = plot_spike_rate_vs_var(contrast,
                                                                        crange,
                                                                        goodcells,
                                                                        worldT,
                                                                        t,
                                                                        'contrast')
        pdf.savefig()
        plt.close()

        print('getting spike-triggered average')
        _, STA_singlelag_fig = plot_STA(goodcells, img_norm, worldT, movInterp,
                                        ch_count, lag=2, show_title=True)
        pdf.savefig()
        plt.close()

        print('getting spike-triggered average with range in lags')
        _, STA_multilag_fig = plot_STA(goodcells, img_norm, worldT, movInterp,
                                       ch_count, lag=np.arange(-2,8,2), show_title=False)
        pdf.savefig()
        plt.close()

        # print('getting spike-triggered variance')
        # _, STV_fig = plot_STV(goodcells, movInterp, img_norm, worldT)
        # pdf.savefig()

        plt.close()
        print('closing pdf')
        pdf.close()

        print('Done!')


if __name__ == '__main__':

    prelimRF_sort()
    
    