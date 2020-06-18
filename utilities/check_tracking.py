#####################################################################################
"""
check_tracking.py of FreelyMovingEphys/utilities/

Functions to open videos...
Opens videos for each trial and plays them in one window side-by-side.

Function draw_points() comes from Elliott Abe's DLCEyeVids

last modified: June 18, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################

import cv2
import numpy as np
import xarray as xr

from utilities.data_reading import test_trial_presence

####################################################
def read_videos(current_trial, topdown_vid_path, lefteye_vid_path=None, righteye_vid_path=None, worldcam_vid_path=None):
    # the last of the three optional video feeds (world, right, and then left) will be the one to set the rescaling dimensions
    # of all videos when they are each resized prior to concatonation together into one grid of videos

    # read video in
    topdown_vid_read = cv2.VideoCapture(topdown_vid_path)
    # if this is the first video feed that exists, it will be used to rescale the dimensions of others
    td_width = int(topdown_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
    td_height = int(topdown_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
    td_num_frames = int(topdown_vid_read.get(cv2.CAP_PROP_FRAME_COUNT))

    # WORLD CAMERA READ IN
    if worldcam_vid_path is not None:
        # read video in
        worldcam_vid_read = cv2.VideoCapture(righteye_vid_path)
        # if this is the first video feed that exists, it will be used to rescale the dimensions of others
        max_width = int(worldcam_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
        max_height = int(worldcam_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
        set_size = (max_width, max_height)
    elif worldcam_vid_path is None:
        # if it doesn't exist, make a version fo it that matches the dimensions of the topdown feed
        worldcam_vid_read = topdown_vid_read

    # RIGHT CAMERA READ IN
    if righteye_vid_path is not None:
        # read video in
        righteye_vid_read = cv2.VideoCapture(righteye_vid_path)
        # if this is the first video feed that exists, it will be used to rescale the dimensions of others
        max_width = int(righteye_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
        max_height = int(righteye_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
        set_size = (max_width, max_height)
    elif righteye_vid_path is None:
        # if it doesn't exist, make a version fo it that matches the dimensions of the topdown feed
        righteye_vid_read = topdown_vid_read

    # LEFT CAMERA READ IN
    if lefteye_vid_path is not None:
        # read video in
        lefteye_vid_read = cv2.VideoCapture(lefteye_vid_path)
        # if this is the first video feed that exists, it will be used to rescale the dimensions of others
        max_width = int(lefteye_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
        max_height = int(lefteye_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
        set_size = (max_width, max_height)
    elif lefteye_vid_path is None:
        # if it doesn't exist, make a version fo it that matches the dimensions of the topdown feed
        lefteye_vid_read = topdown_vid_read

    # check frame length of each video -- are they the same?
    td_frame_count = int(topdown_vid_read.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        wc_frame_count = int(worldcam_vid_read.get(cv2.CAP_PROP_FRAME_COUNT))
    except AttributeError:
        wc_frame_count = td_num_frames
    try:
        le_frame_count = int(lefteye_vid_read.get(cv2.CAP_PROP_FRAME_COUNT))
    except AttributeError:
        le_frame_count = td_num_frames
    try:
        re_frame_count = int(righteye_vid_read.get(cv2.CAP_PROP_FRAME_COUNT))
    except AttributeError:
        re_frame_count = td_num_frames
    print('frame counts for trial ' + str(current_trial) + ': topdown=' + str(td_frame_count) + ' worldcam=' + str(wc_frame_count) + ' lefteye=' + str(le_frame_count) + ' righteye=' + str(re_frame_count))

    return topdown_vid_read, worldcam_vid_read, lefteye_vid_read, righteye_vid_read, set_size

####################################################
def plot_pts_on_video(topdown_vid_read, worldcam_vid_read=None, lefteye_vid_read=None, righteye_vid_read=None, set_size=None, topdown_data=None, left_ellipse=None, right_ellipse=None, topdown_names=None, lefteye_names=None, righteye_names=None,  trial_name=None, savepath_input=None, thresh=0.99):
    # get list of timestamps
    topdown_pd = xr.DataArray.to_pandas(topdown_data).T
    topdown_timestamp_list = topdown_pd.index.values
    leftellipse_timestamp_list = left_ellipse['time'].values
    rightellipse_timestamp_list = right_ellipse['time'].values

    # get ready to write the combined video file
    savepath = str(savepath_input) + str(trial_name) + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    main_width = (set_size[0] * 2)
    main_height = (set_size[1] * 2)

    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (main_width, main_height))

    while(1):
        # get frame of each video
        ret_re, frame_re = righteye_vid_read.read()
        ret_le, frame_le = lefteye_vid_read.read()
        ret_wc, frame_wc = worldcam_vid_read.read()
        ret_td, frame_td = topdown_vid_read.read()

        if np.shape(frame_re) != () and np.shape(frame_le) != () and np.shape(frame_wc) != () and np.shape(frame_td) != ():
            font = cv2.FONT_HERSHEY_SIMPLEX
            plot_color0 = (225, 255, 0)

            frame_td = cv2.putText(frame_td, 'topdown', (50, 50), font, 3, plot_color0, 2, cv2.LINE_4)
            frame_wc = cv2.putText(frame_wc, 'worldcam', (50, 50), font, 3, plot_color0, 2, cv2.LINE_4)
            frame_le = cv2.putText(frame_le, 'left eye', (50, 50), font, 1, plot_color0, 2, cv2.LINE_4)
            frame_re = cv2.putText(frame_re, 'right eye', (50, 50), font, 1, plot_color0, 2, cv2.LINE_4)

            # convert frame to timestamp
            try:
                current_td_time = str(topdown_timestamp_list[int(topdown_vid_read.get(cv2.CAP_PROP_POS_FRAMES))])[:-3]
                current_le_time = str(leftellipse_timestamp_list[int(lefteye_vid_read.get(cv2.CAP_PROP_POS_FRAMES))])[:-3]
                current_re_time = str(rightellipse_timestamp_list[int(righteye_vid_read.get(cv2.CAP_PROP_POS_FRAMES))])[:-3]
            except IndexError:
                break

            # get topdown points for this timestamp and plot them on the current frame
            try:
                for k in range(0, 30, 3):
                    topdownTS = topdown_data.sel(time=current_td_time)
                    try:
                        td_pts_x = topdownTS.isel(point_loc=k)
                        td_pts_y = topdownTS.isel(point_loc=k+1)
                        center_xy = (int(td_pts_x), int(td_pts_y))
                        print(center_xy)
                        if k == 0:
                            # plot them on the fresh topdown frame
                            pt_frame_td = cv2.circle(frame_td, center_xy, 10, plot_color0, -1)
                        elif k >= 3:
                            # plot them on the topdown frame with all past topdown points
                            pt_frame_td = cv2.circle(pt_frame_td, center_xy, 10, plot_color0, -1)
                    except ValueError:
                        print('a NAN stopped topdown from plotting')
                        pt_frame_td = frame_td
            except KeyError:
                print('ran into key error for time: ' + str(current_le_time))
                pt_frame_td = frame_td

            if left_ellipse is not None:
                try:
                    leftellipseTS = left_ellipse.sel(time=current_le_time)
                    try:
                        ellipse_center = (int(leftellipseTS['cam_center_x'].values), int(leftellipseTS['cam_center_y'].values))
                        ellipse_longaxis = int(leftellipseTS.sel(ellipse_params='longaxis_all').values)
                        ellipse_shortaxis = int(leftellipseTS.sel(ellipse_params='shortaxis_all').values)
                        ellipse_axes = (ellipse_longaxis, ellipse_shortaxis)
                        ellipse_theta = int(leftellipseTS.sel(ellipse_params='theta').values)
                        ellipse_phi = int(leftellipseTS.sel(ellipse_params='phi').values)
                        plot_lellipse = cv2.ellipse(frame_le, ellipse_center, ellipse_axes, ellipse_theta, 0, 360, plot_color0, 4)
                    except ValueError:
                        print('a NAN stopped from plotting')
                        plot_lellipse = frame_le
                except KeyError:
                    print('ran into key error for time: ' + str(current_le_time))
                    plot_lellipse = frame_le

            if right_ellipse is not None:
                try:
                    rightellipseTS = right_ellipse.sel(time=current_re_time)
                    try:
                        ellipse_center = (int(rightellipseTS['cam_center_x'].values), int(rightellipseTS['cam_center_y'].values))
                        ellipse_longaxis = int(rightellipseTS.sel(ellipse_params='longaxis_all').values)
                        ellipse_shortaxis = int(rightellipseTS.sel(ellipse_params='shortaxis_all').values)
                        ellipse_axes = (ellipse_longaxis, ellipse_shortaxis)
                        ellipse_theta = int(rightellipseTS.sel(ellipse_params='theta').values)
                        ellipse_phi = int(rightellipseTS.sel(ellipse_params='phi').values)
                        plot_rellipse = cv2.ellipse(frame_re, ellipse_center, ellipse_axes, ellipse_theta, 0, 360, plot_color0, 4)
                    except ValueError:
                        print('a NAN stopped from plotting')
                        plot_rellipse = frame_re
                except KeyError:
                    print('ran into key error for time: ' + str(current_le_time))
                    plot_rellipse = frame_re

            # resize videos to match
            frame_td_resized = cv2.resize(pt_frame_td, set_size)
            frame_wc_resized = cv2.resize(frame_wc, set_size)
            frame_re_resized = cv2.resize(plot_rellipse, set_size)
            frame_le_resized = cv2.resize(plot_lellipse, set_size)

            # stitch all videos together, side-by-side
            top_row_vids = np.concatenate((frame_td_resized, frame_wc_resized), axis=1)
            bottom_row_vids = np.concatenate((frame_le_resized, frame_re_resized), axis=1)
            all_vids = np.concatenate((top_row_vids, bottom_row_vids), axis=0)

            # display the frame with the window titled with the trial's name
            cv2.imshow(trial_name, all_vids)
            # save the frame into out_vid to be saved as a file
            out_vid.write(all_vids)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_vid.release()
    cv2.destroyAllWindows()

####################################################
def parse_data_for_playback(savepath_input, trial_list, preened_topdown, left_ellipse=None, right_ellipse=None,
                            topdown_vid_list=None, lefteye_vid_list=None, righteye_vid_list=None, worldcam_vid_list=None,
                            topdown_names=None, lefteye_names=None, righteye_names=None, n_td_pts=8, n_le_pts=8, n_re_pts=8):
    # run through each trial individually
    for trial_num in range(0, len(trial_list)):
        # get the name of the current trial
        current_trial = trial_list[trial_num]
        mouse_key = current_trial[6:11]
        trial_key = current_trial[18:]
        print('mouse_key = ' + mouse_key + ' trial_key = ' + trial_key)

        # find video files
        righteye_vids = [i for i in righteye_vid_list if mouse_key and trial_key in i]
        lefteye_vids = [i for i in lefteye_vid_list if mouse_key and trial_key in i]
        topdown_vids = [i for i in topdown_vid_list if mouse_key and trial_key in i]
        worldcam_vids = [i for i in worldcam_vid_list if mouse_key and trial_key in i]
        # this turns it into a string from a list of only one string item
        topdown_vid = topdown_vids[0]
        try:
            righteye_vid = righteye_vids[0]
        except IndexError:
            righteye_vid = None
        try:
            worldcam_vid = worldcam_vids[0]
        except IndexError:
            worldcam_vid = None
        try:
            lefteye_vid = lefteye_vids[0]
        except IndexError:
            lefteye_vid = None

        test_trial_td = test_trial_presence(preened_topdown, current_trial)
        if test_trial_td is True:
            topdown_data = preened_topdown.sel(trial=current_trial)
            test_trial_le = test_trial_presence(left_ellipse, current_trial)
            test_trial_re = test_trial_presence(right_ellipse, current_trial)

            topdown_data = xr.DataArray.dropna(topdown_data, dim='time', thresh=20)

            # test each case of absence/presence of DLC-based ellipse data for eyes
            # NOT looking at presence/absence of videos, that will be dealt with in function read_videos()
            if test_trial_le is True and test_trial_re is True:
                # get the left and right data for the current trial
                leftellipse_data = left_ellipse.sel(trial=current_trial)
                rightellipse_data = right_ellipse.sel(trial=current_trial)
                left_timestart = leftellipse_data['time_start'].values
                left_timeend = leftellipse_data['time_end'].values
                right_timestart = rightellipse_data['time_start'].values
                right_timeend = rightellipse_data['time_end'].values
                leftellipse_data = leftellipse_data.sel(time=slice(left_timestart, left_timeend))
                rightellipse_data = rightellipse_data.sel(time=slice(right_timestart, right_timeend))
                # then, read in the video files to get them ready to be plotted on
                topdown_vid_read, worldcam_vid_read, lefteye_vid_read, righteye_vid_read, set_size = read_videos(current_trial, topdown_vid, lefteye_vid, righteye_vid, worldcam_vid)
                # then, plot points on them on the videos, display them in a window, and save them out
                plot_pts_on_video(topdown_vid_read, worldcam_vid_read, lefteye_vid_read, righteye_vid_read, set_size,
                                  topdown_data, leftellipse_data, rightellipse_data, topdown_names, lefteye_names,
                                  righteye_names, current_trial, savepath_input)

            elif test_trial_le is True and test_trial_re is False:
                leftellipse_data = left_ellipse.sel(trial=current_trial)
                left_timestart = leftellipse_data['time_start'].values
                left_timeend = leftellipse_data['time_end'].values
                leftellipse_data = leftellipse_data.sel(time=slice(left_timestart, left_timeend))
                # then, read in the video files to get them ready to be plotted on
                topdown_vid_read, worldcam_vid_read, lefteye_vid_read, righteye_vid_read, set_size = read_videos(current_trial, topdown_vid, lefteye_vid, worldcam_vid_path=worldcam_vid)
                # then, plot points on them on the videos, display them in a window, and save them out
                plot_pts_on_video(topdown_vid_read, worldcam_vid_read, lefteye_vid_read, righteye_vid_read, set_size,
                                  topdown_data, leftellipse_data, topdown_names=topdown_names, lefteye_names=lefteye_names,
                                  current_trial=current_trial, savepath_input=savepath_input)

            # TO DO: Finish making these modular and not take in the data that won't exist for their case
            elif test_trial_le is False and test_trial_re is True:
                rightellipse_data = right_ellipse.sel(trial=current_trial)
                right_timestart = rightellipse_data['time_start'].values
                right_timeend = rightellipse_data['time_end'].values
                rightellipse_data = rightellipse_data.sel(time=slice(right_timestart, right_timeend))
                # then, read in the video files to get them ready to be plotted on
                topdown_vid_read, worldcam_vid_read, lefteye_vid_read, righteye_vid_read, set_size = read_videos(current_trial, topdown_vid, righteye_vid_path=righteye_vid, worldcam_vid_path=worldcam_vid)
                # then, plot points on them on the videos, display them in a window, and save them out
                plot_pts_on_video(topdown_vid_read, worldcam_vid_read, lefteye_vid_read, righteye_vid_read, set_size,
                                  topdown_data, leftellipse_data, rightellipse_data=rightellipse_data, topdown_names=topdown_names,
                                  righteye_names=righteye_names, current_trial=current_trial, savepath_input=savepath_input)

            elif test_trial_le is False and test_trial_re is False:
                # then, read in the video files to get them ready to be plotted on
                topdown_vid_read, worldcam_vid_read, lefteye_vid_read, righteye_vid_read, set_size = read_videos(current_trial, topdown_vid, worldcam_vid)
                # then, plot points on them on the videos, display them in a window, and save them out
                plot_pts_on_video(topdown_vid_read, worldcam_vid_read, lefteye_vid_read, righteye_vid_read, set_size,
                                  topdown_data, topdown_names=topdown_names, current_trial=current_trial, savepath_input=savepath_input)