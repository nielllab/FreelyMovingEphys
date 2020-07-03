#####################################################################################
"""
check_ind_tracking.py

Functions to open .avi video and plot on this video the DeepLabCut points or ellipse
parameters for the left eye, right eye, worldcam, or topdown camera in the format of
an xarray DataArrays.
Note that this script, check_ind_tracking.py runs through one input at a time and saves
out a separate .mp4 video each time it is called, while check_all_tracking.py connects
the videos and DLC points/parameters for an entire trial and saves out one video for
each trial.

last modified: July 03, 2020
"""
#####################################################################################
# import packages
import cv2
import numpy as np
import xarray as xr

# import functions
from utilities.data_reading import test_trial_presence
from utilities.time_management import find_first_time
####################################################
def plot_pts_on_vid(trial_name, camtype, vid_path, savepath, dlc_data=None, ell_data=None):
    '''
    Open video from any camera passed in plot its DLC points over the video feed saved out as an .mp4 file.
    '''

    # read topdown video in
    vidread = cv2.VideoCapture(vid_path)
    width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # setup the file to save out of this
    savepath = str(savepath) + '/' + str(trial_name) + '/' + str(trial_name) + '_' + str(camtype) + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (width, height))

    # small aesthetic things to set
    plot_color0 = (225, 255, 0)
    plot_color1 = (0, 255, 255)

    if camtype == 't':
        print('plotting points on topdown view')
        while (1):
            # read the frame for this pass through while loop
            ret_td, frame_td = vidread.read()

            if not ret_td:
                break

            # get current frame number to be displayed, so that it can be used to slice DLC data
            frame_time = vidread.get(cv2.CAP_PROP_POS_FRAMES)

            try:
                for k in range(0, 30, 3):
                    topdownTS = dlc_data.sel(frame=frame_time)
                    try:
                        td_pts_x = topdownTS.isel(point_loc=k)
                        td_pts_y = topdownTS.isel(point_loc=k + 1)
                        center_xy = (int(td_pts_x), int(td_pts_y))
                        if k == 0:
                            # plot them on the fresh topdown frame
                            pt_frame_td = cv2.circle(frame_td, center_xy, 6, plot_color0, -1)
                        elif k >= 3:
                            # plot them on the topdown frame with all past topdown points
                            pt_frame_td = cv2.circle(pt_frame_td, center_xy, 6, plot_color0, -1)
                    except ValueError:
                        pt_frame_td = frame_td
            except KeyError:
                pt_frame_td = frame_td

            out_vid.write(pt_frame_td)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out_vid.release()
        cv2.destroyAllWindows()

    elif camtype == 'l':
        print('plotting ellipse and points on left eye view')
        while (1):
            # read the frame for this pass through while loop
            ret_le, frame_le = vidread.read()

            if not ret_le:
                break

            # get current frame number to be displayed, so that it can be used to slice DLC data
            frame_time = vidread.get(cv2.CAP_PROP_POS_FRAMES)

            try:
                leftellipseTS = ell_data.sel(frame=frame_time)
                try:
                    # get out ellipse parameters and plot them on the video
                    ellipse_center = (int(leftellipseTS['cam_center_x'].values), int(leftellipseTS['cam_center_y'].values))
                    ellipse_longaxis = int(leftellipseTS.sel(ellipse_params='longaxis_all').values)
                    ellipse_shortaxis = int(leftellipseTS.sel(ellipse_params='shortaxis_all').values)
                    ellipse_axes = (ellipse_longaxis, ellipse_shortaxis)
                    ellipse_theta = int(leftellipseTS.sel(ellipse_params='theta').values)
                    ellipse_phi = int(leftellipseTS.sel(ellipse_params='phi').values)
                    plot_lellipse = cv2.ellipse(frame_le, ellipse_center, ellipse_axes, ellipse_theta, 0, 360, plot_color0, 4)
                except ValueError:
                    plot_lellipse = frame_le

                for k in range(0, 24, 3):
                    try:
                        # get out the DLC points and plot them on the video
                        leftptsTS = dlc_data.sel(time=frame_time)
                        le_pts_x = leftptsTS.isel(point_loc=k)
                        le_pts_y = leftptsTS.isel(point_loc=k + 1)
                        le_center_xy = (int(le_pts_x), int(le_pts_y))
                        if k == 0:
                            # plot them on the fresh lefteye frame
                            plot_lellipse = cv2.circle(plot_lellipse, le_center_xy, 6, plot_color1, -1)
                        elif k >= 3:
                            # plot them on the lefteye frame with all past lefteye points
                            plot_lellipse = cv2.circle(plot_lellipse, le_center_xy, 6, plot_color1, -1)
                    except ValueError:
                        # print('ignoring ValueError raised by left eye DLC points')
                        pass

            except KeyError:
                plot_lellipse = plot_lellipse

            out_vid.write(plot_lellipse)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out_vid.release()
        cv2.destroyAllWindows()

    elif camtype == 'r':
        print('plotting ellipse and points on right eye view')
        while (1):
            # read the frame for this pass through while loop
            ret_re, frame_re = vidread.read()

            if not ret_re:
                break

            # get current frame number to be displayed, so that it can be used to slice DLC data
            frame_time = vidread.get(cv2.CAP_PROP_POS_FRAMES)

            try:
                rightellipseTS = ell_data.sel(frame=frame_time)
                try:
                    # get out ellipse parameters and plot them on the video
                    ellipse_center = (int(rightellipseTS['cam_center_x'].values), int(rightellipseTS['cam_center_y'].values))
                    ellipse_longaxis = int(rightellipseTS.sel(ellipse_params='longaxis_all').values)
                    ellipse_shortaxis = int(rightellipseTS.sel(ellipse_params='shortaxis_all').values)
                    ellipse_axes = (ellipse_longaxis, ellipse_shortaxis)
                    ellipse_theta = int(rightellipseTS.sel(ellipse_params='theta').values)
                    ellipse_phi = int(rightellipseTS.sel(ellipse_params='phi').values)
                    plot_rellipse = cv2.ellipse(frame_re, ellipse_center, ellipse_axes, ellipse_theta, 0, 360,
                                                plot_color0, 4)
                except ValueError:
                    plot_rellipse = frame_re

                for k in range(0, 24, 3):
                    try:
                        # get out the DLC points and plot them on the video
                        rightptsTS = dlc_data.sel(time=frame_time)
                        le_pts_x = rightptsTS.isel(point_loc=k)
                        le_pts_y = rightptsTS.isel(point_loc=k + 1)
                        le_center_xy = (int(le_pts_x), int(le_pts_y))
                        if k == 0:
                            # plot them on the fresh righteye frame
                            plot_rellipse = cv2.circle(plot_rellipse, le_center_xy, 6, plot_color1, -1)
                        elif k >= 3:
                            # plot them on the righteye frame with all past lefteye points
                            plot_rellipse = cv2.circle(plot_rellipse, le_center_xy, 6, plot_color1, -1)
                    except ValueError:
                        # print('ignoring ValueError raised by right eye DLC points')
                        pass

            except KeyError:
                plot_rellipse = plot_rellipse

            out_vid.write(plot_rellipse)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out_vid.release()
        cv2.destroyAllWindows()

    elif camtype == 'w':
        print('writing worldcam view')
        while (1):
            # read the frame for this pass through while loop
            ret_wc, frame_wc = vidread.read()

            if not ret_wc:
                break

            out_vid.write(frame_wc)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out_vid.release()
        cv2.destroyAllWindows()

    else:
        print('unknown camtype argument... exiting')
