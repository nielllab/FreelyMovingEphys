"""
FreelyMovingEphys plotting on top of videos of any source
plot_video.py

Last modified July 27, 2020
"""

# package imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from skimage import measure
from itertools import product

# plot points on topdown video as a saftey check, then save as .avi
def check_topdown_tracking(trial_name, vid_path, savepath, dlc_data=None, head_ang=None, vext=None):

    # read topdown video in
    vidread = cv2.VideoCapture(vid_path)
    width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # setup the file to save out of this
    savepath = str(savepath) + '/' + str(trial_name) + '/' + str(trial_name) + '_' + vext + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (width, height))

    # set colors
    plot_color0 = (225, 255, 0)
    plot_color1 = (0, 255, 255)

    while (1):
        # read the frame for this pass through while loop
        ret_td, frame_td = vidread.read()

        if not ret_td:
            break

        if dlc_data is not None:
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

        elif dlc_data is None:
            out_vid.write(frame_td)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_vid.release()
    cv2.destroyAllWindows()

# plot points and ellipse on eye video as a saftey check, then save as .avi
def check_eye_tracking(trial_name, vid_path, savepath, dlc_data=None, ell_data=None, vext=None):

    # read topdown video in
    vidread = cv2.VideoCapture(vid_path)
    width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # setup the file to save out of this
    savepath = str(savepath) + '/' + str(trial_name) + '/' + str(trial_name) + '_' + vext + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (width, height))

    # set colors
    plot_color0 = (225, 255, 0)
    plot_color1 = (0, 255, 255)

    while (1):
        # read the frame for this pass through while loop
        ret_le, frame_le = vidread.read()

        if not ret_le:
            break

        if dlc_data is not None and ell_data is not None:
            # get current frame number to be displayed, so that it can be used to slice DLC data
            try:
                frame_time = vidread.get(cv2.CAP_PROP_POS_FRAMES)
                ell_data_thistime = ell_data.sel(frame=frame_time)
                dlc_data_thistime = dlc_data.sel(frame=frame_time)
                ellipseparams = np.empty((0,5))
                thetas = np.linspace(-np.pi, np.pi, 50)
                # get out ellipse parameters and plot them on the video
                ellipse_axes = (int(ell_data_thistime.sel(ellipse_params='longaxis').values), int(ell_data_thistime.sel(ellipse_params='shortaxis').values))
                ellipse_phi = int(ell_data_thistime.sel(ellipse_params='phi').values)
                dlc_names = dlc_data_thistime.coords['point_loc'].values
                dlc_x_names = [name for name in dlc_names if '_x' in name]
                dlc_y_names = [name for name in dlc_names if '_y' in name]
                # get nanmean of x and y in (y, x) tuple as center of ellipse
                x_val = []; y_val = []
                for ptpairnum in range(0, len(dlc_x_names)):
                    x_val.append(dlc_data_thistime.sel(point_loc=dlc_x_names[ptpairnum]).values)
                    y_val.append(dlc_data_thistime.sel(point_loc=dlc_y_names[ptpairnum]).values)
                mean_cent = (int(np.nanmean(x_val)), int(np.nanmean(y_val)))
                frame_le = cv2.ellipse(frame_le, mean_cent, ellipse_axes, ellipse_phi, 0, 360, plot_color0, 4)
            except (ValueError, KeyError) as e:
                pass

            # get out the DLC points and plot them on the video
            try:
                leftptsTS = dlc_data.sel(frame=frame_time)
                for k in range(0, 24, 3):
                    pt_cent = (int(leftptsTS.isel(point_loc=k).values), int(leftptsTS.isel(point_loc=k+1).values))
                    frame_le = cv2.circle(frame_le, pt_cent, 4, plot_color1, -1)
            except (ValueError, KeyError) as e:
                pass

        elif dlc_data is None or ell_data is None:
            pass
            
        out_vid.write(frame_le)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_vid.release()
    cv2.destroyAllWindows()
