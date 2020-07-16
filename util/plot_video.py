"""
FreelyMovingEphys plotting on top of videos of any source
plot_video.py

Last modified July 15, 2020
"""

# package imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from skimage import measure

# open video from camera of any type passed in, plot its DLC points over the video feed, and save out as an .avi format
def check_tracking(trial_name, camtype, vid_path, savepath, dlc_data=None, ell_data=None, head_ang=None, vext=None):

    # read topdown video in
    vidread = cv2.VideoCapture(vid_path)
    width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # setup the file to save out of this
    savepath = str(savepath) + '/' + str(trial_name) + '/' + str(trial_name) + '_' + vext + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (width, height))

    # small aesthetic things to set
    plot_color0 = (225, 255, 0)
    plot_color1 = (0, 255, 255)

    if camtype == 't':
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

    elif camtype == 'e':
        while (1):
            # read the frame for this pass through while loop
            ret_le, frame_le = vidread.read()

            if not ret_le:
                break

            if dlc_data is not None and ell_data is not None:
                # get current frame number to be displayed, so that it can be used to slice DLC data
                frame_time = vidread.get(cv2.CAP_PROP_POS_FRAMES)

                try:
                    leftellipseTS = ell_data.sel(frame=frame_time)
                    try:
                        # get out ellipse parameters and plot them on the video
                        emod = measure.EllipseModel()
                        ellipse_center = (int(leftellipseTS['cam_center_y'].values), int(leftellipseTS['cam_center_x'].values))
                        ellipse_longaxis = int(leftellipseTS.sel(ellipse_params='longaxis_all').values)
                        ellipse_shortaxis = int(leftellipseTS.sel(ellipse_params='shortaxis_all').values)
                        ellipse_axes = (ellipse_longaxis, ellipse_shortaxis)
                        ellipse_theta = int(leftellipseTS.sel(ellipse_params='theta').values)
                        ellipse_phi = int(leftellipseTS.sel(ellipse_params='phi').values)
                        e_points = emod.predict_xy(ellipse_theta).astype(np.int)
                        for k in range(0, np.size(e_points, axis=1)):
                            ptk = e_points[k]
                            if k == 0:
                                pt_frame_le = cv2.circle(frame_le, ptk, 2, plot_color0, -1)
                            elif k >= 1:
                                pt_frame_le = cv2.circle(pt_frame_le, ptk, 2, plot_color0, -1)
                    except TypeError:
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

            elif dlc_data is None or ell_data is None:
                out_vid.write(frame_le)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out_vid.release()
        cv2.destroyAllWindows()

    elif camtype == 'w':
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
