# Quick script that opens videos for each trial and plays them in one window side-by-side.

import os
import cv2
import numpy as np
from glob import glob
from itertools import product

####################################################
def set_colors(n_pts):
    output_colors = np.column_stack((np.linspace(255, 0, num=n_pts, dtype=np.int),
                                 np.linspace(0, 255, num=n_pts, dtype=np.int),
                                 np.zeros((n_pts), dtype=np.int)))
    return output_colors

####################################################
def draw_points(frame, x, y, ptsize):
    color = set_colors(1)
    point_adds = product(range(-ptsize,ptsize), range(-ptsize,ptsize))
    for pt in point_adds:
        try:
            frame[x+pt[0], y+pt[1]] = color
        except IndexError:
            pass
    return frame

####################################################

main_path = '/Users/dylanmartins/data/Niell/PreyCapture/WorldCamCohort/J475c/112219/Approach/'

topdown_vid_list = glob(main_path + '*top*.avi')
righteye_vid_list = glob(main_path + '*eye1r*.avi')
lefteye_vid_list = glob(main_path + '*eye2l*.avi')
worldcam_vid_list = glob(main_path + '*world*.avi')

loop_count = 0
for topdown_vid in topdown_vid_list:
    if loop_count < 1:
        # get unique keys out of the topdown video's file name and path
        split_path = os.path.split(topdown_vid)
        file_name = split_path[1]
        mouse_key = file_name[0:5]
        trial_key = file_name[17:28]

        # pull the right associated files out of the glob-created lists
        lefteye_vids = [i for i in lefteye_vid_list if mouse_key and trial_key in i]
        lefteye_vid = lefteye_vids[0]  # this turns it into a string from a list of only one string item
        worldcam_vids = [i for i in worldcam_vid_list if mouse_key and trial_key in i]
        worldcam_vid = worldcam_vids[0]

        # read all videos in
        lefteye_vid_read = cv2.VideoCapture(lefteye_vid)
        topdown_vid_read = cv2.VideoCapture(topdown_vid)
        worldcam_vid_read = cv2.VideoCapture(worldcam_vid)

        # get pixel dimensions of lefteye video, which is the smallest in dimensions out of the other videos
        max_width = int(lefteye_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
        max_height = int(lefteye_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
        set_size = (max_width, max_height)

        while(1):
            ret_le, frame_le = lefteye_vid_read.read()
            ret_wc, frame_wc = worldcam_vid_read.read()
            ret_td, frame_td = topdown_vid_read.read()

            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_frame_td = cv2.putText(frame_td, 'topdown', (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_4)
            txt_frame_wc = cv2.putText(frame_wc, 'worldcam', (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_4)

            pt_frame_td = draw_points(txt_frame_td, 30, 30, 5)

            # resize videos to match
            frame_td_resized = cv2.resize(pt_frame_td, set_size)
            frame_wc_resized = cv2.resize(txt_frame_wc, set_size)

            # stitch all videos together, side-by-side
            all_vids = np.concatenate((frame_td_resized, frame_wc_resized, frame_le), axis=1)

            cv2.imshow('frame', all_vids)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        all_vids.release()
        cv2.destroyAllWindows()
        loop_count = loop_count + 1
