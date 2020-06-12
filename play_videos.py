import os, fnmatch
import cv2
import pandas as pd
import numpy as np
from skimage import draw, measure
from skvideo import io
from itertools import product
# from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
# from numba import jit
from scipy import stats
from skimage.feature import match_template
import matplotlib as mpl
from glob import glob

####################################

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
        lefteye_vid = lefteye_vids[0] # this turns it into a string from a list of only one string item
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

            # resize videos to match
            frame_td_resized = cv2.resize(frame_td, set_size)
            frame_wc_resized = cv2.resize(frame_wc, set_size)

            # stitch all videos together, side-by-side
            all_vids = np.concatenate((frame_td_resized, frame_wc_resized, frame_le), axis=1)

            cv2.imshow('frame', all_vids)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        all_vids.release()
        cv2.destroyAllWindows()
        loop_count = loop_count + 1
