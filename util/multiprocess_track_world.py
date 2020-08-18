"""
FreelyMovingEphys world tracking utilities using parallel processing
multiprocess_track_world.py

Last modified August 14, 2020
"""

# package imports
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cv2
from scipy import signal
from scipy.optimize import curve_fit
import scipy.stats
import time
import subprocess as sp
import multiprocessing as mp
import argparse

# module imports
from util.track_world import nanxcorr

# desc.
def multiprocess_find_pupil_rotation(group_number):
    cap = cv.VideoCapture(file_name)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_jump_unit * group_number)
    # get height, width and frame count of the video
    width, height = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    no_of_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    proc_frames = 0

    try:
        while(1):
            ret, eye_frame = cap.read()
            if not ret:
                break
            eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)



            proc_frames += 1
    except:
        # Release resources
        cap.release()
    cap.release()
    return rfit


# desc.
def multiprocess_run():
    print("Video processing using {} processes...".format(num_processes))
    start_time = time.time()

    p = mp.Pool(num_processes)
    p.map(process_video_multiprocessing, range(num_processes))

    combine_output_files(num_processes)

    end_time = time.time()

    total_processing_time = end_time - start_time
    print("Time taken: {}".format(total_processing_time))
    print("FPS : {}".format(frame_count/total_processing_time))

##########################
width, height, frame_count = get_video_frame_details(vid_path)
print("Video frame count = {}".format(frame_count))
print("Width = {}, Height = {}".format(width, height))
num_processes = mp.cpu_count()
print("Number of CPUs: " + str(num_processes))
frame_jump_unit =  frame_count // num_processes
multiprocess_run()
