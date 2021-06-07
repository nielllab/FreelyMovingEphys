"""
img_processing.py
"""
import cv2
import numpy as np
from tqdm import tqdm
import math
import os, argparse

from util.paths import find

def auto_contrast(config):
    """
    read in eyecam videos and apply a gamma contrast correction
    INPUTS
        config: options dict
    OUTPUTS
        None
    """
    if config['img_correction']['run_img_correction'] is True and config['img_correction']['apply_auto_gamma'] is True:
        input_list = find('*EYE.avi', config['animal_dir'])
        # iterate through input videos
        for video in input_list:
            print('correcting gamma for '+video)
            # build the save path
            head, tail = os.path.split(video)
            new_name = os.path.splitext(tail)[0] + 'deinter.avi'
            savepath = os.path.join(head, new_name)
            # write new video with gamma correction
            vid_read = cv2.VideoCapture(video)
            width = int(vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))
            num_frames = int(vid_read.get(cv2.CAP_PROP_FRAME_COUNT))
            print('num_frames', num_frames)
            # iterate through frames, find ideal gamma, apply, write frame
            for step in tqdm(range(0,num_frames)):
                ret, frame = vid_read.read()
                if not ret:
                    break
                # convert img to gray
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # compute gamma = log(mid*255)/log(mean)
                mid = 0.5
                mean = np.mean(gray)
                gamma = math.log(mid*255)/math.log(mean)
                # do gamma correction
                img_gamma1 = np.power(frame, gamma).clip(0,255).astype(np.uint8)
                # write frame
                out_vid.write(img_gamma1)
            out_vid.release()