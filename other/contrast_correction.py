"""
contrast_correction.py
"""
import cv2
import numpy as np
from tqdm import tqdm
import math
from glob import glob
import os, argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--readdir', type=str)
    parser.add_argument('--savedir', type=str)
    args = parser.parse_args()
    return args

def main(readdir, savedir):
    input_list = glob(os.path.join(readdir, '*.avi'))

    for video in input_list:
        # build the save path
        head, tail = os.path.split(video)
        split_name = tail.split('_')
        new_name = '_'.join(split_name[:3])+'Gammacor_'+'_'.join(split_name[3:])
        savepath = os.path.join(savedir, new_name)

        # write new video with gamma correction
        vid_read = cv2.VideoCapture(video)
        width = int(vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))

        num_frames = int(vid_read.get(cv2.CAP_PROP_FRAME_COUNT))

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

            out_vid.write(img_gamma1)
        out_vid.release()

if __name__ == '__main__':
    args = get_args()
    main(args.readdir, args.savedir)