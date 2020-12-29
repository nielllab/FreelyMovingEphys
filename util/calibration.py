"""
calibration.py

calibration steps of the preprocessing pipeline

Dec. 28, 2020
"""
# package imports
import numpy as np
import pandas as pd
import cv2
import os, json
import xarray as xr
from tqdm import tqdm
# module imports
from util.dlc import run_DLC_on_LED
from util.paths import find

# get calibration parameters for checkerboard video
def get_checkerboard_calib(checker_vid_path, save_path, cam_save_name):
    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #prepare object points
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # read in file path of video
    calib_vid = cv2.VideoCapture(checker_vid_path)

    print('getting calibration parameters for each frame')
    for step in tqdm(range(0,int(calib_vid.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # open frame
        ret, img = calib_vid.read()
        if not ret:
            break
        # to b/w
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
    print('calibrating camera')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # format as xarray
    print('formatting caibration parameters together')
    save_name = os.path.join(save_path, cam_save_name+'_checkerboard_calib.npz')
    np.savez(save_name, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print('checkerboard calibration file saved as ' + save_name)

# save out an undistorted video
def undistort_vid(vid, savepath, mtx, dist, rvecs, tvecs):
    print('undistorting video ' + vid)
    cap = cv2.VideoCapture(vid)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))
    for step in tqdm(range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break
        undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)
        out_vid.write(frame)
    out_vid.release()
    print('new video saved at '+savepath)

# get out the parameters that can be used to calibrate videos
# looks for the checkerboard recording
def get_calibration_params(config):
    calib_config = config['calibration']
    # world first
    world_vid_path = calib_config['world_checker_vid']
    get_checkerboard_calib(world_vid_path, config['save_path'], 'world')
    # then top
    top_vid_path = calib_config['top_checker_vid']
    get_checkerboard_calib(top_vid_path, config['save_path'], 'top1')

# calibrate novel world videos using previously genreated .npy of parameters
def calibrate_new_world_vids(config):
    # load the parameters
    calib_config = config['calibration']
    checker_in = pd.DataFrame(np.load(calib_config['world_checker_npz']))
    mtx = checker_in['mtx']; dist = checker_in['dist']; rvecs = checker_in['rvecs']; tvecs = checker_in['tvecs']
    # iterate through eye videos and save out a copy which has had distortions removed
    world_list = find('*WORLDdeinter*.avi', config['save_path'])
    for world_vid in world_list:
        savepath = '_'.join(world_vid.split('_')[:-1])+'_WORLDcalib.avi'
        undistort_vid(world_vid, savepath, mtx, dist, rvecs, tvecs)

# calibrate novel top videos using previously genreated .npy of parameters
def calibrate_new_top_vids(config):
    calib_config = config['calibration']
    checker_in = pd.DataFrame(np.load(calib_config['top1_checker_npz']))
    mtx = checker_in['mtx']; dist = checker_in['dist']; rvecs = checker_in['rvecs']; tvecs = checker_in['tvecs']
    top_list = find('*TOP1*.avi', config['save_path'])
    for top_vid in top_list:
        save_path = '_'.join(world_vid.split('_')[:-1])+'_TOP1calib.avi'
        undistort_vid(top_vid, savepath, mtx, dist, rvecs, tvecs)