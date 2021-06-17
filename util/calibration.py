"""
calibration.py

video calibration utilities
"""
import numpy as np
import pandas as pd
import cv2
import os, json
import xarray as xr
from tqdm import tqdm

from util.dlc import run_DLC_on_LED
from util.paths import find

def get_checkerboard_calib(checker_vid_path, savepath):
    """
    get calibration parameters for a camera using a video of the checkerboard pattern
    requires both a path to read the checkerboard video from and a save path
    INPUTS
        checkerboard_vid_path: file path (not a directory)
        savepath: specific file path, including the file name, that the data will be saved to
    OUTPUTS
        None
    camera properties will be saved to file as a .npz using the provided savepath
    """
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
    # iterate through frames
    print('getting distortion out of each frame')
    for step in tqdm(range(0,int(calib_vid.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # open frame
        ret, img = calib_vid.read()
        # make sure the frame is read in correctly
        if not ret:
            break
        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # if found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
    # calibrate the camera (this is a little slow)
    print('calculating calibration correction paramters')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # format as xarray and save the file
    np.savez(savepath, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

def undistort_vid(vidpath, savepath, mtx, dist, rvecs, tvecs):
    """
    undistort novel videos using provided camera calibration properties
    INPUTS
        vidpath: path to the video file
        savepath: file path (not a directory) into which the undistorted video will be saved
        mtx: camera matrix
        dist: distortion coefficients
        rvecs: rotation vectors
        tvecs: translation vectors
    OUTPUTS
        None
    if vidpath and savepath are the same filename, the file will be overwritten
    saves a new copy of the video, after it has been undistorted
    """
    # open the video
    cap = cv2.VideoCapture(vidpath)
    # setup the file writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    # iterate through all frames
    print('undistorting video')
    for step in tqdm(range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # open frame and check that it opens correctly
        ret, frame = cap.read()
        if not ret:
            break
        # run opencv undistortion function
        undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)
        # write the frame to the video
        out_vid.write(undist_frame)
    out_vid.release()

def get_calibration_params(config):
    """
    get out the parameters that can be used to calibrate videos
    looks for the checkerboard recording
    INPUTS
        config: options dictionary
    OUTPUTS
        None
    """
    W_savepath = config['calibration']['world_checker_npz']
    T_savepath = config['calibration']['top_checker_npz']
    # world
    world_vid_path = config['calibration']['world_checker_vid']
    get_checkerboard_calib(world_vid_path, W_savepath)
    # top
    top_vid_path = config['calibration']['top_checker_vid']
    get_checkerboard_calib(top_vid_path, T_savepath)

def calibrate_new_world_vids(config):
    """
    calibrate novel world videos using previously genreated .npy of parameters
    INPUTS
        config: options dictionary
    OUTPUTS
        None
    """
    # load the parameters
    checker_in = np.load(config['calibration']['world_checker_npz'])
    # unpack camera properties
    mtx = checker_in['mtx']; dist = checker_in['dist']; rvecs = checker_in['rvecs']; tvecs = checker_in['tvecs']
    # iterate through eye videos and save out a copy which has had distortions removed
    world_list = find('*WORLDdeinter*.avi', config['animal_dir'])
    for world_vid in world_list:
        if 'plot' not in world_vid:
            savepath = '_'.join(world_vid.split('_')[:-1])+'_WORLDcalib.avi'
            undistort_vid(world_vid, savepath, mtx, dist, rvecs, tvecs)

def calibrate_new_top_vids(config):
    """
    calibrate novel top videos using previously genreated .npy of parameters
    INPUTS
        config: options dictionary
    OUTPUTS
        None
    """
    # load the parameters
    checker_in = np.load(config['calibration']['top_checker_npz'])
    # unpack camera properties
    mtx = checker_in['mtx']; dist = checker_in['dist']; rvecs = checker_in['rvecs']; tvecs = checker_in['tvecs']
    top_list = find('*TOP1*.avi', config['animal_dir'])
    for top_vid in top_list:
        if 'plot' not in top_vid:
            savepath = '_'.join(top_vid.split('_')[:-1])+'_TOP1calib.avi'
            undistort_vid(top_vid, savepath, mtx, dist, rvecs, tvecs)
