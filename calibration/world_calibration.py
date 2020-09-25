"""
world_calibration.py

get calibration parameters for the world camera using checkerboard videos

Sept. 24, 2020
"""

# package imports
import cv2
import numpy as np
from tqdm import tqdm
import xarray as xr
import argparse

# get user inputs
parser = argparse.ArgumentParser(description='calibrate worldcam from video of moving checkerboard')
parser.add_argument('-d', '--checker_vid_path', help='path to checkerboard video as .avi')
parser.add_argument('-s', '--save_path', help='save path for calibration parameters')
args = parser.parse_args()

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#prepare object points
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# read in file path of video
calib_vid = cv2.VideoCapture(args.checker_vid_path)

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
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#format as xarray
checker_calib = xr.DataArray([mtx, dist, rvecs, tvecs], dims=['mtx', 'dist', 'rvecs', 'tvecs'])
# save as .nc file
checker_calib.to_netcdf(os.path.join(args.save_path, 'world_checker_calib_params.nc'))
print('calibration file saved at ' + args.save_path)
