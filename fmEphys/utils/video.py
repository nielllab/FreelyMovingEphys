"""
fmEphys/utils/video.py

Video processing functions.

Functions
---------
make_avi_savepath
    Make a savepath for an avi file.
deinterlace
    Deinterlace a video.
rotate_video
    Rotate a video along either or both axes.
calc_distortion
    Calculate the distortion in a camera image.
fix_distortion
    Apply the camera matrix to novel videos to correct distortion.
fix_contrast
    Rescale the contrast of a video.
avi_to_arr
    Convert an avi video to a numpy array.


Written by DMM, 2021
"""


import os
import cv2
import subprocess
import numpy as np
from tqdm import tqdm

import fmEphys as fme


def make_avi_savepath(path, add_key):
    """ Make a savepath for an avi file.

    Optionally add an additional key to note modifications
    to the video (e.g., 'deinter', 'rotate', etc.)

    Parameters
    ----------
    path : str
        File path to the original video file.
    add_key : str
        Additional key to add to the file name.
        e.g., 'deinter', 'rotate', etc.

    Returns
    -------
    savepath : str
        File path to the new video file.

    """

    savedir, _ = os.path.split(path)
    savename = '.'.join((os.path.split(path)[1]).split('.')[:-1])
    savepath = os.path.join(savedir,
                    ('{}-{}.avi'.format(savename, add_key)))
    
    return savepath


def deinterlace(path, savepath=None, rotate=True,
                exp_fps=30, quiet=True):
    """ Deinterlace a video.

    Parameters
    ----------
    path : str
        Path to the video file.
    savepath : str
        Path to save the new video file. Default is None.
    rotate : bool
        Whether to rotate the video 180 degrees. Default is True.
    exp_fps : int
        Expected frame rate of the video. If the video matches
        this frame rate (in Hz), it will be deinterlaced. Otherwise,
        it will be skipped. Default is 30 Hz.
    quiet : bool
        Whether to suppress the output from ffmpeg. Default is True.
    
    Returns
    -------
    savepath : str
        Path to the new video file.

    """

    # Make the savepath
    if savepath is None:
        savepath = make_avi_savepath(path, 'deinter')

    # Open video, get frame count and rate
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Skip this video if it doesn't match the expected frame rate.
    if fps != exp_fps:
        return
        
    # Create the FFMPEG commands for video rotation.
    if rotate:
        vf_val = 'yadif=1:-1:0, vflip, hflip, scale=640:480'
    elif not rotate:
        vf_val = 'yadif=1:-1:0, scale=640:480'

    # Create the full FFMPEG command
    cmd = ['ffmpeg', '-i', path, '-vf', vf_val, '-c:v', 'libx264',
          '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a',
          '256k', '-y', savepath]
    
    # Set the log level
    if quiet is True:
        cmd.extend(['-loglevel', 'quiet'])

    # Run the FFMPEG command.
    subprocess.call(cmd)

    return savepath


def rotate_video(path, savepath=None, h=False, v=False,
                 quiet=True):
    """ Rotate a video along either or both axes.

    Together, a horizontal and vertical flip are equivalent
    to a 180 degree rotation.

    Parameters
    ----------
    path : str
        Path to the video file.
    savepath : str
        Path to save the new video file. Default is None.
    h : bool
        Whether to flip the video horizontally. Default is False.
    v : bool
        Whether to flip the video vertically. Default is False.
    quiet : bool
        Whether to suppress the output from ffmpeg. Default is True.
    
    Results
    -------
    savepath : str
        Path to the new video file.
        
    """
    
    if savepath is None:
        savepath = make_avi_savepath(path, 'rotate')

    if h is True and v is True:
        vf_val = 'vflip, hflip'

    elif h is True and v is False:
        vf_val = 'hflip'

    elif h is False and v is True:
        vf_val = 'vflip'

    cmd = ['ffmpeg', '-i', path, '-vf', vf_val, '-c:v',
           'libx264', '-preset', 'slow', '-crf', '19',
           '-c:a', 'aac', '-b:a', '256k', '-y', savepath]

    if quiet is True:
        cmd.extend(['-loglevel', 'quiet'])

    # Only do the rotation is at least one axis is being flipped
    if h is True or v is True:
        subprocess.call(cmd)

    return savepath
 

def calc_distortion(path, savepath,
                    board_w=7, board_h=5):
    """ Calculate the distortion in a camera image.

    Parameters
    ----------
    path : str
        File path of an .avi video showing a printed-out
        checkerboard moving in the view of the camera at
        different positions and orientations.
    savepath : str
        Path to save a .npz matrix of parameters to correct
        the image distortions. This needs to include the
        desired file name and end with the .npz extension.
    board_w : int
        Number of horizontal squares in the checkerboard.
        Default is 7.
    board_h : int
        Number of vertical squares in the checkerboard. Default
        is 5.

    """

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((board_h*board_w,3), np.float32)
    objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

    # Read in video
    calib_vid = cv2.VideoCapture(path)

    nF = int(calib_vid.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames

    # Iterate through frames
    print('Finding checkerboard corners for {} frames'.format(nF))
    for _ in tqdm(range(nF)):

        # Open frame
        ret, img = calib_vid.read()

        # Make sure the frame is read in correctly
        if not ret:
            break

        # Convert to grayscale
        if img.shape[2] > 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h), None)

        if ret == True:

            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            imgpoints.append(corners)

    # Calculate the distortion
    print('Calculating distortion (slow)')

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                gray.shape[::-1], None, None)

    date, time = fme.fmt_now()
    np.savez(savepath,
             mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs,
             source_video=os.path.split(path)[1],
             written='{}_{}'.format(date, time))


def fix_distortion(path, proppath, savepath=None):
    """ Apply the camera matrix to novel videos to correct distortion.
    
    Parameters
    ----------
    path : str
        Path to the video file.
    proppath : str
        Path to the .npz file containing the camera matrix.
    savepath : str
        Path to save the new video file. Default is None, in which
        case the video will be named with the added key 'unwarp'.

    Returns
    -------
    savepath : str
        Path to the new video file.
    
    """

    if savepath is None:
        savepath = make_avi_savepath(path, 'unwarp')
    
    # Load the camera properties
    camprops = np.load(proppath)

    # Unpack camera properties
    mtx = camprops['mtx']
    dist = camprops['dist']
    # rvecs = camprops['rvecs']
    # tvecs = camprops['tvecs']

    # Read in video
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup the file writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    newvid = cv2.VideoWriter(savepath, fourcc, fps,
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Iterate through frames
    for _ in tqdm(range(nF)):

        # Read frame, make sure it opens correctly
        ret, frame = cap.read()
        if not ret:
            break

        # Fix distortion
        f = cv2.undistort(frame, mtx, dist, None, mtx)

        # write the frame to the video
        newvid.write(f)

    newvid.release()

    return savepath


def fix_contrast(path, savepath):
    """ Rescale the contrast of a video.

    Parameters
    ----------
    path : str
        Path to the video file.
    savepath : str
        Path to save the new video file.

    Returns
    -------
    savepath : str
        Path to the new video file.

    """

    if savepath is None:
        savepath = make_avi_savepath(path, 'fixcontrast')

    # Read in existing video, get some properties
    vid = cv2.VideoCapture(path)
    nF = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the new video set up
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    newvid = cv2.VideoWriter(savepath, fourcc, 60.0, (w, h))

    print('Increasing contrast for {} frames'.format(nF))

    for _ in tqdm(range(nF)):
        
        ret, f = vid.read()
        if not ret:
            break

        # Convert to greyscale
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        # compute gamma
        # gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = np.mean(gray)
        gamma = np.log(mid*255) / np.log(mean)

        # apply gamma correction to frame
        newF = np.power(f, gamma).clip(0, 255).astype(np.uint8)

        # Write frame
        newvid.write(newF)

    newvid.release()

    return savepath


def avi_to_arr(path, ds=0.25):
    """ Convert an avi video to a numpy array.

    Parameters
    ----------
    path : str
        Path to the video file.
    ds : float
        Downsampling factor. Default is 0.25. This will
        scale the video in x and y by this factor, so that a
        value of 1 returns the video at its original dimensions,
        and a value of 0.5 returns the video at half its origional
        resolution in the x and y dimensions. It does not rescale
        the video in z (time).

    Returns
    -------
    arr : np.ndarray
        Array of video frames with shape (frames, height, width).

    """

    vid = cv2.VideoCapture(path)

    # array to put video frames into
    # will have the shape: [frames, height, width] and be returned with dtype=int8
    arr = np.empty([int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)*ds),
                        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)*ds)], dtype=np.uint8)
    
    # iterate through each frame
    for f in range(0,int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
        
        # read the frame in and make sure it is read in correctly
        ret, img = vid.read()
        if not ret:
            break
        
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # downsample the frame by an amount specified in the config file
        img_s = cv2.resize(img, (0,0), fx=ds, fy=ds, interpolation=cv2.INTER_NEAREST)
        
        # add the downsampled frame to all_frames as int8
        arr[f,:,:] = img_s.astype(np.int8)

    return arr

