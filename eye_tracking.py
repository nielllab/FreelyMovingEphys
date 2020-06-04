#####################################################################################
"""
eye_tracking.py of FreelyMovingEphys

[desc.]

Adapted from Elliott's file, DLCEyeVids.py

last modified: June 3, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################

import os, fnmatch
import cv2
import pandas as pd
import numpy as np
from skimage import draw, measure
from skvideo import io
from itertools import product
import argparse
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from numba import jit
from scipy import stats
import matplotlib as mpl
​
### Define Functions
# Get the list of videos in a path or a list of videos from a given list
def Getlistofvideos(videos,videotype):
    #checks if input is a directory
    if [os.path.isdir(i) for i in videos] == [True]:#os.path.isdir(video)==True:
        """
        Analyzes all the videos in the directory.
        """
        # print("Analyzing all the videos in the directory")
        videofolder= videos[0]
        os.chdir(videofolder)
        videolist=[fn for fn in os.listdir(os.curdir) if (videotype in fn) and ('labeled.mp4' not in fn)] #exclude labeled-videos!
        Videos = videolist #sample(videolist,len(videolist)) # this is useful so multiple nets can be used to analzye simultanously
    else:
        if isinstance(videos,str):
            if os.path.isfile(videos): # #or just one direct path!
                Videos=[v for v in videos if os.path.isfile(v) and ('labeled.mp4' not in v)]
            else:
                Videos=[]
        else:
            Videos=[v for v in videos if os.path.isfile(v) and ('labeled.mp4' not in v)]
    return Videos
​
# Find the corresponding DLC file for the video that is loaded in
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
​
# Draw points on frame
def draw_points(frame, x, y, color,ptsize):
    point_adds = product(range(-ptsize,ptsize), range(-ptsize,ptsize))
    for pt in point_adds:
        try:
            frame[x+pt[0],y+pt[1]] = color
        except IndexError:
            pass
    return frame
​
# Load DLC Points and Organize them
​
def loadDLCEye(dlcfile):
    pts = pd.read_hdf(dlcfile[0])
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pts.rename(columns={'Left IR x':'LeftIR x', 'Left IR y':'LeftIR y', 'Left IR likelihood':'LeftIR likelihood',
       'Right IR x':'RightIR x', 'Right IR y':'RightIR y',
       'Right IR likelihood':'RightIR likelihood', 'Right Ear x':'RightEar x', 'Right Ear y':'RightEar y',
       'Right Ear likelihood':'RightEar likelihood', 'Back of Head x':'BackofHead x', 'Back of Head y':'BackofHead y',
       'Back of Head likelihood':'BackofHead likelihood', 'cricket Head x':'cricketHead x', 'cricket Head y':'cricketHead y',
       'cricket Head likelihood':'cricketHead likelihood', 'cricket Body x':'cricketBody x', 'cricket Body y':'cricketBody y',
       'cricket Body likelihood':'cricketBody likelihood'}, inplace=True)
    # Clean data based on likelihood
    likelihood = np.array(pts.iloc[:,2::3])
    pts[~(np.abs(stats.zscore(pts)) < 2).all(axis=1)]=np.nan
    bdfit, y = np.where(likelihood<.8)
    pts.iloc[np.unique(bdfit),(y*3)]=np.nan
    pts.iloc[np.unique(bdfit),((y+1)*3-2)]=np.nan
    pts = pts.interpolate(method='linear', limit_direction='both', axis=0)
    ptsx = np.array(pts.iloc[:,::3])
    ptsy = np.array(pts.iloc[:,1::3])
    # Add Frame Number column
    pts['frame'] = pts.index
    ptso = pts.copy()
    # put data into long form
    pts = pts.melt(id_vars='frame')
    pts['point'], pts['type'] = pts['variable'].str.split(' ', 1).str
    n_pts = len(pts.point.unique())
    pts.drop('variable', axis=1,inplace=True)
    # pivot the df to two indices, frame, point, and the likelihood and coords as columns
    pts = pd.pivot_table(pts, values='value', index=['frame', 'point'], columns='type')
    frame_points = pts.groupby('frame')
    return pts,ptsx,ptsy,frame_points,n_pts,ptso
​
def loadDLCTop(dlcfile):
    pts = pd.read_hdf(dlcfile[0])
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    # Clean data based on likelihood
    likelihood = np.array(pts.iloc[:,2::3])
#     pts[~(np.abs(stats.zscore(pts)) < 2).all(axis=1)]=np.nan
#     bdfit, y = np.where(likelihood<.8)
#     pts.iloc[np.unique(bdfit),(y*3)]=np.nan
#     pts.iloc[np.unique(bdfit),((y+1)*3-2)]=np.nan
#     pts = pts.interpolate(method='linear', limit_direction='both', axis=0)
    ptsx = np.array(pts.iloc[:,::3])
    ptsy = np.array(pts.iloc[:,1::3])
    # Add Frame Number column
    pts['frame'] = pts.index
    ptso = pts.copy()
    # put data into long form
    pts = pts.melt(id_vars='frame')
    pts['point'], pts['type'] = pts['variable'].str.split(' ', 1).str
    n_pts = len(pts.point.unique())
    pts.drop('variable', axis=1,inplace=True)
    # pivot the df to two indices, frame, point, and the likelihood and coords as columns
    pts = pd.pivot_table(pts, values='value', index=['frame', 'point'], columns='type')
    frame_points = pts.groupby('frame')
    return pts,ptsx,ptsy,frame_points,n_pts,ptso
​
# Parallized version to calculate ellipse parameters without video
@jit
def calcEllip_numba(ptsx,ptsy):
    emod = measure.EllipseModel()
    total_frames = ptsx.shape[0]
    ellipseparams = np.empty((0,5)) # 5 is the number of parameters the ellipse outputs
​
    for i in range(total_frames):
        # first the ellipse
        xy = np.column_stack((ptsy[i,:], ptsx[i,:]))
        emod.estimate(xy)
        ellipseparams = np.append(ellipseparams,np.expand_dims(np.array(emod.params),axis=0),axis=0)
    return ellipseparams
​
# Checks pupil size for outlairs and cleans data
def CleanData(ellipseparams,thresh):
    bdfit2,temp = np.where(ellipseparams[:,2:4]>thresh)
    eparams = pd.DataFrame(ellipseparams)
    eparams.iloc[bdfit2,:] = np.nan
    eparams = eparams.interpolate(method='linear', limit_direction='both', axis=0)
    ellipseparams[bdfit2,:] = eparams.iloc[bdfit2,:]
    return ellipseparams
​
​
def playEyeVid(base_dir,Videos,vidnum,pts,frame_points,colors):
​
    vid = cv2.VideoCapture(os.path.join(base_dir + Videos[vidnum]))
    cv2.namedWindow('play', flags=cv2.WINDOW_NORMAL)
    n_frame = 0
​
    thetas = np.linspace(-np.pi, np.pi, 50)
    emod = measure.EllipseModel()
​
​
    # iter frames, draw points
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    ellipseparams = np.empty((0,5))
    for i in tqdm(range(total_frames)):
        # if we try to quit, quit nicely
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
​
        # grab frame
        ret, frame = vid.read()
​
        # draw points
        n_frame = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
        # first the ellipse
        rows = frame_points.get_group(n_frame)
        xy = np.column_stack((rows['y'], rows['x']))
        emod.estimate(xy)
        ellipseparams = np.append(ellipseparams,np.expand_dims(np.array(emod.params),axis=0),axis=0)
        e_points = emod.predict_xy(thetas).astype(np.int)
        for e_pts in e_points:
            frame = draw_points(frame, e_pts[0], e_pts[1], [0,0,255])
​
        # and then the points themselves
        for color, (idx, row) in zip(colors, rows.iterrows()):
            frame = draw_points(frame, int(row['y']), int(row['x']), color, 3)
​
        cv2.imshow('play', frame)
    # vid.release()
    cv2.destroyAllWindows()
    return ellipseparams
​
def GetEyeAngles(ellipseparams):
    R = np.linspace(0,2*np.pi,100)
    longaxis_all = np.maximum(ellipseparams[:,2],ellipseparams[:,3])
    shortaxis_all = np.minimum(ellipseparams[:,2],ellipseparams[:,3])
    Ellipticity = shortaxis_all/longaxis_all
    lis, = np.where(Ellipticity<.9)
    A = np.vstack([np.cos(ellipseparams[lis,4]),np.sin(ellipseparams[lis,4])])
    b = np.expand_dims(np.diag(A.T@np.squeeze(ellipseparams[lis,0:2].T)),axis=1)
    CamCent=np.linalg.inv(A@A.T)@A@b
    longaxis = np.squeeze(np.maximum(ellipseparams[lis,2],ellipseparams[lis,3]))
    shortaxis = np.squeeze(np.minimum(ellipseparams[lis,2],ellipseparams[lis,3]))
    Ellipticity = shortaxis/longaxis
    scale = np.sum(np.sqrt(1-(Ellipticity)**2)*(np.linalg.norm(ellipseparams[lis,0:2]-CamCent.T,axis=1)))/np.sum(1-(Ellipticity)**2);
#     scale = np.max((ellipseparams[:,0]-CamCent[0]))
    if scale < 40:
        scale = 40
    temp = (ellipseparams[:,0]-CamCent[0])/scale
    theta = np.arcsin(temp)
    phi = np.arcsin((ellipseparams[:,1]-CamCent[1])/np.cos(theta)/scale)
    return theta, phi, longaxis_all, shortaxis_all, CamCent
​
def CreateTimeColor(N):
    tag = np.arange(0,N) # Tag each point with a corresponding label    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return tag,cmap,norm