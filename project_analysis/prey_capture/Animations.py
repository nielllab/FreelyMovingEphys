import os
import argparse
import glob
import sys 
import yaml 
import glob
import cv2
import time
import itertools
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


from matplotlib.backends.backend_pdf import PdfPages
from tqdm.notebook import tqdm, trange
from scipy import interpolate 
from scipy import signal
from scipy.signal import medfilt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from celluloid import Camera

from pathlib import Path
sys.path.append(str(Path('.').absolute().parent.parent))
# sys.path.append(str(Path('~/Research/MyRepos/SensoryMotorPred/PredCoding/').expanduser()))
# from util.paths import find, list_subdirs
# import io_dict_to_hdf5 as ioh5



import matplotlib as mpl
mpl.rcParams.update({'font.size':         24,
                     'axes.linewidth':    3,
                     'xtick.major.size':  5,
                     'xtick.major.width': 2,
                     'ytick.major.size':  5,
                     'ytick.major.width': 2,
                     'axes.spines.right': False,
                     'axes.spines.top':   False, 
                     'font.sans-serif':  "Arial",
                     'font.family':      "sans-serif",
                    })
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import platform
if platform.system() == 'Linux':
    base_path = Path('~/NewMonster/T/BinocOptoPreyCapture/').expanduser()
    print('Running on Linux')
else:
    base_path = Path('T:/BinocOptoPreyCapture').expanduser()
    print('Running on Windows')
    
########## Checks if path exists, if not then creates directory ##########
def check_path(basepath, path):
    ''' Created by Elliott Abe '''
    if path in basepath.as_posix():
        return basepath
    elif not (basepath / path).exists():
        (basepath / path).mkdir(exist_ok=True,parents=True)
        print('Added Directory:'+ (basepath / path).as_posix())
        return (basepath / path)
    else:
        return (basepath / path)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def add_colorbar(mappable,linewidth=2):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax, drawedges=False)
    cbar.outline.set_linewidth(linewidth)
    plt.sca(last_axes)
    return cbar

cmap_name = 'Accent'

def get_laser_onoff(imufile,laser_ksize=3):
    imu_data = xr.open_dataset(imufile)
    LaserT = imu_data.IMU_data.sample.data # imu timestamps
    LaserT0 = LaserT[0]
    LaserT = LaserT - LaserT0
    laser = imu_data.IMU_data.sel(channel='acc_x_raw').data # imu dample data
    dlaser = np.diff(medfilt(laser, kernel_size=laser_ksize),append=0)
    l_inds = np.where((np.abs(dlaser)>1.5))[0]
    LOn_ind = l_inds[0]
    LOff_ind = l_inds[-1]
    return LOn_ind, LOff_ind, LaserT


SavePath = check_path(Path('~/Research/PreyCapture/').expanduser(),'Data')
df_meta = pd.read_feather(SavePath/'df_meta.feather')
df_all = pd.read_feather(SavePath/'df_all.feather')
df_all = df_all.set_index(['experiment_date','animal_name','Trial'])
df = df_all.dropna()
FigPath = check_path(Path('~/Research/PreyCapture/').expanduser(),'Figures/')
blue_patch = mpatches.Patch(color='blue', label='Angle')
black_patch = mpatches.Patch(color='black', label='Speed')
m_patch = mpatches.Patch(color='m', label='Dist')
metrics = ['Angle','Speed','Dist']
perms = np.array(list(itertools.combinations(np.arange(len(metrics)), 2)))
order=['Wno','Lno','Hno','Wsw','Wlb','Wsb','Lsb','Hsb',]


##### Histogram data #####
h2dbins=10
hist_data_all = np.zeros((h2dbins,h2dbins,2,2,perms.shape[0],len(order)))# (bin x bin x Exp x LaserOn x Comb x Env)
xedges_all = np.zeros((h2dbins+1,2,2,perms.shape[0],len(order)))# (bin x Exp x LaserOn x Comb x Env)
yedges_all = np.zeros((h2dbins+1,2,2,perms.shape[0],len(order)))# (bin x Exp x LaserOn x Comb x Env)
for en, env in enumerate(tqdm(order)):
    for n in range(perms.shape[0]):
        for m, Exp in enumerate([True,False]):
            hist_data_Lon =df[['Angle','Speed','Dist','Environment','LaserOn']][(df['ExpAni']==Exp) & (df['LaserOn']==True) & (df['Environment']==env)].reset_index()
            hist_data_Loff = df[['Angle','Speed','Dist','Environment','LaserOn']][(df['ExpAni']==Exp) & (df['LaserOn']==False) & (df['Environment']==env)].reset_index()
            H_on,xedges_on,yedges_on = np.histogram2d(hist_data_Lon[metrics[perms[n,0]]],hist_data_Lon[metrics[perms[n,1]]],bins=h2dbins, density=True)
            H_off,xedges_off,yedges_off = np.histogram2d(hist_data_Loff[metrics[perms[n,0]]],hist_data_Loff[metrics[perms[n,1]]],bins=10, density=True)
            hist_data_all[:,:,0,m,n,en] = H_on
            hist_data_all[:,:,1,m,n,en] = H_off
            xedges_all[:,0,m,n,en] = xedges_on
            xedges_all[:,1,m,n,en] = xedges_off
            yedges_all[:,0,m,n,en] = yedges_on
            yedges_all[:,1,m,n,en] = yedges_off

for n,row, df_meta.iterrows():
    ##### Extract Values from DataFrames #####
    trial = df_meta['Trial'].iloc[n]
    date = df_meta['experiment_date'].iloc[n]
    ani = df_meta['animal_name'].iloc[n]
    lon = df_meta['LaserOnInd'].iloc[n]
    loff = df_meta['LaserOffInd'].iloc[n]
    print('{}, {}, {}, {}'.format(n,date,ani,trial))
    ang_norm = df_all.xs((date,ani,trial),level=(0,1,2))['Angle_norm'].values
    spd_norm = df_all.xs((date,ani,trial),level=(0,1,2))['Speed_norm'].values
    dist_norm = df_all.xs((date,ani,trial),level=(0,1,2))['Dist_norm'].values
    ang = df_all.xs((date,ani,trial),level=(0,1,2))['Angle'].values
    spd = df_all.xs((date,ani,trial),level=(0,1,2))['Speed'].values
    dist = df_all.xs((date,ani,trial),level=(0,1,2))['Dist'].values
    asd = np.vstack((ang,spd,dist))
    asd_norm = np.vstack((ang_norm,spd_norm,dist_norm))
    tvid = df_all.xs((date,ani,trial),level=(0,1,2))['t'].values 
    df_trial = df_meta[((df_meta['experiment_date']==date)&(df_meta['animal_name']==ani)&(df_meta['Trial']==trial))].iloc[0]
    Exp = True if df_trial['Exp_Ctrl']=='Exp' else False
    LaserOn = df_trial['LaserOn']
    Env = df_trial['Environment']

    # df_trial[['animal_name','experiment_date','Trial','CaptureT','MovieT','LaserOnT','LaserOffT','LaserOn','Environment','Exp_Ctrl','Capture','Cap_dur_laser']]
    ##### Grab Video and h5 files #####
    fname = str(list((base_path /df_trial['experiment_date'] / df_trial['animal_name'] /'{:d}'.format(df_trial['Trial'])).glob('*TOP1.avi'))[0])
    fname_h5 = str(list((base_path /df_trial['experiment_date'] / df_trial['animal_name'] /'{:d}'.format(df_trial['Trial'])).glob('*TOP1*.h5'))[0])

    ##### Read h5 DLC Points #####
    pts = pd.read_hdf(fname_h5)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
    pt_loc_names = pts.columns.values
    npts = len(pts.iloc[0,::3])
    cmap = mpl.cm.get_cmap('jet', npts)    # 16 discrete colors

    ##### Nan out low LL points #####
    thresh=.99
    npts = len(pts.iloc[0,::3])
    for n,key in enumerate(list(pts.keys()[2::3])):
        lowLL = (pts[key]<.99)
        pts[key.split('_')[0] + '_x'].loc[pts[key]<thresh] = np.nan
        pts[key.split('_')[0] + '_y'].loc[pts[key]<thresh] = np.nan

    ##### Load video frames, down sample #####
    cap = cv2.VideoCapture(fname)
    downsamp = 0.5
    width = int(downsamp*cap.get(3)) 
    height = int(downsamp*cap.get(4))
    totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Vid = np.zeros((totalframes, height, width), dtype=np.uint8)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file: " + Videos[num])

    # Read until video is completed
    for t in tqdm(np.arange(0, totalframes)):
        # Capture frame-by-frame
        ret, frame_temp = cap.read()
        Vid[t, :, :] = cv2.resize(cv2.cvtColor(frame_temp, cv2.COLOR_BGR2GRAY),(width,height))

    # When everything done, release the video capture object
    cap.release()

    ##### Start Animations #####
    start = time.time()
    nbins=10
    xbins_on_all = np.zeros((perms.shape[0],nbins+1))
    ybins_on_all = np.zeros((perms.shape[0],nbins+1))
    fig2 = plt.figure(constrained_layout=False, figsize=(15,15))
    spec2 = gridspec.GridSpec(ncols=3, nrows=4, figure=fig2)
    axs1 = fig2.add_subplot(spec2[0, :]) 
    axs2 = fig2.add_subplot(spec2[1:3, :])
    axs3 = np.array([fig2.add_subplot(spec2[3, n]) for n in range(3)])
    camera = Camera(fig2)
    for t in tqdm(np.arange(tvid.shape[0])):
        # Plotting Angle, Speed, Dist
        p1 = axs1.scatter(tvid,ang_norm, 3, c='b',)
        p2 = axs1.plot(tvid,spd_norm, c='k')
        p3 = axs1.plot(tvid,dist_norm, c='m')
        axs1.legend(handles=[blue_patch,black_patch,m_patch],labels=['Angle','Speed','Dist'],loc='upper right',bbox_to_anchor=(1.01, 1), fontsize=10)
        axs1.axhline(y=0,c='k',linestyle='--')
        axs1.set_title(fname.split('/')[-1][:-9])
        axs1.axvline(x=tvid[t],c='b',linestyle='-')
        axs1.set_xlabel('Time (s)')

        # Image Frame and DLC Tracking
        axs2.imshow(Vid[t],cmap='gray')
        axs2.scatter(pts.iloc[t,::3]*downsamp,pts.iloc[t,1::3]*downsamp,c=np.arange(npts-1,-1,-1),cmap=cmap)

        # 2d hist of Angle, Speed, Dist.
        for n in range(perms.shape[0]):
            clim = (hist_data_all[:,:,:,:,n,en].min(),hist_data_all[:,:,:,:,n,en].max())
            hist_data_Lon =df[['Angle','Speed','Dist','Environment','LaserOn']][(df['ExpAni']==Exp) & (df['LaserOn']==LaserOn) & (df['Environment']==Env)].reset_index()
            h_on, xbins_on, ybins_on, im_on, = axs3[n].hist2d(hist_data_Lon[metrics[perms[n,0]]],hist_data_Lon[metrics[perms[n,1]]],bins=nbins, density=True,vmin=clim[0],vmax=clim[1])
            axs3[n].set_xlabel(metrics[perms[n,0]])
            axs3[n].set_ylabel(metrics[perms[n,1]])
            axs3[n].set_title('LOn:{},{}'.format(LaserOn,df_trial['Exp_Ctrl']),fontsize=20)
            if np.isnan(asd[perms[n],t]).any() != True:
                xpt = xbins_on[np.digitize(asd[perms[n,0],t],xbins_on)]+np.mean(np.diff(xbins_on))/2
                ypt = ybins_on[np.digitize(asd[perms[n,1],t],ybins_on)]+np.mean(np.diff(ybins_on))/2
                axs3[n].scatter(xpt, ypt, 200,'r')
        plt.tight_layout()
        camera.snap()


    aniname = fname.split('/')[-1][:-4] + '_summary.mp4'
    animation = camera.animate(interval=(1/fps)*1000)
    animation.save(FigPath / aniname)
    print('Animation: ', time.time()-start)

