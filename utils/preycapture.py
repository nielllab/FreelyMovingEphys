
import os
import argparse
import glob
import sys
import yaml
import traceback
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
sys.path.append(str(Path('.').absolute()))
from tqdm import tqdm

from utils.aux_funcs import find, list_subdirs
from utils.base import BaseInput

class PreyCapture(BaseInput):
    def __init__(self, config, recording_name, recording_path, has_opto=False):
        super.__init__(self, config, recording_name, recording_path)

        self.has_opto = has_opto

    def calc_basic_param_from_file(self, f, pixpercm=14.5, thresh=0.99, framerate=60):
        # load all of data in file
        data = xr.open_dataset(f)

        # load DLC points for mouse
        Cricket_p = data['TOP1_pts'].sel(point_loc='Cricket1_likelihood').data
        Rear_x = data['TOP1_pts'].sel(point_loc='Rear_x').data/pixpercm
        Rear_y = data['TOP1_pts'].sel(point_loc='Rear_y').data/pixpercm
        Rear_xy = np.asarray([Rear_x, Rear_y])
        Lear_x = data['TOP1_pts'].sel(point_loc='Lear_x').data/pixpercm
        Lear_y = data['TOP1_pts'].sel(point_loc='Lear_y').data/pixpercm
        Lear_xy = np.asarray([Lear_x, Lear_y])
        mouse_xy = 0.5*(Rear_xy + Lear_xy)
        
        # load DLC points for cricket
        Cricket_x = (data['TOP1_pts'].sel(point_loc='Cricket1_x').data+data['TOP1_pts'].sel(point_loc='Cricket2_x').data)/2
        Cricket_y = (data['TOP1_pts'].sel(point_loc='Cricket1_y').data+data['TOP1_pts'].sel(point_loc='Cricket2_y').data)/2
        Cricket_x = Cricket_x/pixpercm
        Cricket_y = Cricket_y/pixpercm
        crick_x_thresh = Cricket_x.copy()
        crick_x_thresh[Cricket_p < thresh] = np.nan
        crick_y_thresh = Cricket_y.copy()
        crick_y_thresh[Cricket_p < thresh] = np.nan
        Cricket_xy = [crick_x_thresh, crick_y_thresh]
        
        # interpolate values to fill NaNs
        ind = np.arange(0,len(crick_x_thresh))
        interp = interp1d(ind[~np.isnan(crick_x_thresh)], crick_x_thresh[~np.isnan(crick_x_thresh)], bounds_error=False, fill_value=np.nan)
        cricket_x_interp = interp(ind)

        ind = np.arange(0,len(crick_y_thresh))
        interp = interp1d(ind[~np.isnan(crick_y_thresh)], crick_y_thresh[~np.isnan(crick_y_thresh)], bounds_error=False, fill_value=np.nan)
        cricket_y_interp = interp(ind)
            
        # calculate time to capture, currently not the best, think there is a tradeoff between having the thresh low enough that speed and range look better and getting the exact time to capture
        captureT = np.max(np.where(~np.isnan(Cricket_xy[0])))/framerate # return this
        movieT = len(Cricket_p)/framerate
        if captureT.size==0:
            captureT = movieT
        
        # calculate and plot distance between mouse and cricket
        timestamps = np.asarray(data['timestamps'])
        t = timestamps-timestamps[0]
        dist = np.sqrt(np.square(crick_x_thresh- mouse_xy[0]) + np.square(crick_y_thresh - mouse_xy[1])) #cmn - changes to cricket_x instead of interp.
        dist[-1] = 0  # we know that last point should be 0 range, since it's capture
        
        # more interpolation!!!
        ind = np.arange(0,len(dist))
        interp = interp1d(ind[~np.isnan(dist)], dist[~np.isnan(dist)],bounds_error=False, fill_value=np.nan )
        range_interp = interp(ind)
        dist = range_interp
            
        # calculate mouse speed
        win = 12
        dx = np.diff(mouse_xy[0],prepend=np.nan)
        dx = np.convolve(dx,np.ones(win)/win, 'same')
        dy = np.diff(mouse_xy[1],prepend=np.nan)
        dy = np.convolve(dy,np.ones(win)/win, 'same')
        spd = (np.sqrt(np.square(dx)+np.square(dy)))*framerate
        
        # calculate azimuth 
        mouse_az = np.arctan2((Cricket_xy[1] - mouse_xy[1]),(Cricket_xy[0] - mouse_xy[0]))*180/np.pi
        head_az = np.arctan2((Rear_xy[1] - Lear_xy[1]),(Rear_xy[0] - Lear_xy[0]))*180/np.pi -90
        az = mouse_az-head_az
        az = np.mod(az+180,360)-180
        if np.sum(~np.isnan(az))>0:
            azOld = az
            ind = np.arange(0,len(az))
            interp = interp1d(ind[~np.isnan(az)], az[~np.isnan(az)],bounds_error=False, fill_value=np.nan )
            az = interp(ind)

        return az, spd, dist, mouse_xy, np.array(Cricket_xy), t, movieT, captureT

    def calc_prob (self, az, spd, dist, mouse_xy, Cricket_xy, t, movieT, med_filt_win=15):
        # find the start and end of each approach
        # approach = []
        # paired = list(zip(az,spd))
        approach  = (np.abs(az) < 30) & (spd > 5)
        approach = approach.astype(int)
        # for pair in paired:
        #     if np.abs(pair[0]) < 30 & pair[1] > 5:
        #         approach.append(1)
        #     else:
        #         approach.append(0)

        approach = signal.medfilt(approach, med_filt_win) # 31 is hardcoded half a second based on framerate; 15=.25*60 fps
        approach = np.asarray(approach)

        approachStarts = np.where(np.diff(approach)>0)
        approachEnds = np.where(np.diff(approach)<0)
        if np.size(approachStarts) != 0:
            firstApproach = np.min(approachStarts)
            dist_at_approach = dist[approachStarts]
            timetoapproach = t[firstApproach] # return this
        else:
            firstApproach = np.nan
            dist_at_approach = np.nan
            timetoapproach = np.nan

        freqapproach= np.size(approachStarts) / movieT # return this
        
        # find instances of intercept given an approach (end of approach range <2cm); index dist using approachEnds, if range value <2, then call an intercept
        maybeIntercept = np.take(dist, approachEnds) # uses approachEnds to index dist
        maybeIntercept = maybeIntercept[0] # np.take returns tuple, first value are the ones you one
        if np.size(maybeIntercept)>0:
            maybeIntercept[-1] = 0 # assuming last approach is intercept/capture, makes things werk
        intercept = []
        for i in maybeIntercept:
            if i < 5:
                intercept.append(1)
            else:
                intercept.append(0)

        # calculate probability of intercept given approach
        tot_approach = np.size(approachEnds)
        tot_intercept = sum(intercept)
        
        # calculate the probability of capture given contact - 1/number of intercepts
        if tot_intercept>0:
            prob_inter = tot_intercept / tot_approach
            prob_capture = 1 / tot_intercept
        else:
            prob_inter = 0
            prob_capture = 0
            print('no capture')
        
        return timetoapproach, freqapproach, prob_inter, prob_capture, dist_at_approach

    def calc_params(self):
        recording_names = [i for i in list_subdirs(config['animal_dir'])]
        recording_paths = [os.path.normpath(os.path.join(config['animal_dir'], recording_name)) for recording_name in recording_names]
        recordings_dict = dict(zip(recording_names, recording_paths))

        for dir_name in recordings_dict:
            self.config['recording_path'] = recordings_dict[dir_name]
            recording_name = '_'.join(os.path.splitext(os.path.split([i for i in find('*.avi', config['recording_path']) if all(bad not in i for bad in ['plot','IR','rep11','betafpv','side_gaze'])][0])[1])[0].split('_')[:-1])
            topfile = os.path.join(recordings_dict[dir_name], recording_name + '_TOP1.nc')

            az, spd, dist, mouse_xy, Cricket_xy, t, movieT, captureT = self.calc_basic_param_from_file(topfile)
            timetoapproach, freqapproach, prob_inter, prob_capture, dist_at_approach = self.calc_prob(az, spd, dist, mouse_xy, Cricket_xy, t, movieT)


            df = pd.DataFrame({'Angle': az,
                            'Speed': spd,
                            'Dist':  dist,
                            'Mouse_x': mouse_xy[0],
                            'Mouse_y': mouse_xy[1],
                            'Cricket_x': Cricket_xy[0],
                            'Cricket_y': Cricket_xy[1],
                            't': t,
                            })
            metadata = {
                'MovieT': movieT,
                'CaptureT': captureT,
                'TimeToApproach': timetoapproach,
                'FreqApproach': freqapproach,
                'ProbInter': prob_inter,
                'ProbCapture': prob_capture,
                'dist_at_approach': dist_at_approach,
            }
            
            ##### Save Data into h5 file #####
            fpath = os.path.join(self.config['animal_directory'], dir_name, recording_name +'_BasicParams.h5')
            with pd.HDFStore(fpath) as store:
                store.put('df', df)
                store.get_storer('df').attrs.metadata = metadata

            ##### Plot Trials Data #####
            with PdfPages(os.path.join(self.config['animal_directory'], dir_name, recording_name +'_BasicParams_Plots.pdf')) as pdf:
                fig, ax = plt.subplots(figsize=(10,5))
                ax.plot(df['t'],df['Angle'], c='b')
                ax.plot(df['t'],df['Speed'], c='k')
                ax.plot(df['t'],df['Dist'], c='m')
                ax.axvline(x=metadata['TimeToApproach'], c='g')
                ax.axvline(x=metadata['CaptureT'], c='r')
                ax.set_title('Basic Params')
                ax.legend(['Angle', 'Speed', 'Dist. to Cricket', 'TimeToApproach', 'CaptureT'], bbox_to_anchor=(1.01, 1), fontsize=10)
                plt.tight_layout()
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

                fig, ax = plt.subplots(figsize=(10,8))
                plot_min, plot_max = np.nanmin(df[['Mouse_x','Mouse_y','Cricket_x','Cricket_y']].to_numpy()), np.nanmax(df[['Mouse_x','Mouse_y','Cricket_x','Cricket_y']].to_numpy())
                ax.plot(df['Mouse_x'], df['Mouse_y'],c='k')
                ax.plot(df['Cricket_x'], df['Cricket_y'],c='r')
                ax.set_xlim([plot_min-1,plot_max+1])
                ax.set_ylim([plot_min-1,plot_max+1])
                ax.set_title('DLC Tracking')
                ax.legend(['Mouse', 'Cricket'],bbox_to_anchor=(1.01, 1), fontsize=10)
                ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
                ax.set_aspect('equal', 'box')
                plt.tight_layout()
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
            
    def calc_days_data(csv2, base_path):
        row = csv2.iloc[0]
        fpath = str(list((base_path /row['experiment_date'] / row['animal_name'] /'trial_{:d}'.format(row['Trial'])).glob('*_BasicParams.h5'))[0])
        with pd.HDFStore(fpath) as store:
            data = store['df']
            metadata = store.get_storer('df').attrs.metadata
        df_meta = pd.DataFrame(columns=list(metadata.keys()))
        for ind, row in csv2.iterrows():
            fname = str(list((base_path /row['experiment_date'] / row['animal_name'] /'trial_{:d}'.format(row['Trial'])).glob('*_BasicParams.h5'))[0])
            with pd.HDFStore(fname) as store:
                data = store['df']
                metadata = store.get_storer('df').attrs.metadata
                metadata.update(csv2[['animal_name','experiment_date','Trial','LaserOn','Wallpaper']].iloc[ind])
                df_meta = df_meta.append(metadata, ignore_index=True)
        with PdfPages(base_path / 'Todays_BasicParams_Plots.pdf') as pdf:
            fig, ax = plt.subplots()
            g = sns.catplot(
                data=df_meta, kind="bar",
                x="Wallpaper", y="CaptureT", hue="LaserOn",
                ci='sd', palette="dark", alpha=.6, height=6
                )
            g.despine(left=True)
            g.set_axis_labels("", "Time (s)")
            g.legend.set_title("")
            g.set_xticklabels(rotation=90)
            plt.tight_layout()
            pdf.savefig()  # saves the current figure into a pdf page