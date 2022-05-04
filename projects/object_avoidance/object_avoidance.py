import json, os, cv2
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

from src.base import BaseInput
from src.topcam import Topcam
from src.utils.auxiliary import flatten_series, find_index_in_list
from src.utils.path import find

class AvoidanceTrial(BaseInput):
    def __init__(self, s_input, path_input, metadata_input):
        self.s = s_input # series from dataframe of all trials
        self.likelihood_thresh = 0.99
        self.dist_across_arena = 30.48 # cm between bottom-right and bottom-left pillar
        self.path = path_input
        self.camname = 'top1'
        self.shared_metadata = metadata_input

        self.num_clusters_to_use = self.shared_metadata[self.s['date']][self.s['animal']][str(self.s['task'])]['num_positions']
        self.trial_path = os.path.join(*[self.path, str(self.s['date']), self.s['animal'],str(self.s['task'])])

        self.generic_camconfig = {
            'internals': {
                'follow_strict_naming': False,
                'likelihood_threshold': 0.99
            }
        }

    def convert_pxls_to_dist(self):
        x_cols = [i for i in self.data.columns.values if '_x' in i]
        y_cols = [i for i in self.data.columns.values if '_y' in i]
        for i in range(len(x_cols)):
            self.data[x_cols[i]+'_cm'] = self.data.loc[:,x_cols[i]] / self.pxls2cm
            self.data[y_cols[i]+'_cm'] = self.data.loc[:,y_cols[i]] / self.pxls2cm

    def setup_trial(self, c, odd):

    def make_task_df(self):
        num_odd_trials = np.min([len(self.s['poke1_ts']), len(self.s['poke2_ts'])])
        df1 = pd.DataFrame([])
        count = -1
        for c in range(num_odd_trials):
            # odd
            count += 1
            df1.at[count, 'first_poke'] = self.s['poke1_ts'][c]
            df1.at[count, 'second_poke'] = self.s['poke2_ts'][c]
            time = self.s['top1_ts']; time = time[time > df1.loc[count,'first_poke']]; time = time[time < df1.loc[count,'second_poke']]
            df1.at[count, 'trial_timestamps'] = time.astype(object)
            start_stop_inds = (int(np.where([self.s['top1_ts']==time[0]])[1]), int(np.where([self.s['top1_ts']==time[-1]])[1]))
            for pos in list(self.positions['point_loc'].values):
                df1.at[count, pos] = np.array(self.positions.loc[start_stop_inds[0]:start_stop_inds[1], pos]).astype(object)
            df1.at[count, 'len'] = start_stop_inds[1] - start_stop_inds[0]
            # even
            count += 1
            if c+1 < len(self.s['poke1_ts']):
                df1.at[count, 'first_poke'] = self.s['poke2_ts'][c]
                df1.at[count, 'second_poke'] = self.s['poke1_ts'][c+1]
                time = self.s['top1_ts']; time = time[time > df1.loc[count,'first_poke']]; time = time[time < df1.loc[count,'second_poke']]
                vidframes = np.array(list(find_index_in_list(list(self.s['top1_ts']), list(time))))
                df1.at[count, 'recording_timestamps'] = self.s['top1_ts'].astype(object)
                df1.at[count, 'trial_timestamps'] = time.astype(object)
                df1.at[count, 'trial_vidframes'] = vidframes.astype(object)
                start_stop_inds = (int(np.where([self.s['top1_ts']==time[0]])[1]), int(np.where([self.s['top1_ts']==time[-1]])[1]))
                for pos in list(self.positions['point_loc'].values):
                    df1.at[count, pos] = np.array(self.positions.loc[start_stop_inds[0]:start_stop_inds[1], pos]).astype(object)
                df1.at[count, 'len'] = start_stop_inds[1] - start_stop_inds[0]
        df1['animal'] = self.s['animal']; df1['date'] = self.s['date']; df1['task'] = self.s['task']
        self.data = df1

    def drop_lickport_times(self, ):

    def get_median_trace(self, df):
        fake_time = np.linspace(0,1,100)
        all_nose_positions = np.zeros([len(df), 2, 100])
        count = 0
        for ind, row in df.iterrows():
            xT = np.linspace(0,1,len(row['nose_x'])); yT = np.linspace(0,1,len(row['nose_y']))
            all_nose_positions[count,0,:] = interp1d(xT, row['nose_x_cm'], bounds_error=False)(fake_time)
            all_nose_positions[count,1,:] = interp1d(yT, row['nose_y_cm'], bounds_error=False)(fake_time)
            count += 1
        median_trace = np.nanmedian(all_nose_positions, axis=0)
        for ind, row in df.iterrows():
            df.at[ind,'median_x_cm'] = median_trace[0,:].astype(object); df.at[ind,'median_y_cm'] = median_trace[1,:].astype(object)
        return df

    def distance_from_nose(self, row, target, xy_together=False):
        if not xy_together:
            x_dist = np.abs(row[target+'_x_cm'] - row['nose_x_cm'])
            y_dist = np.abs(row[target+'_y_cm'] - row['nose_y_cm'])
        elif xy_together:
            x_dist = np.abs(row[target][0] - row['nose_x_cm'])
            y_dist = np.abs(row[target][1] - row['nose_y_cm'])
        length = len(x_dist)
        dist = np.zeros([length])
        for i in range(length):
            dist[i] = np.sqrt(x_dist[i]**2 + y_dist[i]**2)
        return dist

    def angle_from_nose(self, row, target, xy_together=False):
        if not xy_together:
            x_dist = np.abs(row[target+'_x_cm'] - row['nose_x_cm'])
            y_dist = np.abs(row[target+'_y_cm'] - row['nose_y_cm'])
        elif xy_together:
            x_dist = np.abs(row[target][0] - row['nose_x_cm'])
            y_dist = np.abs(row[target][1] - row['nose_y_cm'])
        length = len(x_dist)
        ang = np.zeros([length])
        for i in range(length):
            try:
                ang[i] = (y_dist[i]/x_dist[i])
            except ZeroDivisionError:
                ang[i] = np.nan
        return ang%np.pi

    def get_head_angle(self):
        for ind, row in self.data.iterrows():
            leftear_x = row['leftear_x_cm']
            leftear_y = row['leftear_y_cm']
            rightear_x = row['rightear_x_cm']
            rightear_y = row['rightear_y_cm']
            nose_x = row['nose_x_cm']
            nose_y = row['nose_y_cm']
            angs = []
            for step in range(len(leftear_x)):
                ang = np.arctan2(np.mean([leftear_y[step],rightear_y[step]])-nose_y[step],np.mean([leftear_x[step],rightear_x[step]])-nose_x[step])
                angs.append(ang)
            self.data.at[ind, 'head_angle'] = np.array(angs).astype(object)

    def approaching_target(self, dist):
        if np.nanmean(dist[:10]) > np.nanmean(dist[:-10]):
            return True
        else:
            return False

    def add_tracking(self, name, path):
        tc = Topcam(self.generic_camconfig, name, path, self.camname)
        tc.gather_files()
        tc.pack_position_data()
        tc.filter_likelihood()
        self.positions = tc.xrpts
        self.trial_name = name
        self.trial_path = path

    def pillar_avoidance(self):
        self.make_task_df()

        # label odd/even trials (i.e. moving leftwards or moving rightwards?)
        self.data['odd'] = np.nan
        for i, ind in enumerate(self.data.index.values):
            if ind%2 == 0: # odd values
                self.data.at[ind, 'odd'] = True
            elif ind%2 == 1:
                self.data.at[ind, 'odd'] = False

        dist_to_posts = np.median(self.data['arenaTR_x'].iloc[0],0) - np.median(self.data['arenaTL_x'].iloc[0],0)
        self.pxls2cm = dist_to_posts/self.dist_across_arena
        self.convert_pxls_to_dist()

        self.get_head_angle()

        pdf = PdfPages(os.path.join(self.trial_path, (self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'_figs.pdf')))

        # get obstacle position for all times
        for ind, row in self.data.iterrows():
            for x in ['b','w']:
                xvals = np.stack([row['obstacle'+x+'TL_x_cm'], row['obstacle'+x+'TR_x_cm'], row['obstacle'+x+'BL_x_cm'], row['obstacle'+x+'BR_x_cm']]).astype(float)
                xvals_cm = np.stack([row['obstacle'+x+'TL_x_cm'], row['obstacle'+x+'TR_x_cm'], row['obstacle'+x+'BL_x_cm'], row['obstacle'+x+'BR_x_cm']]).astype(float)
                self.data.at[ind, x+'obstacle_x'] = np.nanmean(xvals)
                self.data.at[ind, x+'obstacle_x_cm'] = np.nanmean(xvals_cm)
                self.data.at[ind, x+'obstacle_x_std'] = np.mean(np.nanstd(xvals, axis=1))
                yvals = np.stack([row['obstaclewTL_y_cm'], row['obstaclewTR_y_cm'], row['obstaclewBL_y_cm'], row['obstaclewBR_y_cm']]).astype(float)
                yvals_cm = np.stack([row['obstaclewTL_y_cm'], row['obstaclewTR_y_cm'], row['obstaclewBL_y_cm'], row['obstaclewBR_y_cm']]).astype(float)
                self.data.at[ind, x+'obstacle_y'] = np.nanmean(yvals)
                self.data.at[ind, x+'obstacle_y_cm'] = np.nanmean(yvals_cm)
                self.data.at[ind, x+'obstacle_y_std'] = np.mean(np.nanstd(yvals, axis=1))

        # animal speed
        for ind, row in self.data.iterrows():
            temp_time = np.diff(row['trial_timestamps'])
            x = np.diff(row['nose_x_cm']); y = np.diff(row['nose_y_cm'])
            if len(x) == len(temp_time):
                xspeed = list((x/temp_time)**2)
            elif len(x) > len(temp_time):
                xspeed = list((x[:len(temp_time)]/temp_time)**2)
            elif len(x) < len(temp_time):
                xspeed = list((x/temp_time[:len(x)])**2)
            if len(y) == len(temp_time):
                yspeed = list((y/temp_time)**2)
            elif len(y) > len(temp_time):
                yspeed = list((y[:len(temp_time)]/temp_time)**2)
            elif len(y) < len(temp_time):
                yspeed = list((y/temp_time[:len(y)])**2)
            self.data.at[ind, 'speed'] = np.sqrt(xspeed + yspeed).astype(object)
        
        # cluster obstacle positions
        kmeans_input = np.stack([self.data['wobstacle_x_cm'].map(np.nanmean), self.data['wobstacle_y_cm'].map(np.nanmean)])
        kmeans_mask = np.any(np.isnan(kmeans_input), axis=0)
        kmeans_input = kmeans_input[:,~kmeans_mask]
        labels = KMeans(n_clusters=self.num_clusters_to_use, random_state=0).fit(kmeans_input.T).labels_

        # transit time histogram
        plt.figure()
        plt.xlabel('transit time')
        plt.hist(self.data['len'], bins=25)
        pdf.savefig(); plt.close()

        # spatial positions of obstacle clusters
        plt.figure()
        for i in range(len(labels)):
            obstacles = kmeans_input[:,i]
            plt.plot(obstacles[0], obstacles[1], '*', color=list(mcolors.TABLEAU_COLORS)[labels[i]])
            plt.ylim([20.03, 0]); plt.xlim([0, 33.30])
        pdf.savefig(); plt.close()

        # add kmeans labels to trial data
        self.data = self.data[~pd.isnull(self.data['wobstacle_x_cm'])][~pd.isnull(self.data['wobstacle_y_cm'])]
        self.data['obstacle_cluster'] = labels

        # drop any transits that were really slow (only drop slowest 10% of transits)
        time_thresh = self.data['len'].quantile(0.9)
        self.data = self.data[self.data['len']<time_thresh]

        # plot all the traces
        odd = self.data[self.data.index%2==0]
        even = self.data[self.data.index%2==1]
        direction_count = 0
        for direction_df in [odd, even]:
            if direction_count == 0:
                leftcolor='g'; rightcolor='b'
            else:
                leftcolor='b'; rightcolor='g'
            plt.subplots(3,3, figsize=(9,6))
            for c in range(9):
                this_cluster = direction_df[direction_df['obstacle_cluster']==c].copy().reset_index()
                plt.subplot(3,3,c+1)
                plt.gca().set_aspect('equal', adjustable='box')
                colors = plt.cm.magma(np.linspace(0,1,len(this_cluster)))
                for ind, row in this_cluster.iterrows():
                    plt.plot([np.median(row['obstaclewTL_x_cm'],0),
                            np.median(row['obstaclewTR_x_cm'],0),
                            np.median(row['obstaclewBR_x_cm'],0),
                            np.median(row['obstaclewBL_x_cm'],0),
                            np.median(row['obstaclewTL_x_cm'],0)],
                            [np.median(row['obstaclewTL_y_cm'],0),
                            np.median(row['obstaclewTR_y_cm'],0),
                            np.median(row['obstaclewBR_y_cm'],0),
                            np.median(row['obstaclewBL_y_cm'],0),
                            np.median(row['obstaclewTL_y_cm'],0)],'k-')
                    plt.plot([np.median(row['arenaTL_x_cm'],0),
                            np.median(row['arenaTR_x_cm'],0),
                            np.median(row['arenaBR_x_cm'],0),
                            np.median(row['arenaBL_x_cm'],0),
                            np.median(row['arenaTL_x_cm'],0)],
                            [np.median(row['arenaTL_y_cm'],0),
                            np.median(row['arenaTR_y_cm'],0),
                            np.median(row['arenaBR_y_cm'],0),
                            np.median(row['arenaBL_y_cm'],0),
                            np.median(row['arenaTL_y_cm'],0)],'k-')
                    plt.plot(row['nose_x_cm'], row['nose_y_cm'], '-', color=colors[ind])
                    plt.plot(row['leftportT_x_cm'], row['leftportT_y_cm'],'.',color=leftcolor)
                    plt.plot(row['rightportT_x_cm'], row['rightportT_y_cm'],'.',color=rightcolor)
                plt.ylim([20.03,0]); plt.xlim([0,33.30])
                if len(this_cluster) > 0:
                    plt.plot(self.get_median_trace(this_cluster)['median_x_cm'].iloc[0], self.get_median_trace(this_cluster)['median_y_cm'].iloc[0], 'b-')
            direction_count += 1
            plt.tight_layout()
            pdf.savefig(); plt.close()

        # overlay of cluster medians
        direction_count = 0
        plt.subplots(1,2, figsize=(8,5))
        for direction_df in [odd, even]:
            if direction_count == 0:
                leftcolor='g'; rightcolor='b'
            else:
                leftcolor='b'; rightcolor='g'
            plt.subplot(1,2,direction_count+1)
            for c in range(9):
                cluster_color = list(mcolors.TABLEAU_COLORS)[c]
                this_cluster = direction_df[direction_df['obstacle_cluster']==c].copy().reset_index()
                for ind, row in this_cluster.iterrows():
                    plt.plot([np.median(row['obstaclewTL_x_cm'],0),
                            np.median(row['obstaclewTR_x_cm'],0),
                            np.median(row['obstaclewBR_x_cm'],0),
                            np.median(row['obstaclewBL_x_cm'],0),
                            np.median(row['obstaclewTL_x_cm'],0)],
                            [np.median(row['obstaclewTL_y_cm'],0),
                            np.median(row['obstaclewTR_y_cm'],0),
                            np.median(row['obstaclewBR_y_cm'],0),
                            np.median(row['obstaclewBL_y_cm'],0),
                            np.median(row['obstaclewTL_y_cm'],0)],'-', color=cluster_color)
                    plt.plot(row['leftportT_x_cm'], row['leftportT_y_cm'], '.', color=leftcolor)
                    plt.plot(row['rightportT_x_cm'], row['rightportT_y_cm'], '.', color=rightcolor)
                plt.ylim([20.03,0]); plt.xlim([0,33.30])
                if len(this_cluster) > 0:
                    plt.plot(self.get_median_trace(this_cluster)['median_x_cm'].iloc[0], self.get_median_trace(this_cluster)['median_y_cm'].iloc[0], '-', color=cluster_color)
            direction_count += 1
            plt.tight_layout()
            pdf.savefig(); plt.close()

        # properties over time (e.g. time active, speed, etc.)
        plt.subplots(2,3, figsize=(15,8))
        median_speed = []; max_speed = []; time_active = []
        for ind, row in self.data.iterrows():
            median_speed.append(np.median(row['speed']))
            max_speed.append(np.max(row['speed']))
            time_active.append(np.sum(row['speed']>5)/60)
        plt.subplot(2,3,1)
        slow = self.data['speed'].iloc[np.nanargmin(median_speed)]
        slowT = np.linspace(0,1,len(slow))
        med = self.data['speed'].iloc[np.argsort(median_speed)[len(median_speed)//2]]
        medT = np.linspace(0,1,len(med))
        fast = self.data['speed'].iloc[np.nanargmax(median_speed)]
        fastT = np.linspace(0,1,len(fast))
        fake_time = np.linspace(0,1,100)
        plt.plot(interp1d(slowT, slow, bounds_error=False)(fake_time))
        plt.plot(interp1d(medT, med, bounds_error=False)(fake_time))
        plt.plot(interp1d(fastT, fast, bounds_error=False)(fake_time))
        plt.subplot(2,3,2)
        plt.hist(median_speed); plt.xlabel('median speed (cm/sec)')
        plt.subplot(2,3,3)
        plt.hist(max_speed); plt.xlabel('max speed (cm/sec)')
        plt.subplot(2,3,4)
        plt.hist(time_active); plt.xlabel('time active (sec)')
        plt.subplot(2,3,5)
        plt.plot([i for i in max_speed if ~np.isnan(i)]); plt.xlabel('trial'); plt.ylabel('max speed (cm/sec)')
        plt.subplot(2,3,6)
        plt.plot([i for i in time_active if ~np.isnan(i)]); plt.xlabel('trial'); plt.ylabel('time active (sec)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.subplots(4,4,figsize=(20,18))
        fake_time = np.linspace(0,1,100)
        plt.subplot(4,4,1)
        for ind, data in self.data['speed'].iteritems():
            plt.plot(interp1d(np.linspace(0,1,len(data)), data, bounds_error=False)(fake_time), '-', alpha=0.2); plt.ylabel('speed'); plt.xlabel('time')
        plt.subplot(4,4,2)
        for ind, data in self.data['head_angle'].iteritems():
            plt.plot(interp1d(np.linspace(0,1,len(data)), data, bounds_error=False)(fake_time), '-', alpha=0.2); plt.ylabel('angle to horizontal'); plt.xlabel('time')
        odd = self.data[self.data.index%2==0]
        even = self.data[self.data.index%2==1]
        for direction_num in range(2):
            plt.subplot(4,4,3+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                dist = self.distance_from_nose(row, 'leftportT')
                plt.plot(interp1d(np.linspace(0,1,len(dist)), dist, bounds_error=False)(fake_time), alpha=0.2)
                if self.approaching_target(dist) is True:
                    plt.title('distance to target port (moving left)')
                    direction_df.at[ind, 'approach'] = True
                else:
                    plt.title('distance to previous port (moving left)')
                    direction_df.at[ind, 'approach'] = False
                plt.plot(0,dist[0],'.',color='g')
                plt.plot(100,dist[-1],'.',color='b')
        for direction_num in range(2):
            plt.subplot(4,4,5+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                dist = self.distance_from_nose(row, 'rightportT')
                plt.plot(interp1d(np.linspace(0,1,len(dist)), dist, bounds_error=False)(fake_time), alpha=0.2)
                if row['approach']:
                    plt.title('distance to previous port (moving right)')
                else:
                    plt.title('distance to target port (moving right)')
                plt.plot(0,dist[0],'.',color='g')
                plt.plot(100,dist[-1],'.',color='b')
        for direction_num in range(2):
            plt.subplot(4,4,7+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                ang = self.angle_from_nose(row, 'leftportT')
                plt.plot(interp1d(np.linspace(0,1,len(ang)), ang, bounds_error=False)(fake_time), alpha=0.2)
                if row['approach']:
                    plt.title('angle to target port (headed left')
                else:
                    plt.title('angle to previous port (headed left)')
                plt.plot(0,ang[0],'.',color='g')
                plt.plot(100,ang[-1],'.',color='b')
        for direction_num in range(2):
            plt.subplot(4,4,9+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                ang = self.angle_from_nose(row, 'rightportT')
                plt.plot(interp1d(np.linspace(0,1,len(ang)), ang, bounds_error=False)(fake_time), alpha=0.2)
                if row['approach']:
                    plt.title('angle to target port')
                else:
                    plt.title('angle to previous port')
                plt.plot(0,ang[0],'.',color='g')
                plt.plot(100,ang[-1],'.',color='b')
        for direction_num in range(2):
            plt.subplot(4,4,11+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                dist = self.distance_from_nose(row, 'wobstacle')
                plt.plot(interp1d(np.linspace(0,1,len(dist)), dist, bounds_error=False)(fake_time), alpha=0.2)
                plt.title('distance to obstacle')
        for direction_num in range(2):
            plt.subplot(4,4,13+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                ang = self.angle_from_nose(row, 'wobstacle')
                plt.plot(interp1d(np.linspace(0,1,len(ang)), ang, bounds_error=False)(fake_time), alpha=0.2)
                plt.title('angle to obstacle')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        pdf.close()
        self.data.to_hdf(os.path.join(self.trial_path, (self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')

    def format_frames_oa(self, vid_path):
        # open the .avi file
        vidread = cv2.VideoCapture(vid_path)
        # empty array that is the target shape
        # should be number of frames x downsampled height x downsampled width
        all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                            int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))], dtype=np.uint8)
        # iterate through each frame
        for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
            # read the frame in and make sure it is read in correctly
            ret, frame = vidread.read()
            if not ret:
                break
            # convert to grayyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # add the downsampled frame to all_frames as int8
            all_frames[frame_num,:,:] = frame.astype(np.int8)
        return all_frames

    def get_row_for_timestamp(self, df, seek_timestamp):
        for ind, row in df.iterrows():
            if seek_timestamp in row['trial_timestamps']:
                return row

    def plot_frame(self, vid_arr, timestamps, df, seek_frame, return_as_array=False):
        seek_timestamp = timestamps[seek_frame]
        row = self.get_row_for_timestamp(df, seek_timestamp)
        if row is None:
            if return_as_array:
                return np.zeros(np.shape(vid_arr[0]))
            elif not return_as_array:
                plt.figure()
                plt.imshow(np.zeros(np.shape(vid_arr[0])), cmap='gray')
                plt.show()
        row_time_index = np.where(row['trial_timestamps']==seek_timestamp)
        current_ang = row['head_angle'][row_time_index][0]
        x1 = row['nose_x'][row_time_index]
        y1 = row['nose_y'][row_time_index]
        x2 = x1+60 * np.cos(current_ang)
        y2 = y1+60 * np.sin(current_ang)
        frame = vid_arr[seek_frame,:,:]
        fig = plt.figure()
        plt.imshow(frame, cmap='gray')
        plt.plot((x1,x2), (y1,y2), '-')
        row_time_index = row_time_index[0][0]
        plt.plot(row['nose_x'][:row_time_index], row['nose_y'][:row_time_index],'r.')
        plt.plot(row['leftear_x'][:row_time_index], row['leftear_y'][:row_time_index], 'g.')
        plt.plot(row['rightear_x'][:row_time_index], row['rightear_y'][:row_time_index], 'g.')
        if not return_as_array:
            plt.show()
        elif return_as_array:
            fig.canvas.draw()
            frame_as_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            frame_as_array = frame_as_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return frame_as_array

    def plot_all_trials(self, vid_arr, timestamps, df, vid_savepath):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vid = cv2.VideoWriter(vid_savepath, fourcc, 60.0, (640, 480))
        start = 3600*5
        for seek_frame in tqdm(range(start,start+3600)):
            frame = self.plot_frame(vid_arr, timestamps, df, seek_frame, return_as_array=True)
            out_vid.write(frame.astype('uint8'))
        out_vid.release()

    def make_videos(self):
        vid_savepath = os.path.join(self.trial_path, (self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'plot.avi'))
        vid_path = find('*'+str(self.s['date'])+'*'+self.s['animal']+'*'+str(self.s['task'])+'*.avi', self.trial_path)[0]
        self.timestamp_path = find('*'+str(self.s['date'])+'*'+self.s['animal']+'*'+str(self.s['task'])+'*_top1_BonsaiTS.csv', self.trial_path)[0]
        print('formating video frames as array')
        vid_arr = self.format_frames_oa(vid_path)
        print('plotting video of traces')
        self.plot_all_trials(vid_arr, self.read_timestamp_file(), self.data, vid_savepath)

    def gap_detection(self):

        self.make_task_df()

        dist_to_posts = np.median(self.data['arenaTR_x'].iloc[0],0) - np.median(self.data['arenaTL_x'].iloc[0],0)
        self.pxls2cm = dist_to_posts/self.dist_across_arena
        self.convert_pxls_to_dist()

        self.get_head_angle()

        pdf = PdfPages(os.path.join(self.trial_path, (self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'_figs.pdf')))

        self.timestamp_path = find('*'+str(self.s['date'])+'*'+self.s['animal']+'*'+str(self.s['task'])+'*_top1_BonsaiTS.csv', self.trial_path)[0]

        # get gap positions
        sq_names = ['sBR','sBL','sTR','sTL','s2BR','s2BL','s2TR','s2TL']
        tr_names = ['tBL','tTL','tM','t2BL','t2TL','t2M']
        for ind, row in self.data.iterrows():
            # first gap
            sq_x_positions = np.zeros(8)
            sq_y_positions = np.zeros(8)
            sq_std = np.zeros(8)
            for i, sq_name in enumerate(sq_names):
                x_positions = row['obstacle'+sq_name+'_x_cm'].astype(float)
                y_positions = row['obstacle'+sq_name+'_y_cm'].astype(float)
                if ~np.isnan(x_positions).all() > 0 and ~np.isnan(y_positions).all() > 0:
                    sq_x_positions[i] = np.nanmedian(x_positions)
                    sq_y_positions[i] = np.nanmedian(y_positions)
                    sq_std[i] = np.mean([np.nanstd(x_positions),np.nanstd(y_positions)])
            self.data.at[ind, 'obs_sq_x_pos'] = sq_x_positions.astype(object)
            self.data.at[ind, 'obs_sq_y_pos'] = sq_y_positions.astype(object)
            self.data.at[ind, 'obs_sq_std'] = sq_std.astype(object)
            
            tr_x_positions = np.zeros(6)
            tr_y_positions = np.zeros(6)
            tr_std = np.zeros(6)
            for i, tr_name in enumerate(tr_names):
                x_positions = row['obstacle'+tr_name+'_x_cm'].astype(float)
                y_positions = row['obstacle'+tr_name+'_y_cm'].astype(float)
                if ~np.isnan(x_positions).all() > 0 and ~np.isnan(y_positions).all() > 0:
                    tr_x_positions[i] = np.nanmedian(x_positions)
                    tr_y_positions[i] = np.nanmedian(y_positions)
                    tr_std[i] = np.mean([np.nanstd(x_positions),np.nanstd(y_positions)])
            self.data.at[ind, 'obs_tr_x_pos'] = tr_x_positions.astype(object)
            self.data.at[ind, 'obs_tr_y_pos'] = tr_y_positions.astype(object)
            self.data.at[ind, 'obs_tr_std'] = tr_std.astype(object)
            self.data.at[ind, 'might_be_bad'] = (True if (np.sum(~np.isnan(tr_x_positions[:4]))<3 or np.sum(~np.isnan(sq_x_positions[:5]))<4) else False)
            has_both_doors = (True if (np.sum(~np.isnan(tr_x_positions))>3 and np.sum(~np.isnan(sq_x_positions))>4) else False)
            door1 = np.array([np.nanmean(sq_x_positions[2:4]), np.nanmean(sq_y_positions[2:4]), np.nanmean(tr_x_positions[:2]), np.nanmean(tr_y_positions[:2])])
            self.data.at[ind, 'door1'] = door1.astype(object)
            self.data.at[ind, 'door1_cent'] = np.array([np.nanmean([door1[0], door1[2]]), np.nanmean([door1[1], door1[3]])]).astype(object)
            if has_both_doors:
                self.data.at[ind, 'has_door2'] = has_both_doors
                door2 = np.array([np.nanmean(sq_x_positions[6:]), np.nanmean(sq_y_positions[6:]), np.nanmean(tr_x_positions[3:5]), np.nanmean(tr_y_positions[3:5])])
                self.data.at[ind, 'door2'] = door2.astype(object)
                self.data.at[ind, 'door2_cent'] = np.array([np.nanmean([door2[0], door2[2]]), np.nanmean([door2[1], door2[3]])]).astype(object)

        # animal speed
        for ind, row in self.data.iterrows():
            temp_time = np.diff(row['trial_timestamps'])
            x = np.diff(row['nose_x_cm']); y = np.diff(row['nose_y_cm'])
            if len(x) == len(temp_time):
                xspeed = list((x/temp_time)**2)
            elif len(x) > len(temp_time):
                xspeed = list((x[:len(temp_time)]/temp_time)**2)
            elif len(x) < len(temp_time):
                xspeed = list((x/temp_time[:len(x)])**2)
            if len(y) == len(temp_time):
                yspeed = list((y/temp_time)**2)
            elif len(y) > len(temp_time):
                yspeed = list((y[:len(temp_time)]/temp_time)**2)
            elif len(y) < len(temp_time):
                yspeed = list((y/temp_time[:len(y)])**2)
            self.data.at[ind, 'speed'] = np.sqrt(xspeed + yspeed).astype(object)

        kmeans_input = np.concatenate([flatten_series(self.data['obs_sq_x_pos']),
                                       flatten_series(self.data['obs_sq_y_pos']),
                                       flatten_series(self.data['obs_tr_x_pos']),
                                       flatten_series(self.data['obs_tr_y_pos'])], axis=1)

        kmeans_input[np.isnan(kmeans_input)] = 0
        labels = KMeans(n_clusters=self.num_clusters_to_use).fit(kmeans_input).labels_
        
        plt.subplots(2,1)
        plt.subplot(2,1,1)
        plt.hist(labels)
        plt.subplot(2,1,2)
        plt.plot(labels)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.subplots(10,int(np.ceil(len(labels)/10)),figsize=(30,30))
        sq_idx = [0,1,3,2,4]
        for c, l in enumerate(labels):
            plt.subplot(10,int(np.ceil(len(labels)/10)),c+1)
            obstacles = kmeans_input[c,:]
            obstacles[obstacles==0] = np.nan
            plt.plot(np.append(obstacles[:4], obstacles[0])[sq_idx], np.append(obstacles[8:12], obstacles[8])[sq_idx], 'k-')
            plt.plot(np.append(obstacles[4:8], obstacles[4])[sq_idx], np.append(obstacles[12:16], obstacles[12])[sq_idx], 'k-')
            plt.plot(np.append(obstacles[16:19], obstacles[16]), np.append(obstacles[22:25], obstacles[22]), 'k-')
            plt.plot(np.append(obstacles[19:22], obstacles[19]), np.append(obstacles[25:], obstacles[25]), 'k-')
            door1 = self.data['door1'].iloc[c]
            plt.plot([door1[0], door1[2]], [door1[1], door1[3]], 'g-', linewidth=6)
            if self.data['has_door2'].iloc[c]:
                door2 = self.data['door2'].iloc[c]
                plt.plot([door2[0], door2[2]], [door2[1], door2[3]], 'g-', linewidth=6)
            plt.title(l); plt.ylim([20.03,0]); plt.xlim([0,33.30])
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        self.data['obstacle_cluster'] = labels

        plt.subplots(2,int(np.ceil(self.num_clusters_to_use/2)),figsize=(20,5))
        for c in range(self.num_clusters_to_use):
            plt.subplot(2,int(np.ceil(self.num_clusters_to_use/2)),c+1)
            this_cluster = self.data[self.data['obstacle_cluster']==c].copy().reset_index()
            colors = plt.cm.magma(np.linspace(0,1,len(this_cluster)))
            mediantrace = self.get_median_trace(this_cluster)
            for ind, row in this_cluster.iterrows():
                plt.plot(mediantrace['nose_x_cm'].iloc[ind], mediantrace['nose_y_cm'].iloc[ind], '-', color=colors[ind])
            if len(this_cluster['door1']) > 1:
                door1 = np.nanmedian(flatten_series(this_cluster['door1']),0)
                plt.plot([door1[0], door1[2]], [door1[1], door1[3]], 'g-', linewidth=6)
            if len(this_cluster['door1']) > 1 and this_cluster['has_door2'].iloc[0]:
                door2 = np.nanmean(flatten_series(this_cluster['door2']),0)
                plt.plot([door2[0], door2[2]], [door2[1], door2[3]], 'g-', linewidth=6)
            plt.title(c)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # overlay of cluster medians
        odd = self.data[self.data.index%2==0]
        even = self.data[self.data.index%2==1]
        direction_count = 0
        plt.subplots(1,2, figsize=(8,5))
        for direction_df in [odd, even]:
            if direction_count == 0:
                leftcolor='g'; rightcolor='b'
            else:
                leftcolor='b'; rightcolor='g'
            plt.subplot(1,2,direction_count+1)
            for c in range(self.num_clusters_to_use):
                cluster_color = plt.cm.magma(np.linspace(0,1,self.num_clusters_to_use))[c]
                this_cluster = direction_df[direction_df['obstacle_cluster']==c].copy().reset_index()
                for ind, row in this_cluster.iterrows():
                    door1 = np.nanmedian(flatten_series(this_cluster['door1']),0)
                    plt.plot([door1[0], door1[2]], [door1[1], door1[3]], '-', linewidth=6, color=cluster_color)
                    if this_cluster['has_door2'].iloc[0]:
                        door2 = np.nanmean(flatten_series(this_cluster['door2']),0)
                        plt.plot([door2[0], door2[2]], [door2[1], door2[3]], '-', linewidth=6, color=cluster_color)
                    plt.plot(row['leftportT_x_cm'], row['leftportT_y_cm'], '.', color=leftcolor)
                    plt.plot(row['rightportT_x_cm'], row['rightportT_y_cm'], '.', color=rightcolor)
                plt.ylim([20.03,0]); plt.xlim([0,33.30])
                if len(this_cluster) > 0:
                    plt.plot(self.get_median_trace(this_cluster)['median_x_cm'].iloc[0], self.get_median_trace(this_cluster)['median_y_cm'].iloc[0], '-', color=cluster_color)
            direction_count += 1
            plt.tight_layout()
            pdf.savefig(); plt.close()

        # properties over time (e.g. time active, speed, etc.)
        plt.subplots(2,3, figsize=(15,8))
        median_speed = []; max_speed = []; time_active = []
        for ind, row in self.data.iterrows():
            median_speed.append(np.median(row['speed']))
            max_speed.append(np.max(row['speed']))
            time_active.append(np.sum(row['speed']>5)/60)
            self.data.at[ind, 'median_speed'] = np.median(row['speed'])
            self.data.at[ind, 'max_speed'] = np.max(row['speed'])
            self.data.at[ind, 'time_active'] = np.sum(row['speed']>5)/60
        plt.subplot(2,3,1)
        slow = self.data['speed'].iloc[np.nanargmin(median_speed)]
        slowT = np.linspace(0,1,len(slow))
        med = self.data['speed'].iloc[np.argsort(median_speed)[len(median_speed)//2]]
        medT = np.linspace(0,1,len(med))
        fast = self.data['speed'].iloc[np.nanargmax(median_speed)]
        fastT = np.linspace(0,1,len(fast))
        fake_time = np.linspace(0,1,100)
        plt.plot(interp1d(slowT, slow, bounds_error=False)(fake_time))
        plt.plot(interp1d(medT, med, bounds_error=False)(fake_time))
        plt.plot(interp1d(fastT, fast, bounds_error=False)(fake_time))
        plt.subplot(2,3,2)
        plt.hist(median_speed); plt.xlabel('median speed (cm/sec)')
        plt.subplot(2,3,3)
        plt.hist(max_speed); plt.xlabel('max speed (cm/sec)')
        plt.subplot(2,3,4)
        plt.hist(time_active); plt.xlabel('time active (sec)')
        plt.subplot(2,3,5)
        plt.plot([i for i in max_speed if ~np.isnan(i)]); plt.xlabel('trial'); plt.ylabel('max speed (cm/sec)')
        plt.subplot(2,3,6)
        plt.plot([i for i in time_active if ~np.isnan(i)]); plt.xlabel('trial'); plt.ylabel('time active (sec)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()


        plt.subplots(4,4,figsize=(20,18))
        fake_time = np.linspace(0,1,100)
        plt.subplot(4,4,1)
        for ind, data in self.data['speed'].iteritems():
            plt.plot(interp1d(np.linspace(0,1,len(data)), data, bounds_error=False)(fake_time), '-', alpha=0.2); plt.ylabel('speed'); plt.xlabel('time')
        plt.subplot(4,4,2)
        for ind, data in self.data['head_angle'].iteritems():
            plt.plot(interp1d(np.linspace(0,1,len(data)), data, bounds_error=False)(fake_time), '-', alpha=0.2); plt.ylabel('angle to horizontal'); plt.xlabel('time')
        odd = self.data[self.data.index%2==0]
        even = self.data[self.data.index%2==1]
        for direction_num in range(2):
            plt.subplot(4,4,3+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                dist = np.array(self.distance_from_nose(row, 'leftportT'))
                plt.plot(interp1d(np.linspace(0,1,len(dist)), dist, bounds_error=False)(fake_time), alpha=0.2)
                if self.approaching_target(dist) is True:
                    plt.title('distance to target port (moving left)')
                    direction_df.at[ind, 'approach'] = True
                else:
                    plt.title('distance to previous port (moving left)')
                    direction_df.at[ind, 'approach'] = False
                plt.plot(0,dist[0],'.',color='g')
                plt.plot(100,dist[-1],'.',color='b')
                if direction_num==0 and self.approaching_target(dist):
                    savename = 'odd_dist_to_target_port'
                elif direction_num==0 and not self.approaching_target(dist):
                    savename = 'odd_dist_to_previous_port'
                elif direction_num==1 and self.approaching_target(dist):
                    savename = 'even_dist_to_target_port'
                elif direction_num==1 and not self.approaching_target(dist):
                    savename = 'even_dist_to_previous_port'
                self.data.at[ind, savename] = dist.astype(object)
        for direction_num in range(2):
            plt.subplot(4,4,5+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                dist = np.array(self.distance_from_nose(row, 'rightportT'))
                plt.plot(interp1d(np.linspace(0,1,len(dist)), dist, bounds_error=False)(fake_time), alpha=0.2)
                if self.approaching_target(dist):
                    plt.title('distance to previous port (moving right)')
                else:
                    plt.title('distance to target port (moving right)')
                plt.plot(0,dist[0],'.',color='g')
                plt.plot(100,dist[-1],'.',color='b')
                if direction_num==0 and self.approaching_target(dist):
                    savename = 'odd_dist_to_target_port'
                elif direction_num==0 and not self.approaching_target(dist):
                    savename = 'odd_dist_to_previous_port'
                elif direction_num==1 and self.approaching_target(dist):
                    savename = 'even_dist_to_target_port'
                elif direction_num==1 and not self.approaching_target(dist):
                    savename = 'even_dist_to_previous_port'
                self.data.at[ind, savename] = dist.astype(object)
        for direction_num in range(2):
            plt.subplot(4,4,7+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                ang = np.array(self.angle_from_nose(row, 'leftportT'))
                plt.plot(interp1d(np.linspace(0,1,len(ang)), ang, bounds_error=False)(fake_time), alpha=0.2)
                if row['approach']:
                    plt.title('angle to target port (headed left')
                else:
                    plt.title('angle to previous port (headed left)')
                plt.plot(0,ang[0],'.',color='g')
                plt.plot(100,ang[-1],'.',color='b')
                if direction_num==0 and row['approach']:
                    savename = 'odd_ang_to_target_port'
                elif direction_num==0 and not row['approach']:
                    savename = 'odd_ang_to_previous_port'
                elif direction_num==1 and row['approach']:
                    savename = 'even_ang_to_target_port'
                elif direction_num==1 and not row['approach']:
                    savename = 'even_ang_to_previous_port'
                self.data.at[ind, savename] = ang.astype(object)
        for direction_num in range(2):
            plt.subplot(4,4,9+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                ang = np.array(self.angle_from_nose(row, 'rightportT'))
                plt.plot(interp1d(np.linspace(0,1,len(ang)), ang, bounds_error=False)(fake_time), alpha=0.2)
                if row['approach']:
                    plt.title('angle to target port')
                else:
                    plt.title('angle to previous port')
                plt.plot(0,ang[0],'.',color='g')
                plt.plot(100,ang[-1],'.',color='b')
                if direction_num==0 and row['approach']:
                    savename = 'odd_ang_to_target_port'
                elif direction_num==0 and not row['approach']:
                    savename = 'odd_ang_to_previous_port'
                elif direction_num==1 and row['approach']:
                    savename = 'even_ang_to_target_port'
                elif direction_num==1 and not row['approach']:
                    savename = 'even_ang_to_previous_port'
                self.data.at[ind, savename] = ang.astype(object)
        for direction_num in range(2):
            plt.subplot(4,4,11+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                dist = np.array(self.distance_from_nose(row, 'door1_cent', xy_together=True))
                plt.plot(interp1d(np.linspace(0,1,len(dist)), dist, bounds_error=False)(fake_time), alpha=0.2)
                plt.title('distance to door1')
                if direction_num==0:
                    savename = 'odd_dist_to_door1'
                else:
                    savename = 'even_dist_to_door1'
                self.data.at[ind, savename] = dist.astype(object)
        for direction_num in range(2):
            plt.subplot(4,4,13+direction_num)
            direction_df = [odd, even][direction_num]
            for ind, row in direction_df.iterrows():
                if row['has_door2']:
                    ang = np.array(self.angle_from_nose(row, 'door1_cent', xy_together=True))
                    plt.plot(interp1d(np.linspace(0,1,len(ang)), ang, bounds_error=False)(fake_time), alpha=0.2)
                    plt.title('angle to door1')
                    if direction_num==0:
                        savename = 'odd_ang_to_door1'
                    else:
                        savename = 'even_ang_to_door1'
                    self.data.at[ind, savename] = ang.astype(object)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        pdf.close()
        self.data.to_hdf(os.path.join(self.trial_path, (self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')

class AvoidanceSession(BaseInput):
    def __init__(self, metadata_path, task='oa'):
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.path = self.metadata['path']
        self.dates_list = [i for i in list(self.metadata.keys()) if i != 'path']

        if task=='oa':
            self.is_pillar_avoidance = True
            self.is_gap_detection = False
        elif task=='gd':
            self.is_pillar_avoidance = False
            self.is_gap_detection = True

        if self.is_pillar_avoidance:
            self.dlc_project = {'light':'/home/niell_lab/Documents/deeplabcut_projects/object_avoidance-Mike-2021-08-31/config.yaml',
                                'dark':None}
        elif self.is_gap_detection:
            self.dlc_project = {'light':'/home/niell_lab/Documents/deeplabcut_projects/gap_determination-Kana-2021-10-19/config.yaml',
                                'dark':'/home/niell_lab/Documents/deeplabcut_projects/dark_gap_determination-Kana-2021-11-08/config.yaml'}
        self.camname = 'top1'
        self.generic_camconfig = {
            'paths': {
                'dlc_projects': {
                    self.camname: self.dlc_project
                },
            },
            'internals': {
                'follow_strict_naming': False,
                'crop_for_dlc': False,
                'filter_dlc_predictions': False,
                'multianimal_top_project': False,
                'likelihood_threshold': 0.99
            }
        }

    def change_dlc_project(self, project_paths):
        """
        project_paths should be a dict like... {'light':/path/to/config.yaml, 'dark':/path/to/config.yaml}
        """
        # update object
        self.dlc_project = project_paths
        # also update config
        self.generic_camconfig['paths']['dlc_projects'][self.camname] = self.dlc_project

    def preprocess(self):
        for date in self.dates_list:
            date_dir = os.path.join(self.path, date)
            for animal in [i for i in list(self.metadata[date].keys())]:
                animal_dir = os.path.join(date_dir, animal)
                camconfig = self.generic_camconfig
                camconfig['animal_directory'] = animal_dir
                isdark = ('dark' in animal_dir)
                if not isdark:
                    print('using light network')
                    camconfig['paths']['dlc_projects'][self.camname] = self.dlc_project['light']
                elif isdark:
                    print('using dark network')
                    camconfig['paths']['dlc_projects'][self.camname] = self.dlc_project['dark']
                for recording_name in [k for k,v in self.metadata[date][animal].items()]:
                    recording_dir = os.path.join(animal_dir, recording_name)
                    name = '_'.join(os.path.splitext(os.path.split([i for i in find('*.avi', recording_dir) if all(bad not in i for bad in ['plot','IR','rep11','betafpv','side_gaze','._'])][0])[1])[0].split('_')[:-2])
                    tc = Topcam(config=camconfig, recording_name=name, recording_path=recording_dir, camname=self.camname)
                    tc.pose_estimation()
                    tc.gather_camera_files()
                    tc.pack_position_data()
                    tc.pack_video_frames()
                    tc.pt_names = list(tc.xrpts['point_loc'].values)
                    tc.filter_likelihood()
                    tc.get_head_body_yaw()
                    tc.save_params()

    def gather_all_sessions(self):
        data_dict = {'date': [],
                    'animal': [],
                    'task': [],             
                    'poke1_ts':[],
                    'poke2_ts': [],
                    'top1_ts': [],
                    'poke1_t0':[],
                    'poke2_t0': [],
                    'top1_t0': []}
        # list of dates for analysis
        data_path = Path(self.path).expanduser()
        # populate dict with metadata and timestamps
        for date in self.dates_list:
            use_animals = [k for k,v in self.metadata[date].items()]
            for ani in use_animals:
                for task in os.listdir(data_path / date / ani):
                    data_paths = [str(i) for i in list((data_path / date / ani/ task).rglob('*.csv'))]
                    data_paths = [i for i in data_paths if 'spout' not in i]
                    if data_paths != []:
                        _, name = os.path.split(data_paths[1])
                        split_name = name.split('_')
                        data_dict['date'].append(split_name[0])
                        data_dict['animal'].append(split_name[1])
                        data_dict['task'].append(split_name[4])
                    for ind, csv in enumerate(data_paths):
                        self.timestamp_path = csv
                        time = self.read_timestamp_file()
                        _, name = os.path.split(data_paths[ind])
                        split_name = name.split('_')
                        data_dict[split_name[5] +'_ts'].append(time)
                        data_dict[split_name[5] +'_t0'].append(time[0])
        self.all_sessions = pd.DataFrame.from_dict(data_dict)

    def process(self, videos=False):
        self.gather_all_sessions()
        for trial_ind, trial_row in tqdm(self.all_sessions.iterrows()):
            try:
                # analyze each trial
                trial = AvoidanceTrial(trial_row, self.path, self.metadata)
                dlc_h5 = find('*'+str(trial_row['date'])+'*'+trial_row['animal']+'*'+str(trial_row['task'])+'*.h5', self.path)
                if dlc_h5 == []:
                    continue
                trial_path, _ = os.path.split(dlc_h5[0])
                trial_name = '_'.join(os.path.splitext(os.path.split([i for i in find('*.avi', trial_path) if all(bad not in i for bad in ['plot','IR','rep11','betafpv','side_gaze','._'])][0])[1])[0].split('_')[:-1])
                trial.add_tracking(trial_name, trial_path)
                if self.is_pillar_avoidance:
                    trial.pillar_avoidance()
                elif self.is_gap_detection:
                    trial.gap_detection()
                # make short diagnostic video
                if videos:
                    trial.make_videos()
            except:
                pass