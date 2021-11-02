import json, os

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.base import BaseInput
from utils.base import Camera
from utils.aux_funcs import find, list_subdirs


class AvoidanceTask(BaseInput):
    def __init__(self, metadata_path):
        with open(metadata_path) as f:    
            self.metadata = json.load(f)

        self.path = self.metadata['path']
        self.dates_list = list(self.metadata.keys())

        self.likelihood_thresh = 0.99
        self.dist_across_arena = 30.48 # cm between bottom-right and bottom-left pillar
        self.camname = 'TOP1'
        self.generic_camconfig = {
            'paths': {
                'dlc_projects': {
                    self.camname: self.dlc_project
                },
            },
            'internals': {
                'follow_strict_naming': False
            }
        }

    def filter_likelihood(self, df):
        x_cols = [i for i in df.columns.values if '_x' in i and 'arena' not in i]
        y_cols = [i for i in df.columns.values if '_y' in i and 'arena' not in i]
        l_cols = [i for i in df.columns.values if '_likelihood' in i and 'arena' not in i]
        for i in range(len(x_cols)):
            x = df.loc[:,x_cols[i]]; y = df.loc[:,y_cols[i]]; l = df.loc[:,l_cols[i]]
            x[l<self.likelihood_thresh] = np.nan; y[l<self.likelihood_thresh] = np.nan
            df.loc[:,x_cols[i]] = x; df.loc[:,y_cols[i]] = y
        return df

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
            for ani in os.listdir(data_path / date): 
                for task in os.listdir(data_path / date/ ani):
                    data_paths = list((data_path / date/ ani/ task).rglob('*.csv'))
                    if data_paths != []:
                        data_dict['date'].append(data_paths[1].name.split('_')[0])
                        data_dict['animal'].append(data_paths[1].name.split('_')[1])
                        data_dict['task'].append(data_paths[1].name.split('_')[4])
                    for ind, csv in enumerate(data_paths):
                        time = self.read_timestamp_file(csv)
                        data_dict[data_paths[ind].name.split('_')[5] +'_ts'].append(time)
                        data_dict[data_paths[ind].name.split('_')[5] +'_t0'].append(time[0])
        self.all_sessions = pd.DataFrame.from_dict(data_dict)

    def preprocess(self, name, path):
        for date in self.dates_list:
            date_dir = os.path.join(path, date)
            for animal in list_subdirs(date_dir):
                animal_dir = os.path.join(date_dir, animal_dir)
                camconfig = self.generic_camconfig
                camconfig['animal_directory'] = animal_dir
                tc = Camera(camconfig, name, path, self.camname)
                tc.pose_estimation()
                tc.
        


class ObjectAvoidance(AvoidanceTask):
    def __init___(self, metadata_path):
        AvoidanceTask.__init__(self, metadata_path)
        self.dlc_project = '/home/niell_lab/Documents/deeplabcut_projects/gap_determination-Kana-2021-10-19/config.yaml'
        

class GapDetection(AvoidanceTask):
    def __init___(self, metadata_path):
        AvoidanceTask.__init__(self, metadata_path)
        self.dlc_project = '/home/niell_lab/Documents/deeplabcut_projects/gap_determination-Kana-2021-10-19/config.yaml'

    def process_recording(self, ):


    def batch_analysis(self):
        self.gather_all_sessions()
        for recording_ind, recording_data in tqdm(self.all_sessions.iterrows()):
            recording_path = os.path.join(*[self.path, str(recording_data['date']), recording_data['animal'],str(recording_data['task'])])
            dlc_h5 = find('*'+str(recording_data['date'])+'*'+recording_data['animal']+'*'+str(recording_data['task'])+'*.h5', recording_path)
            if dlc_h5 == []:
                continue # skip to next recording if DLC hasn't run for current recording
            

            

            df1 = make_task_df(recording_ind, dlc_h5[0])

            dist_to_posts = np.median(df1['arenaTR_x'].iloc[0],0) - np.median(df1['arenaTL_x'].iloc[0],0)
            pxls2cm = dist_to_posts / self.dist_across_arena
            df1 = convert_pxls_to_dist(df1, pxls2cm)
            df1 = get_head_angle(df1)

            num_clusters_to_use = metadata[oa_row['date']][oa_row['animal']][str(oa_row['task'])]['num_positions']

            savepath = trial_path

            pdf = PdfPages(os.path.join(savepath, (df1['animal'].iloc[0]+'_'+str(df1['date'].iloc[0])+'_'+str(df1['task'].iloc[0])+'_figs.pdf')))

            for ind, row in df1.iterrows():
                for x in ['b','w']:
                    xvals = np.stack([row['obstacle'+x+'TL_x_cm'], row['obstacle'+x+'TR_x_cm'], row['obstacle'+x+'BL_x_cm'], row['obstacle'+x+'BL_x_cm']]).astype(float)
                    df1.at[ind, x+'obstacle_x_cm'] = np.nanmean(xvals)
                    df1.at[ind, x+'obstacle_x_std'] = np.mean(np.nanstd(xvals, axis=1))
                    yvals = np.stack([row['obstaclewTL_y_cm'], row['obstaclewTR_y_cm'], row['obstaclewBL_y_cm'], row['obstaclewBL_y_cm']]).astype(float)
                    df1.at[ind, x+'obstacle_y_cm'] = np.nanmean(yvals)
                    df1.at[ind, x+'obstacle_y_std'] = np.mean(np.nanstd(yvals, axis=1))
            
            for ind, row in df1.iterrows():
                temp_time = np.diff(row['trail_timestamps'])
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
                df1.at[ind, 'speed'] = np.sqrt(xspeed + yspeed).astype(object)

            kmeans_input = np.stack([df1['wobstacle_x_cm'].map(np.nanmean), df1['wobstacle_y_cm'].map(np.nanmean)])
            kmeans_mask = np.any(np.isnan(kmeans_input), axis=0)
            kmeans_input = kmeans_input[:,~kmeans_mask]

            labels = KMeans(n_clusters=num_clusters_to_use, random_state=0).fit(kmeans_input.T).labels_

            plt.figure()
            plt.hist(df1['len'], bins=25)
            pdf.savefig(); plt.close()

            plt.figure()
            for i in range(len(labels)):
                label = labels[i]
                c = list(mcolors.TABLEAU_COLORS)[label]
                obstacles = kmeans_input[:,i]
                plt.plot(obstacles[0],obstacles[1],'*',color=c)
                plt.ylim([20.03,0]); plt.xlim([0,33.30])
            pdf.savefig(); plt.close()

            df1 = df1[~pd.isnull(df1['wobstacle_x_cm'])][~pd.isnull(df1['wobstacle_y_cm'])]
            df1['obstacle_cluster'] = labels

            time_thresh = df1['len'].quantile(0.9)
            df1 = df1[df1['len']<time_thresh]

            odd = df1[df1.index%2==0]
            even = df1[df1.index%2==1]
            direction_count = 0
            for direction_df in [odd, even]:
                if direction_count == 0:
                    leftcolor='g'; rightcolor='b'
                else:
                    leftcolor='b'; rightcolor='g'
                plt.subplots(3,3,figsize=(9,6))
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
                        plt.plot(get_median_trace(this_cluster)['median_x_cm'].iloc[0], get_median_trace(this_cluster)['median_y_cm'].iloc[0], 'b-')
                direction_count += 1
                plt.tight_layout()
                pdf.savefig(); plt.close()

            plt.subplots(2,3,figsize=(15,8))
            median_speed = []; max_speed = []; time_active = []
            for ind, row in df1.iterrows():
                median_speed.append(np.median(row['speed']))
                max_speed.append(np.max(row['speed']))
                time_active.append(np.sum(row['speed']>5)/60)
            plt.subplot(2,3,1)
            slow = df1['speed'].iloc[np.nanargmin(median_speed)]
            slowT = np.linspace(0,1,len(slow))
            med = df1['speed'].iloc[np.argsort(median_speed)[len(median_speed)//2]]
            medT = np.linspace(0,1,len(med))
            fast = df1['speed'].iloc[np.nanargmax(median_speed)]
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
            for ind, data in df1['speed'].iteritems():
                plt.plot(interp1d(np.linspace(0,1,len(data)), data, bounds_error=False)(fake_time), '-', alpha=0.2); plt.ylabel('speed'); plt.xlabel('time')
            plt.subplot(4,4,2)
            for ind, data in df1['head_angle'].iteritems():
                plt.plot(interp1d(np.linspace(0,1,len(data)), data, bounds_error=False)(fake_time), '-', alpha=0.2); plt.ylabel('angle to horizontal'); plt.xlabel('time')
            odd = df1[df1.index%2==0]
            even = df1[df1.index%2==1]
            for direction_num in range(2):
                plt.subplot(4,4,3+direction_num)
                direction_df = [odd, even][direction_num]
                for ind, row in direction_df.iterrows():
                    dist = distance_from_nose(row, 'leftportT')
                    plt.plot(interp1d(np.linspace(0,1,len(dist)), dist, bounds_error=False)(fake_time), alpha=0.2)
                    if approaching_target(dist) is True:
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
                    dist = distance_from_nose(row, 'rightportT')
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
                    ang = angle_from_nose(row, 'leftportT')
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
                    ang = angle_from_nose(row, 'rightportT')
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
                    dist = distance_from_nose(row, 'wobstacle')
                    plt.plot(interp1d(np.linspace(0,1,len(dist)), dist, bounds_error=False)(fake_time), alpha=0.2)
                    plt.title('distance to obstacle')
            for direction_num in range(2):
                plt.subplot(4,4,13+direction_num)
                direction_df = [odd, even][direction_num]
                for ind, row in direction_df.iterrows():
                    ang = angle_from_nose(row, 'wobstacle')
                    plt.plot(interp1d(np.linspace(0,1,len(ang)), ang, bounds_error=False)(fake_time), alpha=0.2)
                    plt.title('angle to obstacle')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            pdf.close()
            df1.to_hdf(os.path.join(savepath, (df1['animal'].iloc[0]+'_'+str(df1['date'].iloc[0])+'_'+str(df1['task'].iloc[0])+'.h5')), 'w')

            if args.make_videos:
                vid_savepath = os.path.join(savepath, (df1['animal'].iloc[0]+'_'+str(df1['date'].iloc[0])+'_'+str(df1['task'].iloc[0])+'plot.avi'))
                vid_path = find('*'+str(oa_row['date'])+'*'+oa_row['animal']+'*'+str(oa_row['task'])+'*.avi', trial_path)[0]
                time_path = find('*'+str(oa_row['date'])+'*'+oa_row['animal']+'*'+str(oa_row['task'])+'*_top1_BonsaiTS.csv', trial_path)[0]
                print('formating video frames as array')
                vid_arr = format_frames_oa(vid_path)
                print('plotting video of traces')
                plot_all_trials(vid_arr, open_time(time_path), df1, vid_savepath)


    def gap_detection(self):

    def make_video(self):













class DynamicalSteeringModel:
    def __init__(self, time, agent_x, agent_y, goal_x, goal_y, obstacle_x, obstacle_y):
        self.num_y_bins =  40
        self.agent_x = agent_x
        self.agent_y = agent_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.obstacle_x = obstacle_x
        self.obstacle_y = obstacle_y
        self.time = time

        self.goal_distance
        self.obstacle_distance
        self.turning_rate
        self.goal_attraction
        self.obstacle_repulsion
        self.b # dampening coefficient

    def calc_props(self):
        # goal distance

        # obstacle distance

        # turning rate

        # goal attraction

        # obstacle repulsion

        # dampening coeffeficient

        # agent speed
        self.agent_speed = np.sqrt((np.diff(self.agent_x)/self.time)**2 + (np.diff(self.agent_y)/self.time)**2)
        # heading, angle to goal, angle to obstacle
        self.heading = np.zeros(len(self.agent_x))
        self.goal_ang = np.zeros(len(self.agent_x))
        self.obstacle_ang = np.zeros(len(self.agent_x))
        for frame in range(len(self.agent_x)):
            self.heading = np.arctan((self.agent_x[frame] - self.agent_x[frame-1]) / (self.agent_y[frame] - self.agent_y[frame-1]))
            self.goal_ang = np.arctan((self.goal_x[frame] - self.agent_x[frame]) / (self.goal_y[frame] - self.agent_y[frame]))
            self.obstacle_ang = np.arctan((self.obstacle_x[frame] - self.agent_x[frame]) / (self.obstacle_y[frame] - self.agent_y[frame]))
        self.goal_ang_wrt_heading = self.heading - self.goal_ang
        self.obstacle_ang_wrt_heading = self.heading - self.obstacle_ang

    def calc_paths(self):
        # 
        self.agent_x_binned, self.agent_x_bin_edges = np.digitize(self.agent_x, bins=self.num_y_bins)

    def calc_ang_acc(self):
        self.ang_acc = -self.b*self.turning_rate - self.goal_attraction(self.heading - self.goal_direction) + self.obstacle_repulsion(self.heading - self.obstacle_direction) * np.exp(-self.heading - self.obstacle_direction)












