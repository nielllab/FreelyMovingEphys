"""
trace_by_clustered_obstacle_position.py
"""
import sys
sys.path.insert(0, '/home/niell_lab/Documents/github/FreelyMovingEphys/')
import pandas as pd
import xarray as xr
from utils.format_data import *
from utils.paths import find
from utils.time import open_time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os, argparse, json
import matplotlib.colors as mcolors
from project_analysis.object_avoidance.oa_utils import *
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_path', type=str, default='/home/niell_lab/data/object_avoidance/recordings/')
    parser.add_argument('--metadata_path', type=str, default='/home/niell_lab/data/object_avoidance/recordings/oa_metadata1.json')
    parser.add_argument('--dist_across_arena', type=float, default=30.48)
    parser.add_argument('--make_videos', type=bool, default=False)
    args = parser.parse_args()
    return args

def main(args):
    with open(args.metadata_path) as f:    
        metadata = json.load(f)

    dates_list = list(metadata.keys())

    df = make_oa_df(args.directory_path, dates_list)
    
    for oa_ind, oa_row in tqdm(df.iterrows()):

        trial_path = os.path.join(*[args.directory_path, str(oa_row['date']), oa_row['animal'],str(oa_row['task'])])

        dlc_h5 = find('*'+str(oa_row['date'])+'*'+oa_row['animal']+'*'+str(oa_row['task'])+'*.h5', trial_path)
    
        if dlc_h5 == []:
            continue

        df1 = make_task_df(df, oa_ind, dlc_h5[0])

        dist_to_posts = np.median(df1['arenaTR_x'].iloc[0],0) - np.median(df1['arenaTL_x'].iloc[0],0)
        pxls2cm = dist_to_posts/args.dist_across_arena
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

if __name__ == '__main__':
    args = get_args()
    main(args)