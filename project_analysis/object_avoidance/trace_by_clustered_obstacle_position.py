"""
trace_by_clustered_obstacle_position.py
"""
import sys
sys.path.insert(0, '/home/niell_lab/Documents/github/FreelyMovingEphys/')
import pandas as pd
import xarray as xr
from utils.format_data import *
from utils.paths import find
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os, argparse, json
import matplotlib.colors as mcolors
from project_analysis.object_avoidance.oa_utils import *
from matplotlib.backends.backend_pdf import PdfPages

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_path', type=str, default='/home/niell_lab/data/object_avoidance/recordings/')
    parser.add_argument('--metadata_path', type=str, default='/home/niell_lab/data/object_avoidance/recordings/oa_metadata1.json')
    parser.add_argument('--dist_across_arena', type=float, default=30.48)
    args = parser.parse_args()
    return args

def main(args):
    with open(args.metadata_path) as f:    
        metadata = json.load(f)

    dates_list = list(metadata.keys())

    df = make_oa_df(args.directory_path, dates_list)
    
    for oa_ind, oa_row in tqdm(df.iterrows()):

        dlc_h5 = find('*'+str(oa_row['date'])+'*'+oa_row['animal']+'*'+str(oa_row['task'])+'*.h5', args.directory_path)
    
        if dlc_h5 == []:
            continue

        df1 = make_task_df(df, oa_ind, dlc_h5[0])

        dist_to_posts = np.median(df1['arenaTR_x'].iloc[0],0) - np.median(df1['arenaTL_x'].iloc[0],0)
        pxls2cm = dist_to_posts/args.dist_across_arena
        df1 = convert_pxls_to_dist(df1, pxls2cm)
        
        num_clusters_to_use = metadata[oa_row['date']][oa_row['animal']][str(oa_row['task'])]['num_positions']

        savepath = os.path.split(dlc_h5[0])[0]

        pdf = PdfPages(os.path.join(savepath, (df1['animal'].iloc[0]+'_'+str(df1['date'].iloc[0])+'_'+str(df1['task'].iloc[0])+'_figs.pdf')))

        for ind, row in df1.iterrows():
            for x in ['b','w']:
                xvals = np.stack([row['obstacle'+x+'TL_x_cm'], row['obstacle'+x+'TR_x_cm'], row['obstacle'+x+'BL_x_cm'], row['obstacle'+x+'BL_x_cm']]).astype(float)
                df1.at[ind, x+'obstacle_x'] = np.nanmean(xvals)
                df1.at[ind, x+'obstacle_x_std'] = np.mean(np.nanstd(xvals, axis=1))
                yvals = np.stack([row['obstaclewTL_y_cm'], row['obstaclewTR_y_cm'], row['obstaclewBL_y_cm'], row['obstaclewBL_y_cm']]).astype(float)
                df1.at[ind, x+'obstacle_y'] = np.nanmean(yvals)
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

        kmeans_input = np.stack([df1['wobstacle_x'].map(np.nanmean), df1['wobstacle_y'].map(np.nanmean)])
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

        df1 = df1[~pd.isnull(df1['wobstacle_x'])][~pd.isnull(df1['wobstacle_y'])]
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
            direction_count += 1
            plt.tight_layout()
            pdf.savefig(); plt.close()
    
        pdf.close()
        df1.to_hdf(os.path.join(savepath, (df1['animal'].iloc[0]+'_'+str(df1['date'].iloc[0])+'_'+str(df1['task'].iloc[0])+'.h5')), 'w')

if __name__ == '__main__':
    args = get_args()
    main(args)