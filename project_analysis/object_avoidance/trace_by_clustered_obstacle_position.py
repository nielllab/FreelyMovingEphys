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
import os, argparse
import matplotlib.colors as mcolors
from project_analysis.object_avoidance.oa_utils import *
from matplotlib.backends.backend_pdf import PdfPages

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_path', type=str, default='/home/niell_lab/data/object_avoidance/preliminary_data/recordings/')
    parser.add_argument('--dates_list', type=list, default=['091321'])
    parser.add_argument('--clusters_list', type=list, default=[9,9,3,3])
    args = parser.parse_args()
    return args

def main(args):
    df = make_oa_df(args.directory_path, args.dates_list)
    for oa_ind, oa_row in tqdm(df.iterrows()):

        dlc_h5 = find('*'+str(oa_row['date'])+'*'+oa_row['animal']+'*'+str(oa_row['task'])+'*.h5', args.directory_path)
    
        if dlc_h5 == []:
            break

        df1 = make_task_df(df, oa_ind, dlc_h5[0])
        num_clusters_to_use = args.clusters_list[oa_ind]

        pdf = PdfPages(os.path.join(args.directory_path, (df1['animal'].iloc[0]+'_'+str(df1['date'].iloc[0])+'_'+str(df1['task'].iloc[0])+'_figs.pdf')))

        for ind, row in df1.iterrows():
            for x in ['b','w']:
                xvals = np.stack([row['obstacle'+x+'TL_x'], row['obstacle'+x+'TR_x'], row['obstacle'+x+'BL_x'], row['obstacle'+x+'BL_x']]).astype(float)
                df1.at[ind, x+'obstacle_x'] = np.nanmean(xvals)
                df1.at[ind, x+'obstacle_x_std'] = np.mean(np.nanstd(xvals, axis=1))
                yvals = np.stack([row['obstaclewTL_y'], row['obstaclewTR_y'], row['obstaclewBL_y'], row['obstaclewBL_y']]).astype(float)
                df1.at[ind, x+'obstacle_y'] = np.nanmean(yvals)
                df1.at[ind, x+'obstacle_y_std'] = np.mean(np.nanstd(yvals, axis=1))

        kmeans_input = np.stack([df1['wobstacle_x'].map(np.mean), df1['wobstacle_y'].map(np.mean)])

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
            plt.ylim([0,356]); plt.xlim([0,592])
        pdf.savefig(); plt.close()

        df1['obstacle_cluster'] = labels

        df1 = df1[df1['len']<500]

        odd = df1[df1.index%2==0]
        even = df1[df1.index%2==1]
        direction_count = 0
        for direction_df in [odd, even]:
            if direction_count == 0:
                leftcolor='b'; rightcolor='g'
            else:
                leftcolor='g'; rightcolor='b'
            plt.subplots(3,3,figsize=(9,6))
            for c in range(9):
                this_cluster = direction_df[direction_df['obstacle_cluster']==c].copy().reset_index()
                plt.subplot(3,3,c+1)
                plt.gca().set_aspect('equal', adjustable='box')
                colors = plt.cm.magma(np.linspace(0,1,len(this_cluster)))
                for ind, row in this_cluster.iterrows():
                    plt.plot([np.median(row['obstaclewTL_x'],0),
                            np.median(row['obstaclewTR_x'],0),
                            np.median(row['obstaclewBR_x'],0),
                            np.median(row['obstaclewBL_x'],0),
                            np.median(row['obstaclewTL_x'],0)],
                            [np.median(row['obstaclewTL_y'],0),
                            np.median(row['obstaclewTR_y'],0),
                            np.median(row['obstaclewBR_y'],0),
                            np.median(row['obstaclewBL_y'],0),
                            np.median(row['obstaclewTL_y'],0)],'k-')
                    plt.plot(row['nose_x'], row['nose_y'], '-', color=colors[ind])
                    plt.plot(row['leftportT_x'], row['leftportT_y'],'.',color=leftcolor)
                    plt.plot(row['rightportT_x'], row['rightportT_y'],'.',color=rightcolor)
                plt.ylim([356,0]); plt.xlim([0,592])
            direction_count += 1
            plt.tight_layout()
            plt.show()
            pdf.savefig(); plt.close()
    
        pdf.close()

if __name__ == '__main__':
    args = get_args()
    main(args)