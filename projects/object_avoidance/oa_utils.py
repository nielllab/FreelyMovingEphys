"""
oa_utils.py
"""
import sys
sys.path.insert(0, '/home/niell_lab/Documents/github/FreelyMovingEphys/')
import pandas as pd
import xarray as xr
from utils.format_data import *
from utils.paths import find
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os, cv2
from pathlib import Path
from scipy.interpolate import interp1d

def make_oa_df(directory_path, dates):
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
    data_path = Path(directory_path).expanduser()
    all_paths = []
    # populate dict with metadata and timestamps
    for date in dates:
        for ani in os.listdir(data_path / date): 
            for task in os.listdir(data_path / date/ ani):
                data_paths = list((data_path / date/ ani/ task).rglob('*.csv'))
                if data_paths != []:
                    data_dict['date'].append(data_paths[1].name.split('_')[0])
                    data_dict['animal'].append(data_paths[1].name.split('_')[1])
                    data_dict['task'].append(data_paths[1].name.split('_')[4])
                for ind, csv in enumerate(data_paths):
                    data_dict[data_paths[ind].name.split('_')[5] +'_ts'].append(open_time(csv))
                    data_dict[data_paths[ind].name.split('_')[5] +'_t0'].append(open_time(csv)[0])
    df = pd.DataFrame.from_dict(data_dict)
    return df

def filter_likelihood(da, thresh=0.99):
    x_cols = [i for i in da.columns.values if '_x' in i and 'arena' not in i]
    y_cols = [i for i in da.columns.values if '_y' in i and 'arena' not in i]
    l_cols = [i for i in da.columns.values if '_likelihood' in i and 'arena' not in i]
    for i in range(len(x_cols)):
        x = da.loc[:,x_cols[i]]; y = da.loc[:,y_cols[i]]; l = da.loc[:,l_cols[i]]
        x[l<thresh] = np.nan; y[l<thresh] = np.nan
        da.loc[:,x_cols[i]] = x; da.loc[:,y_cols[i]] = y
    return da

def make_task_df(df, index, dlc_h5):
    row = df.iloc[index]
    num_odd_trails = np.min([len(row['poke1_ts']), len(row['poke2_ts'])])
    df1 = pd.DataFrame([])
    dlc_positions, dlc_labels = open_h5(dlc_h5)
    dlc_positions = filter_likelihood(dlc_positions)
    count = -1
    for c in range(num_odd_trails):
        # odd
        count += 1
        df1.at[count, 'first_poke'] = row['poke1_ts'][c]
        df1.at[count, 'second_poke'] = row['poke2_ts'][c]
        time = row['top1_ts']; time = time[time > df1.loc[count,'first_poke']]; time = time[time < df1.loc[count,'second_poke']]
        df1.at[count, 'trail_timestamps'] = time.astype(object)
        start_stop_inds = (int(np.where([row['top1_ts']==time[0]])[1]), int(np.where([row['top1_ts']==time[-1]])[1]))
        for pos in dlc_positions:
            df1.at[count, pos] = np.array(dlc_positions.loc[start_stop_inds[0]:start_stop_inds[1], pos]).astype(object)
        df1.at[count, 'len'] = start_stop_inds[1] - start_stop_inds[0]
        # even
        count += 1
        if c+1 < len(row['poke1_ts']):
            df1.at[count, 'first_poke'] = row['poke2_ts'][c]
            df1.at[count, 'second_poke'] = row['poke1_ts'][c+1]
            time = row['top1_ts']; time = time[time > df1.loc[count,'first_poke']]; time = time[time < df1.loc[count,'second_poke']]
            df1.at[count, 'trail_timestamps'] = time.astype(object)
            start_stop_inds = (int(np.where([row['top1_ts']==time[0]])[1]), int(np.where([row['top1_ts']==time[-1]])[1]))
            for pos in dlc_positions:
                df1.at[count, pos] = np.array(dlc_positions.loc[start_stop_inds[0]:start_stop_inds[1], pos]).astype(object)
            df1.at[count, 'len'] = start_stop_inds[1] - start_stop_inds[0]
    df1['animal'] = row['animal']; df1['date'] = row['date']; df1['task'] = row['task']
    return df1

def convert_pxls_to_dist(da, pxls2cm):
    x_cols = [i for i in da.columns.values if '_x' in i]
    y_cols = [i for i in da.columns.values if '_y' in i]
    for i in range(len(x_cols)):
        da[x_cols[i]+'_cm'] = da.loc[:,x_cols[i]]/pxls2cm
        da[y_cols[i]+'_cm'] = da.loc[:,y_cols[i]]/pxls2cm
    return da

def get_head_angle(df):
    for ind, row in df.iterrows():
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
        df.at[ind, 'head_angle'] = np.array(angs).astype(object)
    return df

def get_median_trace(df):
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

def format_frames_oa(vid_path):
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

def get_row_for_timestamp(df, seek_timestamp):
    for ind, row in df.iterrows():
        if seek_timestamp in row['trail_timestamps']:
            return row

def plot_frame(vid_arr, timestamps, df, seek_frame, return_as_array=False):
    seek_timestamp = timestamps[seek_frame]
    row = get_row_for_timestamp(df, seek_timestamp)
    if row is None:
        if return_as_array:
            return np.zeros(np.shape(vid_arr[0]))
        elif not return_as_array:
            plt.figure()
            plt.imshow(np.zeros(np.shape(vid_arr[0])), cmap='gray')
            plt.show()
    row_time_index = np.where(row['trail_timestamps']==seek_timestamp)
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

def distance_from_nose(row, target):
    x_dist = np.abs(row[target+'_x_cm'] - row['nose_x_cm'])
    y_dist = np.abs(row[target+'_y_cm'] - row['nose_y_cm'])
    length = len(x_dist)
    dist = np.zeros([length])
    for i in range(length):
        dist[i] = np.sqrt(x_dist[i]**2 + y_dist[i]**2)
    return dist

def angle_from_nose(row, target):
    x_dist = np.abs(row[target+'_x_cm'] - row['nose_x_cm'])
    y_dist = np.abs(row[target+'_y_cm'] - row['nose_y_cm'])
    length = len(x_dist)
    ang = np.zeros([length])
    for i in range(length):
        ang[i] = (y_dist[i]/x_dist[i])
    return ang%np.pi

def approaching_target(dist):
    if np.nanmean(dist[:10]) > np.nanmean(dist[:-10]):
        return True
    else:
        return False

def plot_all_trials(vid_arr, timestamps, df, vid_savepath):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(vid_savepath, fourcc, 60.0, (640, 480))
    start = 3600*5
    for seek_frame in tqdm(range(start,start+3600)):
        frame = plot_frame(vid_arr, timestamps, df, seek_frame, return_as_array=True)
        out_vid.write(frame.astype('uint8'))
    out_vid.release()