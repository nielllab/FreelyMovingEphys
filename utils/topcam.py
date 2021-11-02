"""
track_topdown.py
"""
import numpy as np
import xarray as xr
import os
import cv2
from tqdm import tqdm

from utils.base import Camera

class Topcam(Camera):
    def __init__(self, config, recording_name, recording_path, camname):
        super.__init__(self, config, recording_name, recording_path)
        self.camname = camname

    def filter_likelihood(self):
        thresh = self.config['internals']['likelihood_threshold']
        x_cols = [i for i in self.pt_names if '_x' in i]
        y_cols = [i for i in self.pt_names if '_y' in i]
        l_cols = [i for i in self.pt_names if '_likelihood' in i]
        for i in range(len(x_cols)):
            x = self.xrpts.sel(point_loc=x_cols[i])
            y = self.xrpts.sel(point_loc=y_cols[i])
            l = self.xrpts.sel(point_loc=l_cols[i])
            x[l<thresh] = np.nan; y[l<thresh] = np.nan
            self.xrpts.loc[dict(point_loc=x_cols[i], frame=i)] = x
            self.xrpts.loc[dict(point_loc=y_cols[i], frame=i)] = y

    def get_head_angle(self, pt_input, name1='Nose', name2='BackNeck'):
        """ Get body angle of mouse in topdown view.
        also: MidSpine1, MidSpine2
        """
        angs = []
        for step in tqdm(range(0,np.size(pt_input, 1))):
            step_pts = pt_input.isel(frame=step)
            x1 = step_pts.sel(point_loc=name1+'_x')
            x2 = step_pts.sel(point_loc=name2+'_x')
            y1 = step_pts.sel(point_loc=name1+'_y')
            y2 = step_pts.sel(point_loc=name2+'_y')
            x_dist = x1 - x2
            y_dist = y1 - y2
            th = np.arctan2(y_dist, x_dist)
            angs.append(float(th))
        all_angs = xr.DataArray(angs, dims=['frame'])

        return all_angs

    def diagnostic_video(self):
        vidread = cv2.VideoCapture(self.video_path)
        width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

        savepath = os.path.join(self.recording_path, (self.recordong_name+'_'+self.recording_name+'_plot.avi'))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))
        plot_color0 = (225, 255, 0)

        if self.config['internals']['video_frames_to_save'] > int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)):
            num_save_frames = int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            num_save_frames = self.config['internals']['video_frames_to_save']

        for frame_num in tqdm(range(0,num_save_frames)):
            ret, frame = vidread.read()
            if not ret:
                break
            try:
                for k in range(0, len(self.xrpts['point_loc']), 3):
                    try:
                        td_pts_x = self.xrpts.isel(frame=frame_num, point_loc=k).values
                        td_pts_y = self.xrpts.isel(frame=frame_num, point_loc=k+1).values
                        center_xy = (int(td_pts_x), int(td_pts_y))
                        frame = cv2.circle(frame, center_xy, 6, plot_color0, -1)
                    except (ValueError, OverflowError) as e:
                        pass
            except KeyError:
                pass
            out_vid.write(frame)
        out_vid.release()

    def process(self):
        if self.config['main']['pose_estimation']:
            self.pose_estimation()
        if self.config['main']['parameters']:
            self.gather_files()
            self.pack_position_data()
            self.pt_names = list(self.data['point_loc'].values)
            self.filter_likelihood()
            if self.config['internals']['diagnostic_preprocessing_videos']:
                self.diagnostic_video()
            self.pack_video_frames

    def save(self):
        self.safe_merge([self.xrpts, self.xrframes])
        self.data.to_netcdf(os.path.join(self.recording_path, str(self.recording_name+'_'+self.camname+'.nc')), engine='netcdf4', encoding={self.camname+'_video':{"zlib": True, "complevel": 4}})