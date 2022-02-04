"""
FreelyMovingEphys/src/sidecam.py
"""
import os, sys
import xarray as xr

from src.base import Camera

class Sidecam(Camera):
    def __init__(self, config, recording_name, recording_path, camname):
        Camera.__init__(self, config, recording_name, recording_path, camname)
        
    def save_params(self):
        self.xrpts.name = self.camname+'_times'
        self.xrframes.name = self.camname+'_video'
        merged_data = [self.xrpts, self.xrframes]

        self.safe_merge(merged_data)
        self.data.to_netcdf(os.path.join(self.recording_path,str(self.recording_name+'_side.nc')),engine='netcdf4',encoding={self.camname+'_video':{"zlib": True, "complevel": 4}})

    def process(self):
        if self.config['main']['undistort']:
            self.undistort(mtxkey='sidecam_mtx', readcamkey='SIDE', savecamkey='_SIDEcalib.avi', checkervid='sidecam_checkerboard')
            self.video_path = self.calibvid_path
        if self.config['main']['pose_estimation'] or self.config['main']['parameters']:
            self.gather_camera_files()
        if self.config['main']['pose_estimation']:
            self.pose_estimation()
        if self.config['main']['parameters']:
            self.pack_position_data()
            self.pack_video_frames()
            self.save_params()