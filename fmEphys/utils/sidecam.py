"""
fmEPhys/utils/sidecam.py


Written by DMM, 2020
"""


import os
import sys
import xarray as xr

import fmEphys as fme


class Sidecam(fme.Camera):


    def __init__(self, cfg, recording_name, recording_path, camname):
        fme.Camera.__init__(self, cfg, recording_name, recording_path, camname)
        

    def save_params(self):
        self.xrpts.name = self.camname+'_times'
        self.xrframes.name = self.camname+'_video'
        merged_data = [self.xrpts, self.xrframes]

        self.safe_merge(merged_data)
        self.data.to_netcdf(os.path.join(self.recording_path,
                            str(self.recording_name+'_side.nc')),
                            engine='netcdf4',
                            encoding={self.camname+'_video': {"zlib": True, "complevel": 4}})


    def process(self):

        if self.cfg['run']['undistort']:

            self.undistort(mtxkey='sidecam_mtx',
                           readcamkey='SIDE',
                           savecamkey='_SIDEcalib.avi',
                           checkervid='sidecam_checkerboard')
            
            self.video_path = self.calibvid_path

        if self.cfg['run']['pose_estimation'] or self.cfg['run']['parameters']:
            
            self.gather_camera_files()

        if self.cfg['run']['pose_estimation']:
            
            self.pose_estimation()

        if self.cfg['run']['parameters']:

            self.pack_position_data()

            self.pack_video_frames()

            self.save_params()

