"""
FreelyMovingEphys/src/worldcam.py
"""
import os, sys
import xarray as xr

from src.base import Camera

class Worldcam(Camera):
    def __init__(self, config, recording_name, recording_path, camname):
        Camera.__init__(self, config, recording_name, recording_path, camname)
        
    def save_params(self):
        self.xrpts.name = self.camname+'_times'
        self.xrframes.name = self.camname+'_video'
        merged_data = [self.xrpts, self.xrframes]

        self.data = self.safe_merge(merged_data)
        self.data.to_netcdf(os.path.join(self.recording_path,str(self.recording_name+'_world.nc')),engine='netcdf4',encoding={self.camname+'_video':{"zlib": True, "complevel": 4}})

    def process(self):
        if self.config['main']['deinterlace'] and not self.config['internals']['flip_headcams']['run']:
            self.deinterlace()
        elif not self.config['main']['deinterlace'] and self.config['internals']['flip_headcams']['run']:
            self.flip_headcams()
        elif self.config['main']['deinterlace'] and self.config['internals']['flip_headcams']['run']:
            print('Config options deinterlace and flip_headcams are both True, which conflict with each other.')
            sys.exit()

        if self.config['main']['parameters']:
            self.gather_camera_files()
            self.pack_position_data()
            self.pack_video_frames()
            self.save_params()