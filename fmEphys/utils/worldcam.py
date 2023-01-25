"""
FreelyMovingEphys/src/worldcam.py
"""
import os
import sys
import xarray as xr

import fmEphys

class Worldcam(fmEphys.Camera):
    def __init__(self, cfg, recording_name, recording_path, camname):
        fmEphys.Camera.__init__(self, cfg, recording_name, recording_path, camname)
        
    def save_params(self):
        self.xrpts.name = self.camname+'_times'
        self.xrframes.name = self.camname+'_video'
        merged_data = [self.xrpts, self.xrframes]

        self.safe_merge(merged_data)
        savepath = os.path.join(self.recording_path,str(self.recording_name+'_world.nc'))
        self.data.to_netcdf(savepath, engine='netcdf4',
                    encoding={self.camname+'_video':{"zlib": True, "complevel": 4}})

        print('Saved {}'.format(savepath))

    def process(self):
        if self.cfg['run']['deinterlace']:
            self.deinterlace()
        elif not self.cfg['run']['deinterlace'] and (self.cfg['headcams_hflip'] or self.cfg['headcams_vflip']):
            self.flip_headcams()

        if self.cfg['run']['undistort']:
            self.undistort()

        if self.cfg['run']['parameters']:

            self.gather_camera_files()

            if hasattr(self, 'calibvid_path'):
                self.video_path = self.calibvid_path
            
            self.pack_position_data()
            self.pack_video_frames()
            self.save_params()