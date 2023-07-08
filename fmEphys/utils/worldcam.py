"""
fmEphys/utils/worldcam.py

Process the world camera data.

Classes
-------
Worldcam
    Class for processing world camera data.


Written by DMM, 2021
"""


import os
import fmEphys as fme


class Worldcam(fme.Camera):
    """ Class for processing world camera data.
    """


    def __init__(self, cfg, recording_name, recording_path, camname):
        """ Initialize the Worldcam object.
        """
        fme.Camera.__init__(self, cfg, recording_name,
                            recording_path, camname)
        

    def save_params(self):
        """ Save the preprocessed data to an .nc file.
        """

        self.xrpts.name = self.camname+'_times'
        self.xrframes.name = self.camname+'_video'
        merged_data = [self.xrpts, self.xrframes]

        self.safe_merge(merged_data)
        savepath = os.path.join(self.recording_path,
                                str(self.recording_name+'_world.nc'))
        
        self.data.to_netcdf(savepath, engine='netcdf4',
                    encoding={self.camname+'_video':{"zlib": True,
                                                     "complevel": 4}})

        print('Saved {}'.format(savepath))


    def process(self):
        """ Run the preprocessing pipeline.
        """

        # Deinterlace video
        if self.cfg['run']['deinterlace']:
            self.deinterlace()

        # Flip the video without deinterlacing
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

