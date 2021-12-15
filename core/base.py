"""
FreelyMovingEphys/core/base.py
"""
import os, subprocess, math
import cv2
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from tqdm import tqdm
os.environ["DLClight"] = "True"
import deeplabcut
from scipy.io import loadmat

from core.utils.path import find, list_subdirs

class BaseInput:
    """ Base preprocessing input data.
    """
    def __init__(self, config, recording_name, recording_path):
        self.config = config
        self.recording_name = recording_name
        self.recording_path = recording_path

    def read_timestamp_series(self, s):
        """ Read timestamps as a pd.Series and format time.

        Parameters
        --------
        s : pd.Series
            Timestamps as a Series.
            Expected to be formated as hours : minutes : seconds . microsecond

        Returns
        --------
        output_time : np.array
            Returned as the number of seconds that have passed since the
            previous midnight, with microescond precision, e.g. 700.000000
        """
        output_time = []
        fmt = '%H:%M:%S.%f'
        if s.dtype != np.float64:
            for current_time in s:
                str_time = str(current_time).strip()
                try:
                    t = datetime.strptime(str_time, fmt)
                except ValueError as v:
                    ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                    if ulr:
                        str_time = str_time[:-ulr]
                try:
                    output_time.append((datetime.strptime(str_time, '%H:%M:%S.%f') - datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')).total_seconds())
                except ValueError:
                    output_time.append(np.nan)
            output_time = np.array(output_time)
        else:
            output_time = s.values
        return output_time

    def interp_timestamps(self, camT, use_medstep=False):
        """ Interpolate timestamps for double the number of frames. Compensates for video deinterlacing.
        
        Parameters
        --------
        camT : np.array
            Camera timestamps aquired at 30Hz
        use_medstep : bool
            When True, the median diff(camT) will be used as the timestep
            in interpolation. If False, the timestep between each frame will
            be used instead.

        Returns
        --------
        camT_out : np.array
            Timestamps of camera interpolated so that there are twice the number
            of timestamps in the array. Each timestamp in camT will be replaced by
            two, set equal distances from the original.
        """
        camT_out = np.zeros(np.size(camT, 0)*2)
        medstep = np.nanmedian(np.diff(camT, axis=0))
        if use_medstep:
            # shift each deinterlaced frame by 0.5 frame periods forward/backwards
            camT_out[::2] = camT - 0.25 * medstep
            camT_out[1::2] = camT + 0.25 * medstep
        elif not use_medstep:
            steps = np.diff(camT, axis=0, append=camT[-1]+medstep)
            camT_out[::2] = camT - 0.25 * steps
            camT_out[1::2] = camT + 0.25 * steps
        return camT_out

    def read_timestamp_file(self, position_data_length=None, force_timestamp_shift=False):
        """ Read timestamps from a .csv file.

        Parameters
        --------
        position_data_length : None or int
            Number of timesteps in data from deeplabcut. This is used to
            determine whether or not the number of timestamps is too short
            for the number of video frames.
            Eyecam and Worldcam will have half the number of timestamps as
            the number of frames, since they are aquired as an interlaced
            video and deinterlaced in analysis. To fix this, timestamps need
            to be interpolated.
        """
        # read data and set up format
        s = pd.read_csv(self.timestamp_path, encoding='utf-8', engine='c', header=None).squeeze()
        if s[0] == 0:
            s = s[1:]
        read_time = self.read_timestamp_series(s)
        # auto check if vids were deinterlaced
        if position_data_length is not None:
            # test length of the time just read in as it compares to the length of the data, correct for deinterlacing if needed
            timestep = np.nanmedian(np.diff(read_time, axis=0))
            if position_data_length > len(read_time):
                output_time = np.zeros(np.size(read_time, 0)*2)
                # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
                output_time[::2] = read_time - 0.25 * timestep
                output_time[1::2] = read_time + 0.25 * timestep
            elif position_data_length == len(read_time):
                output_time = read_time
            elif position_data_length < len(read_time):
                output_time = read_time
        elif position_data_length is None:
            output_time = read_time
        # force the times to be shifted if the user is sure it should be done
        if force_timestamp_shift is True:
            # test length of the time just read in as it compares to the length of the data, correct for deinterlacing
            timestep = np.nanmedian(np.diff(read_time, axis=0))
            output_time = np.zeros(np.size(read_time, 0)*2)
            # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
            output_time[::2] = read_time - 0.25 * timestep
            output_time[1::2] = read_time + 0.25 * timestep
        return output_time

class Camera(BaseInput):
    def __init__(self, config, recording_name, recording_path, camname):
        BaseInput.__init__(self, config, recording_name, recording_path)
        self.camname = camname

    def deinterlace(self, videos=None, timestamps=None):
        """ Deinterlace videos and shift timestamps to match new video frames.
        If videos and timestamps are provided (as lists), only the provided filepaths will be processed.
        If lists are not provided, subdirectories will be searched within animal_directory in the options dictionary, config.
        Both videos and timestamps must be provided, for either to be used.

        Parameters:
        videos (list): list of eyecam and/or worldcam videos at 30fps
        timestamps (list): list of timestamp csv files for each video in videos
        """
        if 'EYE' in self.camname:
            if self.config['internals']['rotate_eyecam']:
                do_rotation = True
            else:
                do_rotation = False
        elif 'WORLD' in self.camname:
            if self.config['internals']['rotate_worldcam']:
                do_rotation = True
            else:
                do_rotation = False
        # search subdirectories if both lists are not given
        if videos is None or timestamps is None:
            videos = find('*'+self.camname+'*.avi', self.recording_path)
            timestamps = find('*'+self.camname+'*.csv', self.recording_path)
        # iterate through each video
        for vid in videos:
            current_path = os.path.split(vid)[0]
            # make a save path that keeps the subdirectories
            # get out an key from the name of the video that will be shared with all other data of this trial
            vid_name = os.path.split(vid)[1]
            key_pieces = vid_name.split('.')[:-1]
            key = '.'.join(key_pieces)
            # then, find those other pieces of the trial using the key
            try:
                this_csv = next(i for i in timestamps if key in i)
                csv_present = True
            except:
                csv_present = False
            # open the video
            cap = cv2.VideoCapture(vid)
            # get some info about the video
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # number of total frames
            fps = cap.get(cv2.CAP_PROP_FPS) # frame rate
            # make sure the save directory exists
            if not os.path.exists(current_path):
                os.makedirs(current_path)
            # files that will need to be deinterlaced will be read in with a frame rate of 30 frames/sec
            if fps == 30:
                # create save path
                avi_out_path = os.path.join(current_path, (key + 'deinter.avi'))
                # flip the eye video horizonally and vertically and deinterlace, if this is specified in the config
                if do_rotation:
                    subprocess.call(['ffmpeg', '-i', vid, '-vf', 'yadif=1:-1:0, vflip, hflip, scale=640:480', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
                # or, deinterlace without flipping
                elif not do_rotation:
                    subprocess.call(['ffmpeg', '-i', vid, '-vf', 'yadif=1:-1:0, scale=640:480', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
                # correct the frame count of the video
                # now that it's deinterlaced, the video has 2x the number of frames as before
                # this will be used to correct the timestamps associated with this video
                frame_count_deinter = frame_count * 2
                if csv_present is True:
                    # get the save path for new timestamps
                    csv_out_path = os.path.join(current_path, (key + '_BonsaiTSformatted.csv'))
                    # read in the exiting timestamps, interpolate to match the new number of steps, and format as dataframe
                    csv_out = pd.DataFrame(self.read_timestamp_file(this_csv, int(frame_count_deinter)))
                    # save new timestamps
                    csv_out.to_csv(csv_out_path, index=False)

    def flip_headcams(self):
        vid_list = find('*'+self.camname+'.avi', self.recording_path)
        for this_avi in vid_list:
            vid_name = os.path.split(this_avi)[1]
            key_pieces = vid_name.split('.')[:-1]
            key = '.'.join(key_pieces)
            print(key)
            avi_out_path = os.path.join(os.path.split(this_avi)[0], (key + 'deinter.avi'))
            if self.config['internals']['flip_headcams']['hflip'] and not self.config['internals']['flip_headcams']['vflip']:
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'hflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
            elif not self.config['internals']['flip_headcams']['hflip'] and self.config['internals']['flip_headcams']['vflip']:
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'vflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
            elif self.config['internals']['flip_headcams']['hflip'] and self.config['internals']['flip_headcams']['vflip']:
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'vflip, hflip', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])

    def define_distortion(self):
        """ Define distortion from checkerbaord videos.
        Config files are only setup to track worldcam checkboard videos.
        Could be used for topcam also.
        """
        # arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        #prepare object points
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        # read in file path of video
        calib_vid = cv2.VideoCapture(self.config['paths']['worldcam_checkerboard'])
        # iterate through frames
        print('getting distortion out of each frame')
        for step in tqdm(range(0,int(calib_vid.get(cv2.CAP_PROP_FRAME_COUNT)))):
            # open frame
            ret, img = calib_vid.read()
            # make sure the frame is read in correctly
            if not ret:
                break
            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
            # if found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
        # calibrate the camera (this is a little slow)
        print('calculating calibration correction paramters')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # format as xarray and save the file
        savepath = self.config['paths']['worldcam_mtx']
        np.savez(savepath, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    def undistort(self):
        # load the parameters
        checker_in = np.load(self.config['paths']['worldcam_mtx'])
        # unpack camera properties
        mtx = checker_in['mtx']; dist = checker_in['dist']; rvecs = checker_in['rvecs']; tvecs = checker_in['tvecs']
        # iterate through eye videos and save out a copy which has had distortions removed
        world_list = find('*WORLDdeinter*.avi', self.config['animal_directory'])
        for world_vid in [x for x in world_list if 'plot' not in x]:
            savepath = '_'.join(world_vid.split('_')[:-1])+'_WORLDcalib.avi'
            cap = cv2.VideoCapture(world_vid)
            # setup the file writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            # iterate through all frames
            for step in tqdm(range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
                # open frame and check that it opens correctly
                ret, frame = cap.read()
                if not ret:
                    break
                # run opencv undistortion function
                undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)
                # write the frame to the video
                out_vid.write(undist_frame)
            out_vid.release()

    def auto_contrast(self):
        if self.config['img_correction']['apply_gamma_to_eyecam']:
            input_list = find('*EYE.avi', self.config['animal_directory'])
            # iterate through input videos
            for video in input_list:
                print('correcting gamma for '+video)
                # build the save path
                head, tail = os.path.split(video)
                new_name = os.path.splitext(tail)[0] + 'deinter.avi'
                savepath = os.path.join(head, new_name)
                # write new video with gamma correction
                vid_read = cv2.VideoCapture(video)
                width = int(vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))
                num_frames = int(vid_read.get(cv2.CAP_PROP_FRAME_COUNT))
                print('num_frames', num_frames)
                # iterate through frames, find ideal gamma, apply, write frame
                for step in tqdm(range(0,num_frames)):
                    ret, frame = vid_read.read()
                    if not ret:
                        break
                    # convert img to gray
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # compute gamma = log(mid*255)/log(mean)
                    mid = 0.5
                    mean = np.mean(gray)
                    gamma = math.log(mid*255)/math.log(mean)
                    # do gamma correction
                    img_gamma1 = np.power(frame, gamma).clip(0,255).astype(np.uint8)
                    # write frame
                    out_vid.write(img_gamma1)
                out_vid.release()

    def batch_dlc_analysis(self, videos, project_config):
        if isinstance(videos, str):
            videos = [videos]
        for vid in videos:
            if self.config['internals']['crop_for_dlc'] is True:
                deeplabcut.cropimagesandlabels(project_config, size=(400, 400), userfeedback=False)
            deeplabcut.analyze_videos(project_config, [vid])
            if self.config['internals']['filter_dlc_predictions'] is True:
                deeplabcut.filterpredictions(project_config, vid)

    def pose_estimation(self):
        # get each camera type's entry
        cam_project = self.config['paths']['dlc_projects'][self.camname]
        if cam_project != '' and cam_project != 'None' and cam_project != None:
            # if it's one of the cameras that needs to needs to be deinterlaced first, make sure and read in the deinterlaced 
            if self.camname=='REYE' or self.camname=='LEYE':
                # find all the videos in the data directory that are from the current camera and are deinterlaced
                if self.config['internals']['follow_strict_naming'] is True:
                    vids_this_cam = find('*'+self.camname+'*deinter.avi', self.config['animal_directory'])
                elif self.config['internals']['follow_strict_naming'] is False:
                    vids_this_cam = find('*'+self.camname+'*.avi', self.config['animal_directory'])
                # remove unflipped videos generated during jumping analysis
                bad_vids = find('*'+self.camname+'*unflipped*.avi', self.config['animal_directory'])
                for x in bad_vids:
                    if x in vids_this_cam:
                        vids_this_cam.remove(x)
                ir_vids = find('*IR*.avi', self.config['animal_directory'])
                for x in ir_vids:
                    if x in vids_this_cam:
                        vids_this_cam.remove(x)
                print('found ' + str(len(vids_this_cam)) + ' deinterlaced videos from cam_key ' + self.camname)
                # warning for user if no videos found
                if len(vids_this_cam) == 0:
                    print('no ' + self.camname + ' videos found -- maybe the videos are not deinterlaced yet?')
            else:
                # find all the videos for camera types that don't neeed to be deinterlaced
                if self.config['internals']['follow_strict_naming'] is True:
                    vids_this_cam = find('*'+self.camname+'*.avi', self.config['animal_directory'])
                elif self.config['internals']['follow_strict_naming'] is False:
                    vids_this_cam = find('*'+self.camname+'*.avi', self.config['animal_directory'])
                print('found ' + str(len(vids_this_cam)) + ' videos from cam_key ' + self.camname)
            # analyze the videos with DeepLabCut
            # this gives the function a list of files that it will iterate over with the same DLC config file
            vids2run = [vid for vid in vids_this_cam if 'plot' not in vid]
            self.batch_dlc_analysis(vids2run, cam_project)

    def open_dlc_h5(self, h5key=None):
        if h5key is None:
            # read the .hf file when there is no key
            pts = pd.read_hdf(self.dlc_path)
        else:
            # read in .h5 file when there is a key set in corral_files.py
            pts = pd.read_hdf(self.dlc_path, key=h5key)
        # organize columns
        pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
        pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
        pt_loc_names = pts.columns.values
        return pts, pt_loc_names

    def open_dlc_h5_multianimal(self):
        pts = pd.read_hdf(self.dlc_path)
        # flatten columns from MultiIndex 
        pts.columns = ['_'.join(col[:][1:]).strip() for col in pts.columns.values]
        return pts

    def safe_merge(self, obj_list, dim_name='frame'):
        """ Merge a list of xr DataArrays even when their lengths do not match.
        """
        max_lens = []
        # iterate through objects
        for obj in obj_list:
            # get the sizes of the dim, dim_name
            max_lens.append(dict(obj.frame.sizes)[dim_name])
        # get the smallest of the object's length's
        set_len = np.min(max_lens)
        # shorten everything to the shortest length found
        out_objs = []
        for obj in obj_list:
            # get the length of the current object
            obj_len = dict(obj.frame.sizes)[dim_name]
            # if the size of dim is longer
            if obj_len > set_len:
                # how much does it need to be shortened by?
                diff = obj_len - set_len
                # what indeces should be kept?
                good_inds = range(0,obj_len-diff)
                # index to remove what would be jagged ends
                obj = obj.sel(frame=good_inds)
                # add to the list of objects to merge
                out_objs.append(obj)
            # if it is the smallest length or all objects have the same length
            else:
                # just append it to the list of objects to merge
                out_objs.append(obj)
        # do the merge with the lengths all matching along provided dimension
        self.data = xr.merge(out_objs)

    def gather_files(self):
        # get dlc h5 path
        h5_paths = [x for x in find(('*.h5'), self.recording_path) if x != []]
        h5_paths = [x for x in h5_paths if 'DLC' in x]
        self.dlc_path = next(path for path in h5_paths if self.camname in path)
        # get avi video and timestamps
        if 'eye' in self.camname.lower() or 'world' in self.camname.lower():
            if self.config['internals']['follow_strict_naming']:
                # video
                avi_paths = [x for x in find(('*.avi'), self.recording_path) if x != []]
                self.video_path = next(path for path in avi_paths if self.camname in path and 'deinter' in path and 'plot' not in path)
                # timestamps
                csv_paths = [x for x in find(('*BonsaiTS*.csv'), self.recording_path) if x != []]
                self.timestamp_path = next(i for i in csv_paths if self.camname in i and 'formatted' in i)
            elif not self.config['internals']['follow_strict_naming']:
                # video
                avi_paths = [x for x in find(('*.avi'), self.recording_path) if x != []]
                self.video_path = next(path for path in avi_paths if self.camname in path)
                # timestamps
                csv_paths = [x for x in find(('*BonsaiTS*.csv'), self.recording_path) if x != []]
                self.timestamp_path = next(i for i in csv_paths if self.camname)
        # all other cameras (i.e. topcam and sidecam)
        else:
            avi_paths = [x for x in find(('*.avi'), self.recording_path) if x != []]
            self.video_path = next(path for path in avi_paths if self.camname in path and 'plot' not in path)
            csv_paths = [x for x in find(('*BonsaiTS*.csv'), self.recording_path) if x != []]
            self.timestamp_path = next(i for i in csv_paths if self.camname in i)

    def pack_video_frames(self, usexr=True, dwnsmpl=None):
        if dwnsmpl is None:
            dwnsmpl = self.config['internals']['video_dwnsmpl']
        # open the .avi file
        vidread = cv2.VideoCapture(self.video_path)
        # empty array that is the target shape
        # should be number of frames x downsampled height x downsampled width
        all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                            int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*dwnsmpl),
                            int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*dwnsmpl)], dtype=np.uint8)
        # iterate through each frame
        for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
            # read the frame in and make sure it is read in correctly
            ret, frame = vidread.read()
            if not ret:
                break
            # convert to grayyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # downsample the frame by an amount specified in the config file
            sframe = cv2.resize(frame, (0,0), fx=dwnsmpl, fy=dwnsmpl, interpolation=cv2.INTER_NEAREST)
            # add the downsampled frame to all_frames as int8
            all_frames[frame_num,:,:] = sframe.astype(np.int8)
        if not usexr:
            return all_frames
        # store the combined video frames in an xarray
        formatted_frames = xr.DataArray(all_frames.astype(np.int8), dims=['frame', 'height', 'width'])
        # label frame numbers in the xarray
        formatted_frames.assign_coords({'frame':range(0,len(formatted_frames))})
        # delete all frames, since it's somewhat large in memory
        del all_frames
        self.xrframes = formatted_frames

    def pack_position_data(self):
        """ Pack the camera's dlc points and timestamps together in one DataArray.
        """
        # check that pt_path exists
        if self.dlc_path is not None and self.dlc_path != [] and self.timestamp_path is not None:
            # open multianimal project with a different function than single animal h5 files
            if 'TOP' in self.camname and self.config['internals']['multianimal_top_project'] is True:
                # add a step to convert pickle files here?
                pts = self.open_dlc_h5_multianimal()
            # otherwise, use regular h5 file read-in
            else:
                pts, self.pt_names = self.open_dlc_h5()
            # read time file, pass length of points so that it will know if that length matches the length of the timestamps
            # if they don't match because time was not interpolated to match deinterlacing, the timestamps will be interpolated
            time = self.read_timestamp_file(len(pts))
            # label dimensions of the points dataarray
            xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
            # label the camera view
            xrpts.name = self.camname
            # assign timestamps as a coordinate to the 
            try:
                xrpts = xrpts.assign_coords(timestamps=('frame', time[1:])) # indexing [1:] into time because first row is the empty header, 0
            # correcting for issue caused by small differences in number of frames
            except ValueError:
                diff = len(time[1:]) - len(xrpts['frame'])
                if diff > 0: # time is longer
                    diff = abs(diff)
                    new_time = time.copy()
                    new_time = new_time[0:-diff]
                    xrpts = xrpts.assign_coords(timestamps=('frame', new_time[1:]))
                elif diff < 0: # frame is longer
                    diff = abs(diff)
                    timestep = time[1] - time[0]
                    new_time = time.copy()
                    for i in range(1,diff+1):
                        last_value = new_time[-1] + timestep
                        new_time = np.append(new_time, pd.Series(last_value))
                    xrpts = xrpts.assign_coords(timestamps=('frame', new_time[1:]))
                else: # equal (probably won't happen because ValueError should have been caused by unequal lengths)
                    xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
        # pt_path will have no data in it for world cam data, so it will make an xarray with just timestamps
        elif self.dlc_path is None or self.dlc_path == [] and self.timestamp_path is not None:
            if self.timestamp_path is not None and self.timestamp_path != []:
                # read in the time
                time = self.read_timestamp_file()
                # setup frame indices
                xrpts = xr.DataArray(np.zeros([len(time)-1]), dims=['frame'])
                # assign frame coordinates, then timestamps
                xrpts = xrpts.assign_coords({'frame':range(0,len(xrpts))})
                xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
            elif self.timestamp_path is None or self.timestamp_path == []:
                xrpts = None
        # if timestamps are missing, still read in and format as xarray
        elif self.dlc_path is not None and self.dlc_path != [] and self.timestamp_path is None:
            # open multianimal project with a different function than single animal h5 files
            if 'TOP' in self.camname and self.config['internals']['multianimal_top_project'] is True:
                # add a step to convert pickle files here?
                pts = self.open_dlc_h5_multianimal()
            # otherwise, use regular h5 file read-in
            else:
                pts, self.pt_names = self.open_dlc_h5()
            # label dimensions of the points dataarray
            xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
            # label the camera view
            xrpts.name = self.camname
        self.xrpts = xrpts

    def split_xyl(self):
        """ Convert xarray DataArray of DLC x and y positions and likelihood values into separate pandas data structures
        """
        data = self.xrpts
        names = list(data['point_loc'].values)
        thresh = self.config['internals']['likelihood_threshold']

        x_locs = []
        y_locs = []
        likeli_locs = []
        # seperate the lists of point names into x, y, and likelihood
        for loc_num in range(0, len(names)):
            loc = names[loc_num]
            if '_x' in loc:
                x_locs.append(loc)
            elif '_y' in loc:
                y_locs.append(loc)
            elif 'likeli' in loc:
                likeli_locs.append(loc)
        # get the xarray, split up into x, y,and likelihood
        for loc_num in range(0, len(likeli_locs)):
            pt_loc = likeli_locs[loc_num]
            if loc_num == 0:
                likeli_pts = data.sel(point_loc=pt_loc)
            elif loc_num > 0:
                likeli_pts = xr.concat([likeli_pts, data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
        for loc_num in range(0, len(x_locs)):
            pt_loc = x_locs[loc_num]
            # threshold from likelihood
            data.sel(point_loc=pt_loc)[data.sel(point_loc=pt_loc) < thresh] = np.nan
            if loc_num == 0:
                x_pts = data.sel(point_loc=pt_loc)
            elif loc_num > 0:
                x_pts = xr.concat([x_pts, data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
        for loc_num in range(0, len(y_locs)):
            pt_loc = y_locs[loc_num]
            # threshold from likelihood
            data.sel(point_loc=pt_loc)[data.sel(point_loc=pt_loc) < thresh] = np.nan
            if loc_num == 0:
                y_pts = data.sel(point_loc=pt_loc)
            elif loc_num > 0:
                y_pts = xr.concat([y_pts, data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
        x_pts = xr.DataArray.squeeze(x_pts)
        y_pts = xr.DataArray.squeeze(y_pts)
        likeli_pts = xr.DataArray.squeeze(likeli_pts)
        # convert to dataframe, transpose so points are columns
        x_vals = xr.DataArray.to_pandas(x_pts).T
        y_vals = xr.DataArray.to_pandas(y_pts).T
        likeli_pts = xr.DataArray.to_pandas(likeli_pts).T
        return x_vals, y_vals, likeli_pts