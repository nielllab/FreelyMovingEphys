"""
FreelyMovingEphys/src/eye.py
"""

import os
import sys
import json
from tqdm import tqdm
import multiprocessing

import numpy as np
import pandas as pd
import xarray as xr

import scipy.stats
import scipy.signal
import astropy.convolution

import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys

class Eyecam(fmEphys.Camera):
    """ Processing for head-mounted eye-facing camera.
    """
    def __init__(self, config, recording_name, recording_path, camname):
        fmEphys.Camera.__init__(self, config, recording_name, recording_path, camname)

        self.eye_fig_pts_dwnspl = 100

    def fit_ellipse(self, x, y):
        """ Fit an ellipse to points labeled around the perimeter of pupil.

        Parameters
        --------
        x : np.array
            Positions of points along the x-axis for a single video frame.
        y : np.array
            Positions of labeled points along the y-axis for a single video frame.

        Returns
        --------
        ellipse_dict : dict
            Parameters of the ellipse.
            X0: center at the x-axis of the non-tilt ellipse
            Y0: center at the y-axis of the non-tilt ellipse
            a: radius of the x-axis of the non-tilt ellipse
            b: radius of the y-axis of the non-tilt ellipse
            long_axis: radius of the long axis of the ellipse
            short_axis: radius of the short axis of the ellipse
            angle_to_x: angle from long axis to horizontal plane
            angle_from_x: angle from horizontal plane to long axis
            X0_in: center at the x-axis of the tilted ellipse
            Y0_in: center at the y-axis of the tilted ellipse
            phi: tilt orientation of the ellipse in radians
        """
        # remove bias of the ellipse
        meanX = np.mean(x)
        meanY = np.mean(y)
        x = x - meanX
        y = y - meanY

        # estimation of the conic equation
        X = np.array([x**2, x*y, y**2, x, y])
        X = np.stack(X).T
        a = np.dot(np.sum(X, axis=0), np.linalg.pinv(np.matmul(X.T,X)))

        # extract parameters from the conic equation
        a, b, c, d, e = a[0], a[1], a[2], a[3], a[4]

        # eigen decomp
        Q = np.array([[a, b/2],[b/2, c]])
        eig_val, eig_vec = np.linalg.eig(Q)

        # get angle to long axis
        if eig_val[0] < eig_val[1]:
            angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
        else:
            angle_to_x = np.arctan2(eig_vec[1,1], eig_vec[0,1])
        angle_from_x = angle_to_x
        orientation_rad = 0.5 * np.arctan2(b, (c-a))
        cos_phi = np.cos(orientation_rad)
        sin_phi = np.sin(orientation_rad)
        a, b, c, d, e = [a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
                        0,
                        a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
                        d*cos_phi - e*sin_phi,
                        d*sin_phi + e*cos_phi]
        meanX, meanY = [cos_phi*meanX - sin_phi*meanY,
                        sin_phi*meanX + cos_phi*meanY]

        # check if conc expression represents an ellipse
        test = a*c
        if test > 0:
            # make sure coefficients are positive as required
            if a<0:
                a, c, d, e = [-a, -c, -d, -e]

            # final ellipse parameters
            X0 = meanX - d/2/a
            Y0 = meanY - e/2/c
            F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
            a = np.sqrt(F/a)
            b = np.sqrt(F/c)
            long_axis = 2*np.maximum(a,b)
            short_axis = 2*np.minimum(a,b)

            # rotate axes backwards to find center point of original tilted ellipse
            R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
            P_in = R @ np.array([[X0],[Y0]])
            X0_in = P_in[0][0]
            Y0_in = P_in[1][0]

            # organize parameters in dictionary to return
            # makes some final modifications to values here, maybe those should be done above for cleanliness
            ellipse_dict = {'X0':X0, 'Y0':Y0, 'F':F, 'a':a, 'b':b, 'long_axis':long_axis/2, 'short_axis':short_axis/2,
                            'angle_to_x':angle_to_x, 'angle_from_x':angle_from_x, 'cos_phi':cos_phi, 'sin_phi':sin_phi,
                            'X0_in':X0_in, 'Y0_in':Y0_in, 'phi':orientation_rad}
        else:
            # if the conic equation didn't return an ellipse, don't return any real values and fill the dictionary with NaNs
            ellipse_dict = {'X0':np.nan, 'Y0':np.nan, 'F':np.nan, 'a':np.nan, 'b':np.nan, 'long_axis':np.nan, 'short_axis':np.nan,
                            'angle_to_x':np.nan, 'angle_from_x':np.nan, 'cos_phi':np.nan, 'sin_phi':np.nan,
                            'X0_in':np.nan, 'Y0_in':np.nan, 'phi':np.nan}
        
        return ellipse_dict

    def get_horizontal_vertical_rotation(self):
        # set up the pdf to be saved out with diagnostic figures
        pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_' + self.camname + '_tracking_figs.pdf')))
        
        # if this is a hf recording, read in existing fm camera center, scale, etc.
        # it should run all fm recordings first, so it will be possible to read in fm camera calibration parameters for every hf recording
        if self.config['internals']['force_eyecam_calibration_params'] == 'auto':
            if 'hf' in self.recording_name:
                path_to_existing_props = sorted(fmEphys.find('*fm_eyecameracalc_props.json', self.config['animal_directory'])) # should always go for fm1 before fm2
                if len(path_to_existing_props) == 0:
                    print('found no existing camera calibration properties from freely moving recording')
                    path_to_existing_props = None
                elif len(path_to_existing_props) == 1:
                    print('found one existing file of camera calirbation properties from freely moving recording')
                    path_to_existing_props = path_to_existing_props[0]
                elif len(path_to_existing_props) > 1:
                    print('found multiple existing files of camera calibration properties from freely moving recordings -- using first option from sorted list')
                    print(path_to_existing_props[0])
                    path_to_existing_props = path_to_existing_props[0]
                if path_to_existing_props is not None:
                    with open(path_to_existing_props, 'r') as fp:
                        existing_camera_calib_props = json.load(fp)
                elif path_to_existing_props is None:
                    # if a json of paramters can't be found, though, we'll get these values for the hf recording
                    existing_camera_calib_props = None
            elif 'fm' in self.recording_name or self.config['internals']['follow_strict_directory_naming']:
                existing_camera_calib_props = None
            else:
                existing_camera_calib_props = None
        elif self.config['internals']['force_eyecam_calibration_params'] == 'self':
            calibration_param_file = next(i for i in fmEphys.find('*fm_eyecameracalc_props.json', self.config['recording_path']) if self.camname in i)
            with open(calibration_param_file, 'r') as fp:
                existing_camera_calib_props = json.load(fp)

        # names of the different points
        eye_data = self.xrpts
        x_vals, y_vals, likeli_vals = self.split_xyl()
        likelihood_in = likeli_vals.values

        # subtract center of IR light reflection from all other pts
        if self.config['internals']['eye_has_reflection'] and self.config['internals']['subtract_reflection_position']:
            spot_xcent = np.mean(x_vals.iloc[:,-5:], 1)
            spot_ycent = np.mean(y_vals.iloc[:,-5:], 1)
            spot_likelihood = likelihood_in[:,-5:].copy()
            likelihood = likelihood_in[:,:-5]
            x_vals = x_vals.iloc[:,:-5].subtract(spot_xcent, axis=0)
            y_vals = y_vals.iloc[:,:-5].subtract(spot_ycent, axis=0)
        elif self.config['internals']['eye_has_reflection'] and not self.config['internals']['subtract_reflection_position']:
            x_vals = x_vals.iloc[:,:-5]
            y_vals = y_vals.iloc[:,:-5]
            likelihood = likelihood_in[:,:-5]
        elif self.config['internals']['eye_corners_first']:
            x_vals = x_vals.iloc[:,2:]
            y_vals = y_vals.iloc[:,2:]
            likelihood = likelihood_in[:,2:]
        
        # drop tear/outer eye points
        if self.config['internals']['eye_has_corners']:
            if self.config['internals']['eye_has_reflection'] and self.config['internals']['subtract_reflection_position']:
                x_vals = x_vals.iloc[:,:-2]
                y_vals = y_vals.iloc[:,:-2]
                likelihood = likelihood[:,:-2]
            if not self.config['internals']['eye_has_reflection'] and self.config['internals']['subtract_reflection_position']:
                x_vals = x_vals.iloc[:,:-2]
                y_vals = y_vals.iloc[:,:-2]
                likelihood = likelihood_in[:,:-2]
        else:
            likelihood = likelihood_in

        # get bools of when a frame is usable with the right number of points above threshold
        if self.config['internals']['subtract_reflection_position']:
            # if spot subtraction is being done, we should only include frames where all five pts marked around the ir spot are good (centroid would be off otherwise)
            usegood_eye = (np.sum(likelihood >= self.config['internals']['likelihood_threshold'], 1) >= self.config['internals']['ellipse_pts_needed_for_eye']) & (np.sum(spot_likelihood >= self.config['internals']['likelihood_threshold'], 1) >= self.config['internals']['pts_needed_for_reflection'])
            usegood_eyecalib = (np.sum(likelihood >= self.config['internals']['likelihood_threshold'], 1) >= self.config['internals']['ellipse_pts_needed_for_calibration']) & (np.sum(spot_likelihood >= self.config['internals']['likelihood_threshold'], 1) >= self.config['internals']['pts_needed_for_reflection'])
            usegood_reflec = (np.sum(spot_likelihood >= self.config['internals']['likelihood_threshold'], 1) >= self.config['internals']['pts_needed_for_reflection'])
        else:
            usegood_eye = np.sum(likelihood >= self.config['internals']['likelihood_threshold'], 1) >= self.config['internals']['ellipse_pts_needed_for_eye']
            usegood_eyecalib = np.sum(likelihood >= self.config['internals']['likelihood_threshold'], 1) >= self.config['internals']['ellipse_pts_needed_for_calibration']
        
        # how well did reflection track?
        if self.config['internals']['subtract_reflection_position']:
            plt.figure()
            plt.plot(np.sum(spot_likelihood >= self.config['internals']['likelihood_threshold'], 1)[0:-1:10])
            plt.title(str(np.round(np.mean(usegood_reflec), 3)) + ' good reflection; thresh=' + str(self.config['internals']['likelihood_threshold']))
            plt.ylabel('num good reflection points'); plt.xlabel('every 10th frame')
            pdf.savefig(); plt.close()

        # how well did eye track?
        plt.figure()
        plt.plot(np.sum(likelihood >= self.config['internals']['likelihood_threshold'], 1)[0:-1:10])
        plt.title(str(np.round(np.mean(usegood_eye), 3)) + ' good eye; thresh=' + str(self.config['internals']['likelihood_threshold']))
        plt.ylabel('num good pupil points'); plt.xlabel('every 10th frame')
        pdf.savefig(); plt.close()

        # hist of eye tracking quality
        plt.figure()
        plt.hist(np.sum(likelihood >= self.config['internals']['likelihood_threshold'], 1), bins=9, range=(0,9), density=True)
        plt.xlabel('num good eye points'); plt.ylabel('fraction of frames')
        pdf.savefig()
        plt.close()

        # threshold out pts more than a given distance away from nanmean of that point
        std_thresh_x = np.empty(np.shape(x_vals))
        for point_loc in range(0,np.size(x_vals, 1)):
            std_thresh_x[:,point_loc] = (np.absolute(np.nanmean(x_vals.iloc[:,point_loc]) - x_vals.iloc[:,point_loc]) / self.config['internals']['eye_pxl_per_cm']) > self.config['internals']['eye_dist_thresh_cm']
        std_thresh_y = np.empty(np.shape(y_vals))
        for point_loc in range(0,np.size(x_vals, 1)):
            std_thresh_y[:,point_loc] = (np.absolute(np.nanmean(y_vals.iloc[:,point_loc]) - y_vals.iloc[:,point_loc]) / self.config['internals']['eye_pxl_per_cm']) > self.config['internals']['eye_dist_thresh_cm']
        std_thresh_x = np.nanmean(std_thresh_x, 1)
        std_thresh_y = np.nanmean(std_thresh_y, 1)
        x_vals[std_thresh_x > 0] = np.nan
        y_vals[std_thresh_y > 0] = np.nan
        ellipse_params = np.empty([len(usegood_eye), 14])
        
        # step through each frame, fit an ellipse to points, and
        # add ellipse parameters to array with data for all frames together
        linalgerror = 0
        for step in tqdm(range(0,len(usegood_eye))):
            if usegood_eye[step] == True:
                try:
                    e_t = self.fit_ellipse(x_vals.iloc[step].values, y_vals.iloc[step].values)
                    ellipse_params[step] = [e_t['X0'], e_t['Y0'], e_t['F'], e_t['a'], e_t['b'],
                                            e_t['long_axis'], e_t['short_axis'], e_t['angle_to_x'], e_t['angle_from_x'],
                                            e_t['cos_phi'], e_t['sin_phi'], e_t['X0_in'], e_t['Y0_in'], e_t['phi']]
                except np.linalg.LinAlgError as e:
                    linalgerror = linalgerror + 1
                    e_t = {'X0':np.nan, 'Y0':np.nan, 'F':np.nan, 'a':np.nan, 'b':np.nan, 'long_axis':np.nan, 'short_axis':np.nan,
                                'angle_to_x':np.nan, 'angle_from_x':np.nan, 'cos_phi':np.nan, 'sin_phi':np.nan,
                                'X0_in':np.nan, 'Y0_in':np.nan, 'phi':np.nan}
                    ellipse_params[step] = [e_t['X0'], e_t['Y0'], e_t['F'], e_t['a'], e_t['b'],
                                        e_t['long_axis'] ,e_t['short_axis'], e_t['angle_to_x'], e_t['angle_from_x'],
                                        e_t['cos_phi'], e_t['sin_phi'], e_t['X0_in'], e_t['Y0_in'], e_t['phi']]
            elif usegood_eye[step] == False:
                e_t = {'X0':np.nan, 'Y0':np.nan, 'F':np.nan, 'a':np.nan, 'b':np.nan, 'long_axis':np.nan, 'short_axis':np.nan,
                                'angle_to_x':np.nan, 'angle_from_x':np.nan, 'cos_phi':np.nan, 'sin_phi':np.nan,
                                'X0_in':np.nan, 'Y0_in':np.nan, 'phi':np.nan}
                ellipse_params[step] = [e_t['X0'], e_t['Y0'], e_t['F'], e_t['a'], e_t['b'],
                                        e_t['long_axis'] ,e_t['short_axis'], e_t['angle_to_x'], e_t['angle_from_x'],
                                        e_t['cos_phi'], e_t['sin_phi'], e_t['X0_in'], e_t['Y0_in'], e_t['phi']]
        print('lin alg error count = ' + str(linalgerror))
        
        # list of all places where the ellipse meets threshold
        R = np.linspace(0,2*np.pi, 100)
        # (short axis / long axis) < thresh
        usegood_ellipcalb = np.where((usegood_eyecalib == True) & ((ellipse_params[:,6] / ellipse_params[:,5]) < self.config['internals']['eye_ellipticity_thresh']))
        
        # limit number of frames used for calibration
        if np.size(usegood_ellipcalb,1) > 50000:
            shortlist = sorted(np.random.choice(usegood_ellipcalb[0],size=50000, replace=False))
        else:
            shortlist = usegood_ellipcalb
        
        # find camera center
        A = np.vstack([np.cos(ellipse_params[shortlist,7]),np.sin(ellipse_params[shortlist,7])])
        b = np.expand_dims(np.diag(A.T@np.squeeze(ellipse_params[shortlist,11:13].T)),axis=1)
        # but only use the camera center from this recording if values were not read in from a json
        # in practice, this means hf recordings have their cam center thrown out and use the fm values read in
        if existing_camera_calib_props is None:
            cam_cent = np.linalg.inv(A@A.T)@A@b
        elif existing_camera_calib_props is not None:
            cam_cent = np.array([[float(existing_camera_calib_props['cam_cent_x'])],[float(existing_camera_calib_props['cam_cent_y'])]])
        
        # ellipticity and scale
        ellipticity = (ellipse_params[shortlist,6] / ellipse_params[shortlist,5]).T
        if existing_camera_calib_props is None:
            try:
                scale = np.nansum(np.sqrt(1-(ellipticity)**2)*(np.linalg.norm(ellipse_params[shortlist,11:13]-cam_cent.T,axis=0)))/np.sum(1-(ellipticity)**2)
            except ValueError:
                scale = np.nansum(np.sqrt(1-(ellipticity)**2)*(np.linalg.norm(ellipse_params[shortlist,11:13]-cam_cent.T,axis=1)))/np.sum(1-(ellipticity)**2)
        elif existing_camera_calib_props is not None:
            scale = float(existing_camera_calib_props['scale'])
        
        # angles
        theta = np.arcsin((ellipse_params[:,11]-cam_cent[0])/scale) # horizontal
        phi = np.arcsin((ellipse_params[:,12]-cam_cent[1])/np.cos(theta)/scale) # vertical
        
        # plots of theta and phi
        try:
            plt.figure()
            plt.plot(np.rad2deg(phi)[0:-1:10])
            plt.title('phi')
            plt.ylabel('deg'); plt.xlabel('every 10th frame')
            pdf.savefig(); plt.close()

            plt.figure()
            plt.plot(np.rad2deg(theta)[0:-1:10])
            plt.title('theta')
            plt.ylabel('deg'); plt.xlabel('every 10th frame')
            pdf.savefig(); plt.close()
        except Exception as e:
            print('figure error for plots of theta, phi')
            print(e)

        # organize data to return as an xarray of most essential parameters
        ellipse_df = pd.DataFrame({'theta':list(theta), 'phi':list(phi), 'longaxis':list(ellipse_params[:,5]), 'shortaxis':list(ellipse_params[:,6]),
                                'X0':list(ellipse_params[:,11]), 'Y0':list(ellipse_params[:,12]), 'ellipse_phi':list(ellipse_params[:,7])})
        ellipse_param_names = ['theta', 'phi', 'longaxis', 'shortaxis', 'X0', 'Y0', 'ellipse_phi']
        ellipse_out = xr.DataArray(ellipse_df, coords=[('frame', range(0, len(ellipse_df))), ('ellipse_params', ellipse_param_names)], dims=['frame', 'ellipse_params'])
        ellipse_out.attrs['cam_center_x'] = cam_cent[0,0]
        ellipse_out.attrs['cam_center_y'] = cam_cent[1,0]
        
        # ellipticity histogram
        fig_dwnsmpl = 100
        try:
            # hist of ellipticity
            plt.figure()
            plt.hist(ellipticity, density=True)
            plt.title('ellipticity; thresh='+str(self.config['internals']['eye_ellipticity_thresh']))
            plt.ylabel('ellipticity'); plt.xlabel('fraction of frames')
            pdf.savefig(); plt.close()
            
            # eye axes relative to center
            w = ellipse_params[:,7]
            plt.figure()
            for i in range(0,len(usegood_ellipcalb)):
                plt.plot((ellipse_params[usegood_ellipcalb[i::fig_dwnsmpl],11] + [-5*np.cos(w[usegood_ellipcalb[i::fig_dwnsmpl]]), 5*np.cos(w[usegood_ellipcalb[i::fig_dwnsmpl]])]), (ellipse_params[usegood_ellipcalb[i::fig_dwnsmpl],12] + [-5*np.sin(w[usegood_ellipcalb[i::fig_dwnsmpl]]), 5*np.sin(w[usegood_ellipcalb[i::fig_dwnsmpl]])]))
            plt.plot(cam_cent[0],cam_cent[1],'r*')
            plt.title('eye axes relative to center')
            pdf.savefig(); plt.close()
        except Exception as e:
            print('figure error in plots of ellipticity and axes relative to center')
            print(e)
            
        # check calibration
        try:
            xvals = np.linalg.norm(ellipse_params[usegood_eyecalib, 11:13].T - cam_cent, axis=0)
            yvals = scale * np.sqrt(1-(ellipse_params[usegood_eyecalib,6]/ellipse_params[usegood_eyecalib,5])**2)
            calib_mask = ~np.isnan(xvals) & ~np.isnan(yvals)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xvals[calib_mask], yvals[calib_mask].T)
        except ValueError:
            print('no good frames that meet criteria... check DLC tracking!')

        # save out camera center and scale as np array (but only if this is a freely moving recording)
        if 'fm' in self.recording_name or not self.config['internals']['follow_strict_directory_naming']:
            calib_props_dict = {'cam_cent_x':float(cam_cent[0]), 'cam_cent_y':float(cam_cent[1]), 'scale':float(scale), 'regression_r':float(r_value), 'regression_m':float(slope)}
            calib_props_dict_savepath = os.path.join(self.recording_path, str(self.recording_name+self.camname+'_fm_eyecameracalc_props.json'))
            print('saving calibration parameters to '+calib_props_dict_savepath)
            with open(calib_props_dict_savepath, 'w') as f:
                json.dump(calib_props_dict, f)
        
        # figures of scale and center
        try:
            plt.figure()
            plt.plot(xvals[::fig_dwnsmpl], yvals[::fig_dwnsmpl], '.', markersize=1)
            plt.plot(np.linspace(0,50), np.linspace(0,50), 'r')
            plt.title('scale=' + str(np.round(scale, 1)) + ' r=' + str(np.round(r_value, 1)) + ' m=' + str(np.round(slope, 1)))
            plt.xlabel('pupil camera dist'); plt.ylabel('scale * ellipticity')
            pdf.savefig(); plt.close()

            # calibration of camera center
            delta = (cam_cent - ellipse_params[:,11:13].T)
            short_usegood_req8 = usegood_eyecalib[::fig_dwnsmpl]
            sq_ellipcalib = np.squeeze(usegood_ellipcalb)
            short_list3 = sq_ellipcalib[::fig_dwnsmpl]
            plt.figure()
            plt.plot(np.linalg.norm(delta[:,short_usegood_req8],2,axis=0), ((delta[0,short_usegood_req8].T * np.cos(ellipse_params[short_usegood_req8,7])) + (delta[1,short_usegood_req8].T * np.sin(ellipse_params[short_usegood_req8, 7]))) / np.linalg.norm(delta[:, short_usegood_req8],2,axis=0).T, 'y.', markersize=1)
            plt.plot(np.linalg.norm(delta[:,short_list3],2,axis=0), ((delta[0,short_list3].T * np.cos(ellipse_params[short_list3,7])) + (delta[1,short_list3].T * np.sin(ellipse_params[short_list3, 7]))) / np.linalg.norm(delta[:, short_list3],2,axis=0).T, 'r.', markersize=1)
            plt.title('camera center calibration')
            plt.ylabel('abs([PC-EC]).[cosw;sinw]')
            plt.xlabel('abs(PC-EC)')
            patch0 = mpatches.Patch(color='y', label='all pts')
            patch1 = mpatches.Patch(color='y', label='calibration pts')
            plt.legend(handles=[patch0, patch1])
            pdf.savefig(); plt.close()

        except Exception as e:
            print('figure error in plots of scale and camera center')
            print(e)

        pdf.close()

        self.ellipse_params = ellipse_out

    def eye_diagnostic_video(self):
        """ Plot video of eye tracking.
        """
        # read in video, set up save file
        vidread = cv2.VideoCapture(self.video_path)
        width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))
        savepath = os.path.join(self.recording_path, (self.recording_name + '_' + self.camname + '_plot.avi'))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))

        # only do the first number of frames (limit of frames to use should be set in config dict)
        if self.config['internals']['video_frames_to_save'] > int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)):
            num_save_frames = int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            num_save_frames = self.config['internals']['video_frames_to_save']

        # iterate through frames
        for frame_num in tqdm(range(num_save_frames)):
            # read frame and make sure it's read in correctoy
            ret, frame = vidread.read()
            if not ret:
                break
            # plot on the frame if there is data to be used
            if self.xrpts is not None and self.ellipse_params is not None:
                try:
                    # get out ellipse long/short axes and put into tuple
                    ellipse_axes = (int(self.ellipse_params.sel(frame=frame_num, ellipse_params='longaxis').values), int(self.ellipse_params.sel(frame=frame_num, ellipse_params='shortaxis').values))
                    # get out ellipse phi and round to int -- note: this is ellipse_phi not phi
                    ellipse_phi = int(np.rad2deg(self.ellipse_params.sel(frame=frame_num, ellipse_params='ellipse_phi').values))
                    # get ellipse center out, round to int, and put into tuple
                    ellipse_cent = (int(self.ellipse_params.sel(frame=frame_num, ellipse_params='X0').values), int(self.ellipse_params.sel(frame=frame_num, ellipse_params='Y0').values))
                    # update this frame with an ellipse
                    frame = cv2.ellipse(frame, ellipse_cent, ellipse_axes, ellipse_phi, 0, 360, (255,0,0), 2) # ellipse in blue
                # skip if the ell data from this frame are bad
                except (ValueError, KeyError):
                    pass

                try:
                    # iterate through each point in the list
                    for k in range(0, len(self.xrpts.isel(frame=frame_num)), 3):
                        # get the point center of each point num, k
                        pt_cent = (int(self.xrpts.isel(frame=frame_num, point_loc=k).values), int(self.xrpts.isel(frame=frame_num, point_loc=k+1).values))
                        # compare to threshold set in config and plot
                        if self.xrpts.isel(frame=frame_num, point_loc=k+2).values < self.config['internals']['likelihood_threshold']: # bad points in red
                            frame = cv2.circle(frame, pt_cent, 3, (0,0,255), -1)
                        elif self.xrpts.isel(frame=frame_num, point_loc=k+2).values >= self.config['internals']['likelihood_threshold']: # good points in green
                            frame = cv2.circle(frame, pt_cent, 3, (0,255,0), -1)
                except (ValueError, KeyError):
                    pass
            out_vid.write(frame)
        out_vid.release()

    def sigmoid_curve(self, xval, a, b, c):
        """ Sigmoid curve function.
        """
        return a+(b-a)/(1+10**((c-xval)*2))

    def sigmoid_fit(self, d):
        """ Fit sigmoid.

        Parameters
        --------
        d : np.array
            ...

        Returns
        --------
        (popt, ci) : tuple
            popt : np.array

            ci : np.array
        
        """
        try:
            popt, pcov = self.sigmoid_curve(self.sigmoid_curve, xdata=range(1,len(d)+1),
                                    ydata=d, p0=[100.0,200.0,len(d)/2],
                                    method='lm', xtol=10**-3, ftol=10**-3)
            ci = np.sqrt(np.diagonal(pcov))
        except RuntimeError:
            popt = np.nan*np.zeros(4)
            ci = np.nan*np.zeros(4)
        return (popt, ci)

    def get_torsion_from_ridges(self):
        """ Get torsion (omega) from rotation of ridges along the edge of the pupil.
        """
        print('found ' + str(multiprocessing.cpu_count()) + ' as cpu count for multiprocessing')
        
        pdf = PdfPages(os.path.join(self.recording_path,(self.recording_name+'_'+self.camname+ '_pupil_rotation.pdf')))

        # set up range of degrees in radians
        rad_range = np.deg2rad(np.arange(360))

        # get the ellipse parameters for this trial from the time-interpolated xarray
        eye_theta = self.ellipse_params.sel(ellipse_params='theta').values
        eye_phi = self.ellipse_params.sel(ellipse_params='phi').values
        eye_longaxis= self.ellipse_params.sel(ellipse_params='longaxis').values
        eye_shortaxis = self.ellipse_params.sel(ellipse_params='shortaxis').values
        eye_centX = self.ellipse_params.sel(ellipse_params='X0').values
        eye_centY = self.ellipse_params.sel(ellipse_params='Y0').values

        # set up for the read-in video
        eyevid = cv2.VideoCapture(self.video_path)
        totalF = int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)) # this can be changed to a small number of frames for testing
        set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # set up for the multiprocessing that'll be used during sigmoid fit function
        n_proc = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=n_proc)
        n=0

        print('getting cross-section of pupil at each angle and fitting to sigmoid (slow)')
        key_error_count = 0
        for step in tqdm(np.arange(totalF)):
            try:
                # read frame
                eye_ret, eye_frame = eyevid.read()
                if not eye_ret:
                    break

                # convert to grey image
                eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

                # some configuration
                ranger = 10
                meanr = 0.5 * (eye_longaxis[step] + eye_shortaxis[step]) # mean radius
                r = range(int(meanr - ranger), int(meanr + ranger)) # range of values over mean radius (meanr)
                pupil_edge = np.zeros([360, len(r)]) # empty array that the calculated edge of the pupil will be put into

                rad_range = np.deg2rad(np.arange(360))
                # get cross-section of pupil at each angle 1-360 and fit to sigmoid
                for i in range(0, len(r)):
                    pupil_edge[:,i] = eye_frame[((eye_centY[step] + r[i]*(np.sin(rad_range))).astype(int), (eye_centY[step] + r[i]*(np.cos(rad_range))).astype(int))]
                d = pupil_edge[:,:]

                # apply sigmoid fit with multiprocessing
                param_mp = [pool.apply_async(self.sigmoid_fit, args=(d[n,:],)) for n in range(360)]
                params_output = [result.get() for result in param_mp]

                # unpack outputs of sigmoid fit
                params = []; ci = []
                for vals in params_output:
                    params.append(vals[0])
                    ci.append(vals[1])
                params = np.stack(params); ci = np.stack(ci)

                # extract radius variable from parameters
                rfit = params[:,2] - 1

                # if confidence interval in estimate is > fit_thresh pix, set to to NaN
                ci_temp = (ci[:,0] > 5) | (ci[:,1] > 5)  | (ci[:,2]>0.75)
                rfit[ci_temp] = np.nan

                # remove if luminance goes the wrong way (e.g. from reflectance)
                rfit[(params[:,1] - params[:,0]) < 10] = np.nan
                rfit[params[:,1] > 250] = np.nan

                try:
                    # median filter
                    rfit_filt = fmEphys.nanmedfilt(rfit, 5)

                    # subtract baseline because our points aren't perfectly centered on ellipse
                    filtsize = 31
                    rfit_conv = rfit_filt - astropy.convolution.convolve(rfit_filt, np.ones(filtsize)/filtsize, boundary='wrap')

                except ValueError as e: # in case every value in rfit is NaN
                    rfit = np.nan*np.zeros(360)
                    rfit_conv = np.nan*np.zeros(360)
            except (KeyError, ValueError) as e:
                key_error_count = key_error_count + 1
                rfit = np.nan*np.zeros(360)
                rfit_conv = np.nan*np.zeros(360)

            # get rid of outlier points
            rfit_conv[np.abs(rfit_conv)>1.5] = np.nan

            # save out pupil edge data into one xarray for all frames
            if step == 0:
                rfit_conv_xr = xr.DataArray(rfit_conv)
                rfit_conv_xr['frame'] = step
                rfit_conv_xr = xr.DataArray.rename(rfit_conv_xr, {'dim_0':'deg'})

                rfit_xr = xr.DataArray(rfit)
                rfit_xr['frame'] = step
                rfit_xr = xr.DataArray.rename(rfit_xr, {'dim_0':'deg'})
            
            if step > 0:
                rfit_conv_temp = xr.DataArray(rfit_conv)
                rfit_conv_temp['frame'] = step
                rfit_conv_temp = xr.DataArray.rename(rfit_conv_temp, {'dim_0':'deg'})
                rfit_conv_xr = xr.concat([rfit_conv_xr, rfit_conv_temp], dim='frame', fill_value=np.nan)

                rfit_temp = xr.DataArray(rfit)
                rfit_temp['frame'] = step
                rfit_temp = xr.DataArray.rename(rfit_temp, {'dim_0':'deg'})
                rfit_xr = xr.concat([rfit_xr, rfit_temp], dim='frame', fill_value=np.nan)

        # threshold out any frames with large or small rfit_conv distributions
        for frame in range(0,np.size(rfit_conv_xr,0)):
            if np.min(rfit_conv_xr[frame,:]) < -10 or np.max(rfit_conv_xr[frame,:]) > 10:
                rfit_conv_xr[frame,:] = np.nan

        # correlation across first minute of recording
        timepoint_corr_rfit = pd.DataFrame(rfit_conv_xr.isel(frame=range(0,3600)).values).T.corr()

        # plot the correlation matrix of rfit over all timepoints
        plt.figure()
        fig, ax = plt.subplots()
        im = ax.imshow(timepoint_corr_rfit)
        ax.set_title('correlation of radius fit during first min. of recording')
        fig.colorbar(im, ax=ax)
        pdf.savefig(); plt.close()

        n = np.size(rfit_conv_xr.values, 0)
        pupil_update = rfit_conv_xr.values.copy()
        total_shift = np.zeros(n); peak = np.zeros(n)
        c = total_shift.copy()
        template = np.nanmean(rfit_conv_xr.values, 0)

        # calculate mean as template
        try:
            template_rfitconv_cc, template_rfit_cc_lags = fmEphys.nanxcorr(rfit_conv_xr[7].values, template, 30)
            template_nanxcorr = True
        except ZeroDivisionError:
            template_nanxcorr = False

        plt.figure()
        plt.plot(template)
        plt.title('mean as template')
        pdf.savefig(); plt.close()

        if template_nanxcorr is True:
            plt.figure()
            plt.plot(template_rfitconv_cc)
            plt.title('rfit_conv template cross correlation')
            pdf.savefig(); plt.close()

        # xcorr of two random timepoints
        try:
            t0 = np.random.random_integers(0,totalF-1); t1 = np.random.random_integers(0,totalF-1)
            rfit2times_cc, rfit2times_lags = fmEphys.nanxcorr(rfit_conv_xr.isel(frame=t0).values, rfit_conv_xr.isel(frame=t1).values, 10)
            rand_frames = True
        except ZeroDivisionError:
            rand_frames = False
        if rand_frames is True:
            plt.figure()
            plt.plot(rfit2times_cc, 'b-')
            plt.title('nanxcorr of frames ' + str(t0) + ' and ' + str(t1))
            pdf.savefig(); plt.close()

        num_rfit_samples_to_plot = 100
        ind2plot_rfit = sorted(np.random.randint(0,totalF-1,num_rfit_samples_to_plot))

        # iterative fit to alignment
        # start with mean as template
        # on each iteration, shift individual frames to max xcorr with template
        # then recalculate mean template
        print('doing iterative fit for alignment of each frame')
        for rep in tqdm(range(0,12)): # twelve iterations
            # for each frame, get correlation, and shift
            for frame_num in range(0,n): # do all frames
                try:
                    xc, lags = fmEphys.nanxcorr(template, pupil_update[frame_num,:], 20)
                    c[frame_num] = np.amax(xc) # value of max
                    peaklag = np.argmax(xc) # position of max
                    peak[frame_num] = lags[peaklag]
                    total_shift[frame_num] = total_shift[frame_num] + peak[frame_num]
                    pupil_update[frame_num,:] = np.roll(pupil_update[frame_num,:], int(peak[frame_num]))
                except ZeroDivisionError:
                    total_shift[frame_num] = np.nan
                    pupil_update[frame_num,:] = np.nan

            template = np.nanmean(pupil_update, axis=0) # update template

            # plot template with pupil_update for each iteration of fit
            plt.figure()
            plt.title('pupil_update of rep='+str(rep)+' in iterative fit')
            plt.plot(pupil_update[ind2plot_rfit,:].T, alpha=0.2)
            plt.plot(template, 'k--', alpha=0.8)
            pdf.savefig(); plt.close()

            # histogram of correlations
            plt.figure()
            plt.title('correlations of rep='+str(rep)+' in iterative fit')
            plt.hist(c[c>0], bins=300) # gets rid of NaNs in plot
            pdf.savefig(); plt.close()

        win = 5
        shift_nan = -total_shift
        shift_nan[c < 0.35] = np.nan
        shift_nan = shift_nan - np.nanmedian(shift_nan)
        shift_nan[shift_nan >= 20] = np.nan; shift_nan[shift_nan <= -20] = np.nan # get rid of very large shifts
        shift_smooth = scipy.signal.medfilt(shift_nan,3)  # median filt to get rid of outliers
        shift_smooth = astropy.convolution.convolve(shift_nan, np.ones(win)/win)  # convolve to smooth and fill in nans
        shift_smooth = shift_smooth - np.nanmedian(shift_smooth)

        plt.figure()
        plt.plot(shift_nan)
        plt.title('shift nan')
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(shift_smooth)
        plt.title('shift smooth')
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(shift_smooth[:3600])
        plt.title('shift smooth for first 1min of recording')
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(shift_smooth, linewidth = 4, label = 'shift_smooth')
        plt.plot(-total_shift,linewidth=1, alpha = 0.5, label='raw total_shift')
        plt.legend()
        plt.title('shift_smooth and raw total shift')
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(rfit_xr.isel(frame=ind2plot_rfit).T, alpha=0.2)
        plt.plot(np.nanmean(rfit_xr.T,1), 'b--', alpha=0.8)
        plt.title('rfit for 100 random frames')
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(rfit_conv_xr.isel(frame=ind2plot_rfit).T, alpha=0.2)
        plt.plot(np.nanmean(rfit_conv_xr.T,1), 'b--', alpha=0.8)
        plt.title('rfit_conv for 100 random frames')
        pdf.savefig(); plt.close()

        # plot of 5 random frames' rfit_conv
        plt.figure()
        fig, axs = plt.subplots(5,1)
        axs = axs.ravel()
        for i in range(0,5):
            rand_num = np.random.randint(0,totalF-1)
            axs[i].plot(rfit_conv_xr.isel(frame=rand_num))
            axs[i].set_title(('rfit conv; frame ' + str(rand_num)))
        pdf.savefig()
        plt.close()

        shift_smooth1 = xr.DataArray(shift_smooth, dims=['frame'])

        if self.config['internals']['diagnostic_preprocessing_videos'] is True:
            eyevid = cv2.VideoCapture(self.video_path)
            vidsavepath = os.path.join(self.recording_path,str(self.recording_name+'_pupil_rotation_rep'+str(rep)+'_'+self.camname+'.avi'))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vidout = cv2.VideoWriter(vidsavepath, fourcc, 60.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            if self.config['internals']['video_frames_to_save'] > int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)):
                num_save_frames = int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                num_save_frames = self.config['internals']['video_frames_to_save']

            print('plotting pupil rotation on eye video')
            for step in tqdm(range(num_save_frames)):
                eye_ret, eye_frame = eyevid.read()
                eye_frame0 = eye_frame.copy()
                if not eye_ret:
                    break

                # get ellipse parameters for this time
                current_longaxis = eye_longaxis.sel(frame=step).values
                current_shortaxis = eye_shortaxis.sel(frame=step).values
                current_centX = eye_centX.sel(frame=step).values
                current_centY = eye_centY.sel(frame=step).values

                # plot the ellipse edge
                rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
                for deg_th in range(0,360):
                    rad_th = rad_range[deg_th]
                    edge_x = np.round(current_centX+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.cos(rad_th))
                    edge_y = np.round(current_centY+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.sin(rad_th))
                    if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                        eye_frame1 = cv2.circle(eye_frame, (int(edge_x),int(edge_y)), 1, (235,52,155), thickness=-1)

                # plot the rotation of the eye as a vertical line made up of many circles
                for d in np.linspace(-0.5,0.5,100):
                    rot_x = np.round(current_centX + d*(np.rad2deg(np.cos(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                    rot_y = np.round(current_centY + d*(np.rad2deg(np.sin(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                    if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                        eye_frame1 = cv2.circle(eye_frame1, (int(rot_x),int(rot_y)),1,(255,255,255),thickness=-1)

                # plot the center of the eye on the frame as a larger dot than the others
                if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:
                    eye_frame1 = cv2.circle(eye_frame1, (int(current_centX),int(current_centY)),3,(0,255,0),thickness=-1)

                frame_out = np.concatenate([eye_frame0, eye_frame1], axis=1)

                vidout.write(frame_out)

            vidout.release()

        shift = xr.DataArray(pd.DataFrame(shift_smooth), dims=['frame','shift'])
        print('key/value error count during sigmoid fit: ' + str(key_error_count))

        # plotting omega on some random frames to be saved into the pdf
        eyevid = cv2.VideoCapture(self.video_path)
        rand_frame_nums = list(np.random.randint(0,int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)), size=20))
        
        for step in rand_frame_nums:
            eyevid.set(cv2.CAP_PROP_POS_FRAMES, step)
            eye_ret, eye_frame = eyevid.read()
            if not eye_ret:
                break
            plt.subplots(2,2)
            plt.subplot(221)
            plt.imshow(eye_frame.astype(np.uint8), cmap='gray')

            # get ellipse parameters for this time
            current_longaxis = eye_longaxis.sel(frame=step).values
            current_shortaxis = eye_shortaxis.sel(frame=step).values
            current_centX = eye_centX.sel(frame=step).values
            current_centY = eye_centY.sel(frame=step).values
            
            # plot the ellipse edge
            rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
            plt.subplot(222)
            plt.imshow(eye_frame.astype(np.uint8), cmap='gray')
            for deg_th in range(0,360):
                rad_th = rad_range[deg_th]
                edge_x = np.round(current_centX+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.cos(rad_th))
                edge_y = np.round(current_centY+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.sin(rad_th))
                if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                    plt.plot(edge_x, edge_y, color='orange', marker='.',markersize=1,alpha=0.1)
            
            # plot the rotation of the eye as a vertical line made up of many circles
            plt.subplot(223)
            plt.imshow(eye_frame.astype(np.uint8), cmap='gray')
            for d in np.linspace(-0.5,0.5,100):
                rot_x = np.round(current_centX + d*(np.rad2deg(np.cos(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                rot_y = np.round(current_centY + d*(np.rad2deg(np.sin(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                    plt.plot(rot_x, rot_y, color='white',marker='.',markersize=1,alpha=0.1)

            plt.subplot(223)
            plt.imshow(eye_frame.astype(np.uint8), cmap='gray')
            # plot the center of the eye on the frame as a larger dot than the others
            if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:
                plt.plot(int(current_centX),int(current_centY), color='blue', marker='o')

            plt.subplot(224)
            plt.imshow(eye_frame.astype(np.uint8), cmap='gray')
            for deg_th in range(0,360):
                rad_th = rad_range[deg_th]
                edge_x = np.round(current_centX+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.cos(rad_th))
                edge_y = np.round(current_centY+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.sin(rad_th))
                if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                    plt.plot(edge_x, edge_y, color='orange', marker='.',markersize=1,alpha=0.1)
            for d in np.linspace(-0.5,0.5,100):
                rot_x = np.round(current_centX + d*(np.rad2deg(np.cos(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                rot_y = np.round(current_centY + d*(np.rad2deg(np.sin(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                    plt.plot(rot_x, rot_y, color='white',marker='.',markersize=1,alpha=0.1)
            # plot the center of the eye on the frame as a larger dot than the others
            if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:
                plt.plot(int(current_centX),int(current_centY), color='blue', marker='o')

            pdf.savefig()
            plt.close()

        pdf.close()

        self.shift = shift
        self.rfit = rfit_xr
        self.rfit_conv = rfit_conv_xr

    # def get_torsion_from_markers(self):

    def save_params(self):
        self.xrpts.name = self.camname+'_pts'
        self.xrframes.name = self.camname+'_video'
        self.ellipse_params.name = self.camname+'_ellipse_params'
        merged_data = [self.xrpts, self.ellipse_params, self.xrframes]

        if self.config['internals']['get_torsion_from_ridges']:
            self.rfit.name = self.camname+'_pupil_radius'
            self.shift.name = self.camname+'_omega'
            self.rfit_conv.name = self.camname+'_conv_pupil_radius'
            merged_data = merged_data + [self.rfit, self.shift, self.rfit_conv]
        if self.config['internals']['get_torsion_from_markers']:
            print('Torsion from markers not implemented.')
            sys.exit()

        self.safe_merge(merged_data)
        self.data.to_netcdf(os.path.join(self.recording_path,str(self.recording_name+'_'+self.camname+'.nc')),
                    engine='netcdf4', encoding={self.camname+'_video':{"zlib": True, "complevel": 4}})

    def process(self):
        if self.config['main']['deinterlace'] and not self.config['internals']['flip_headcams']['run']:
            self.deinterlace()
        elif not self.config['main']['deinterlace'] and self.config['internals']['flip_headcams']['run']:
            self.flip_headcams()
        elif self.config['main']['deinterlace'] and self.config['internals']['flip_headcams']['run']:
            print('Config options deinterlace and flip_headcams are both True, which conflict with each other.')
            sys.exit()

        if self.config['internals']['apply_gamma_to_eyecam']:
            self.auto_contrast()

        if self.config['main']['pose_estimation']:
            self.pose_estimation()

        if self.config['main']['parameters']:
            self.gather_camera_files()
            self.pack_position_data()
            self.pack_video_frames()
            self.get_horizontal_vertical_rotation()
            if self.config['internals']['get_torsion_from_ridges']:
                self.get_torsion_from_ridges()
            if self.config['internals']['get_torsion_from_markers']:
                print('Torsion from markers not implemented.')
                sys.exit()
            if self.config['internals']['diagnostic_preprocessing_videos']:
                self.eye_diagnostic_video()
            self.save_params()