"""Tracking mouse pupil from head-mounted eye-facing camera.

Notes
-----
Haven't used the cyclotorsion code in a long time, and it has since
been reorganized/refactored. So, it may error.
"""

import os
import json
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import astropy
import matplotlib.pyplot as plt
from scipy import stats, signal, optimize
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys

def fit_ellipse(x, y):
    """Fit an ellipse to points labeled around the perimeter of pupil.

    Parameters
    ----------
    x : np.array
        Positions of points along the x-axis for a single video frame.
    y : np.array
        Positions of labeled points along the y-axis for a single video frame.

    Returns
    -------
    ellipse_dict : dict
        Parameters of the ellipse...
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

    # Remove bias
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x = x - mean_x
    y = y - mean_y

    # Estimate conic equation
    X = np.array([x**2, x*y, y**2, x, y])
    X = np.stack(X).T
    a,b,c,d,e = np.dot(np.sum(X, axis=0),
                       np.linalg.pinv(np.matmul(X.T, X)))

    # Eigen decomp
    Q = np.array([[a, b/2], [b/2, c]])
    eig_val, eig_vec = np.linalg.eig(Q)

    # Get angle to long axis
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

    mean_x, mean_y = [cos_phi*mean_x - sin_phi*mean_y,
                    sin_phi*mean_x + cos_phi*mean_y]

    # Check if conc expression represents an ellipse
    test = a*c

    if test > 0:
        # Make sure coefficients are positive as required
        if a<0:
            a, c, d, e = [-a, -c, -d, -e]

        # Final ellipse parameters
        X0 = mean_x - d/2/a
        Y0 = mean_y - e/2/c
        F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
        a = np.sqrt(F/a)
        b = np.sqrt(F/c)

        long_axis = 2*np.maximum(a,b)
        short_axis = 2*np.minimum(a,b)

        # Rotate axes backwards to find center point of the
        # original tilted ellipse
        R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
        P_in = R @ np.array([[X0],[Y0]])
        X0_in = P_in[0][0]
        Y0_in = P_in[1][0]

        ellipse_dict = {
            'X0':X0,
            'Y0':Y0,
            'F':F,
            'a':a,
            'b':b,
            'long_axis':long_axis/2,
            'short_axis':short_axis/2,
            'angle_to_x':angle_to_x,
            'angle_from_x':angle_from_x,
            'cos_phi':cos_phi,
            'sin_phi':sin_phi,
            'X0_in':X0_in,
            'Y0_in':Y0_in,
            'phi':orientation_rad
        }
    else:
        # If the conic equation didn't return an ellipse, do not
        # return any real values and fill the dictionary with NaNs.
        dict_keys = ['X0','Y0','F','a','b','long_axis','short_axis',
                     'angle_to_x','angle_from_x','cos_phi','sin_phi',
                     'X0_in','Y0_in','phi']
        dict_vals = list(np.ones([len(dict_keys)]) * np.nan)

        ellipse_dict = dict(zip(dict_keys, dict_vals))
    
    return ellipse_dict

def track_pupil(cfg):
    """
    cfg can be None and default options will be used
    """

    pdf_savepath = os.path.join(cfg['_rpath'],
            '{}_{}_diagnostics.pdf'.format(cfg['_rname'], cfg['_cam']))
    pdf = PdfPages(pdf_savepath)

    # Find files
    dlc_path = fmEphys.find('*{}*DLC*.h5'.format(cfg['_cam']), cfg['_rpath'], mr=True)
        
    # If this is a hf recording, read in existing fm
    # camera center, scale, etc.
    # It should run all fm recordings first, so it
    # will be possible to read in fm camera calibration
    # parameters for every hf recording
    if 'hf' in cfg['_stim']:

        eyeparams_path = fmEphys.find('*eyeparams.json', cfg['apath'], mr=True, none_possible=True)
        
        if eyeparams_path is not None:
            
            eyeparams = fmEphys.read_yaml(eyeparams_path)

    if (eyeparams_path is None) or ('fm' in cfg['_rname']):
        
        eyeparams = None

    # Read in x/y positions and likelihood from DeepLabCut
    all_dlc_data = fmEphys.read_DLC_data(dlc_path).astype(float)
    x_pos, y_pos, pupil_likeli = fmEphys.split_xyl(all_dlc_data)
    
    # Arrange naming for DLC labeled points
    spot_pts = cfg['eye_labels']['spot']
    spotcent_pt = cfg['eye_labels']['spot_center'][0]
    _use_spotxnames = []
    _use_spotynames = []
    _use_spotlnames = []
    for pt in spot_pts:
        _use_spotxnames.append('{}_x'.format(pt))
        _use_spotynames.append('{}_y'.format(pt))
        _use_spotlnames.append('{}_likelihood'.format(pt))
    _use_spotxnames.append('{}_x'.format(spotcent_pt))
    _use_spotynames.append('{}_y'.format(spotcent_pt))
    _use_spotlnames.append('{}_likelihood'.format(spotcent_pt))

    pupil_pts = cfg['eye_labels']['pupil']
    _use_pupilxnames = []
    _use_pupilynames = []
    _use_pupillnames = []
    for pt in pupil_pts:
        _use_pupilxnames.append('{}_x'.format(pt))
        _use_pupilynames.append('{}_y'.format(pt))
        _use_pupillnames.append('{}_likelihood'.format(pt))

    # drop tear/outer eye points
    if cfg['canthus_is_labeled']:

        _drop_xnames = []
        _drop_ynames = []
        _drop_lnames = []
        for pt in cfg['eye_labels']['other']:
            _drop_xnames.append('{}_x'.format(pt))
            _drop_ynames.append('{}_y'.format(pt))
            _drop_lnames.append('{}_likelihood'.format(pt))

        x_pos.drop(columns=_drop_xnames, inplace=True)
        y_pos.drop(columns=_drop_ynames, inplace=True)
        pupil_likeli.drop(columns=_drop_lnames, inplace=True)

    # subtract center of IR light reflection from all other pts
    if cfg['IRspot_is_labeled'] and cfg['do_eyecam_spotsub']:

        spot_xcent = np.nanmean(x_pos.loc[:, _use_spotxnames].copy(), 1)
        spot_ycent = np.nanmean(y_pos.loc[:, _use_spotynames].copy(), 1)
        spot_likeli = pupil_likeli.loc[:, _use_spotlnames].copy()

        x_pos = x_pos.loc[:, _use_pupilxnames].subtract(spot_xcent, axis=0)
        y_pos = y_pos.loc[:, _use_pupilynames].subtract(spot_ycent, axis=0)
        pupil_likeli = pupil_likeli.loc[:, _use_pupillnames]

    # drop the IR spot points without doing their subtraction
    elif cfg['IRspot_is_labeled'] and not cfg['do_eyecam_spotsub']:

        x_pos = x_pos.loc[:, _use_pupilxnames]
        y_pos = y_pos.loc[:, _use_pupilynames]
        pupil_likeli = pupil_likeli.loc[:, _use_pupillnames]

    # get bools of when a frame is usable with the right
    # number of points above threshold
    if cfg['do_eyecam_spotsub']:

        # if spot subtraction is being done, we should only include
        # frames where all five pts marked around the ir spot are
        # good (centroid would be off otherwise)
        pupil_count = np.sum(pupil_likeli >= cfg['l_thresh'], 1)
        spot_count = np.sum(spot_likeli >= cfg['l_thresh'], 1)

        use_pupil = (pupil_count >= cfg['ellipse_useN'])        \
                        & (spot_count >= cfg['reflection_useN'])

        cal_pupil = (pupil_count >= cfg['ellipse_calN'])        \
                        & (spot_count >= cfg['reflection_useN'])

        use_spot = (spot_count >= cfg['reflection_useN'])

    elif not cfg['do_eyecam_spotsub']:

        pupil_count = np.sum(pupil_likeli >= cfg['l_thresh'], 1)

        use_pupil = (pupil_count >= cfg['ellipse_useN'])
        cal_pupil = (pupil_count >= cfg['ellipse_calN'])

    # how well did eye track?
    fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(2,2, figsize=(11,8.5), dpi=300)

    every10_time = np.linspace(0, len(pupil_count)/60/60, 6) # min

    ax0.plot(pupil_count[::10], color='k')
    ax0.set_title('{:.3}% good'.format(np.mean(use_pupil)*100))
    ax0.set_ylabel('# good pupil pts')
    ax0.set_yticks(every10_time)
    ax0.set_xlabel('time (min)')

    ax1.hist(pupil_count, color='k', bins=9, range=(0,9), density=True)
    ax1.set_xlabel('# good pupil pts')
    ax1.set_ylabel('fraction frames')

    if cfg['do_eyecam_spotsub']:
        ax2.plot(spot_count[::10])
        ax2.set_title('{:.3}% good'.format(np.mean(use_spot)*100))
        ax2.set_ylabel('# good reflec. pts')
        ax2.set_xlabel('time (min)')
        ax2.set_yticks(every10_time)

        ax3.hist(spot_count, color='k', bins=9, range=(0,9), density=True)
        ax3.set_xlabel('# good reflec. pts')
        ax3.set_ylabel('frac. frames')
    else:
        ax2.axis('off')
        ax3.axis('off')

    fig.tight_layout()
    pdf.savefig()
    plt.close()

    # Threshold out pts more than a given distance
    # away from nanmean of that point
    std_thresh_x = np.empty(np.shape(x_pos))
    for f in range(np.size(x_pos, 1)):
        std_thresh_x[:,f] = (np.abs(np.nanmean(x_pos.iloc[:,f])  \
            - x_pos.iloc[:,f]) / cfg['pupil_pxl2cm']) > cfg['distance_thresh']
    
    std_thresh_y = np.empty(np.shape(y_pos))
    for f in range(np.size(y_pos, 1)):
        std_thresh_y[:,f] = (np.abs(np.nanmean(y_pos.iloc[:,f])  \
            - y_pos.iloc[:,f]) / cfg['pupil_pxl2cm']) > cfg['distance_thresh']
    
    std_thresh_x = np.nanmean(std_thresh_x, 1)
    std_thresh_y = np.nanmean(std_thresh_y, 1)

    x_pos[std_thresh_x > 0] = np.nan
    y_pos[std_thresh_y > 0] = np.nan
        
    # step through each frame, fit an ellipse to points, and
    # add ellipse parameters to array with data for all frames together
    cols = ['X0','Y0', # 0 1
            'F','a','b', # 2 3 4
            'long_axis','short_axis', # 5 6
            'angle_to_x','angle_from_x', # 7 8
            'cos_phi','sin_phi', # 9 10
            'X0_in','Y0_in', # 11 12
            'phi'] # 13

    ellipse = pd.DataFrame(np.zeros([len(use_pupil), len(cols)])*np.nan, columns=cols)

    linalg_errCount = 0
    for f in tqdm(range(len(use_pupil))):
        if use_pupil[f] is True:
            try:
                ef = fit_ellipse(x_pos.iloc[f].values,
                                 y_pos.iloc[f].values)
                for k,v in ef.items():
                    ellipse.at[f,k] = v

            except np.linalg.LinAlgError:
                linalg_errCount += 1
                pass

    print('ellipse fit encounted {} linalg errors (# frames={})'.format(  \
                    linalg_errCount, len(use_pupil)))
        
    # list of all places where the ellipse meets threshold
    # (short axis / long axis) < thresh
    ellcal = np.where(cal_pupil & ((ellipse['short_axis'] / ellipse['long_axis'])  \
                    < cfg['ellipticity_thresh'])).flatten()
    
    # limit number of frames used for calibration
    # make a shorter version of the list
    if len(ellcal) > 50000:
        ellcal_s = sorted(np.random.choice(ellcal, size=50000, replace=False))
    else:
        ellcal_s = ellcal

    # find camera center
    cam_A = np.vstack([np.cos(ellipse.loc[ellcal_s, 'angle_to_x']),  \
                       np.sin(ellipse.loc[ellcal_s, 'angle_to_x'])])

    cam_b = np.expand_dims(np.diag(cam_A.T @ \
            np.squeeze(ellipse.loc[ellcal_s,['X0_in','Y0_in']].T)), axis=1)

    # but only use the camera center from this recording if
    # values were not read in from a json
    # in practice, this means hf recordings have their
    # cam center thrown out and use the fm values read in
    if eyeparams is None:
        cam_center = np.linalg.inv(cam_A @ cam_A.T) @ cam_A @ cam_b

    elif eyeparams is not None:
        cam_center = np.array([[float(eyeparams['cam_cent_x'])],  \
                               [float(eyeparams['cam_cent_y'])]])
        
    # ellipticity and scale
    ellipticity = (ellipse.loc[ellcal_s,'short_axis'] / ellipse.loc[ellcal_s,'long_axis']).T

    if eyeparams is None:
        try:
            scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *  \
                    (np.linalg.norm(ellipse.loc[ellcal_s,['X0_in','Y0_in']] -  \
                    cam_center.T, axis=0))) / np.sum(1 - (ellipticity)**2 )
        
        except ValueError:
            # swap axis that linalg.norm is calculated over (from axis=0 to axis=1)
            # I don't remember why I did this try/except or how often
            # the code is going into this exception
            # I should debug this and see why it would ever happen...
            scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *  \
                    (np.linalg.norm(ellipse.loc[ellcal_s,['X0_in','Y0_in']] -  \
                    cam_center.T, axis=1))) / np.sum(1 - (ellipticity)**2 )
    
    elif eyeparams is not None:
        scale = float(eyeparams['scale'])
        
    # horizontal angle (rad)
    theta = np.arcsin((ellipse['X0_in'] - cam_center[0]) / scale)

    # vertical angle (rad)
    phi = np.arcsin((ellipse['Y0_in'] - cam_center[1]) / np.cos(theta) / scale)

    # FIGURE: theta & phi
    fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)
    
    ax0.plot(np.rad2deg(theta)[::10], color='k')
    ax0.set_ylabel('theta (deg)')
    ax0.set_xlabel('every 10th frame')

    ax1.plot(np.rad2deg(phi)[::10], color='k')
    ax1.set_ylabel('phi (deg)')
    ax1.set_xlabel('every 10th frame')

    ax2.plot(ellipticity[::10], color='k')
    ax2.set_ylabel('ellipticity')
    ax2.set_xlabel('every 10th frame')

    ax3.hist(np.rad2deg(theta), density=True, color='k')
    ax3.set_ylabel('theta (deg)')
    ax3.set_xlabel('frac. frames')

    ax4.hist(np.rad2deg(phi), density=True, color='k')
    ax4.set_ylabel('')
    ax4.set_xlabel('frac. frames')

    ax5.hist(ellipticity, density=True, color='k')
    ax5.set_ylabel('ellipticity')
    ax5.set_xlabel('frac. frames')

    fig.tight_layout()
    pdf.savefig(); plt.close()
        
    # Eye axes relative to center
    ds = 100
    w = ellipse['angle_to_x'].copy().to_numpy()
    x = ellipse['X0_in'].copy().to_numpy() # 11
    y = ellipse['Y0_in'].copy().to_numpy() # 12

    fig, [[ax0,ax1],[ax3,ax4]] = plt.subplots(2,2, figsize=(11,8.5), dpi=300)
    
    # Position for frames
    for f in ellcal[::ds]:
        ax0.plot(
            (x[f] + [-5 * np.cos(w[f]), 5 * np.cos(w[f])]),
            (y[f] + [-5 * np.sin(w[f]), 5 * np.sin(w[f])]),
            '.', markersize=1.5)
    # Camera center
    ax0.plot(cam_center[0], cam_center[1], 'r*')
    ax0.set_title('eye axes relative to center')
            
    # Check calibration
    xvals = np.linalg.norm(ellipse.loc[cal_pupil, ['X0_in','Y0_in']].copy().to_numpy().T  \
            - cam_center, axis=0)
    yvals = scale * np.sqrt(1 - (ellipse.loc[cal_pupil,'short_axis'].copy().to_numpy()  \
            / ellipse.loc[cal_pupil,'long_axis'].copy().to_numpy()) **2 )
    tmp_mask = ~np.isnan(xvals) & ~np.isnan(yvals)

    slope, _, r_value, _, _ = stats.linregress(xvals[tmp_mask], yvals[tmp_mask].T)
    
    # Scale and center
    ax1.plot(xvals[::ds], yvals[::ds], '.', markersize=1)
    ax1.plot(np.linspace(0,np.max(xvals[::ds])), np.linspace(0,np.max(yvals[::ds])), 'r')
    ax1.set_title('scale={:.3} r={:.3} m={:.3}'.format(scale, r_value, slope))
    ax1.set_xlabel('pupil camera dist')
    ax1.set_ylabel('scale * ellipticity')

    # Calibration of camera center
    delta = (cam_center - ellipse[['X0_in','Y0_in']].copy().to_numpy().T)
    show_cal = cal_pupil[::ds]
    show_use = use_pupil[::ds]
    ang2x = ellipse['angle_to_x'].copy().to_numpy()
    # Plot pts used for calibration
    ax2.plot(
        np.linalg.norm(delta[:,show_cal], 2, axis=0),  \
        ((delta[0,show_cal].T * np.cos(ang2x[show_cal]))  \
        + (delta[1,show_cal].T * np.sin(ang2x[show_cal])))  \
        / np.linalg.norm(delta[:,show_cal], 2, axis=0).T,  \
        'r.', markersize=1, label='cal')
    # Plot all pts
    ax2.plot(
        np.linalg.norm(delta[:,show_use], 2, axis=0),  \
        ((delta[0,show_use].T * np.cos(ang2x[show_use]))  \
        + (delta[1,show_use].T * np.sin(ang2x[show_use])))  \
        / np.linalg.norm(delta[:, show_use], 2, axis=0).T,  \
        'k.', markersize=1, label='all')
    ax2.set_title('camera center calibration')
    ax2.set_ylabel('abs([PC-EC]).[cosw;sinw]')
    ax2.set_xlabel('abs(PC-EC)')

    ax3.axis('off')

    fig.tight_layout()
    pdf.savefig()
    plt.close()

    pdf.close() # close and save the pdf

    # save out camera center and scale as np array
    # (but only if this is a freely moving recording)
    if 'fm' in cfg['_rname']:
        caldict = {
            'cam_cent_x': float(cam_center[0]),
            'cam_cent_y': float(cam_center[1]),
            'scale': float(scale),
            'regression_r': float(r_value),
            'regression_m': float(slope)
            }
        
        # Save the calibration parameters as a .json, but also return it
        # in the output data.
        caldict_savepath = os.path.join(cfg['_rpath'],
                '{}_{}_eyeparams.json'.format(cfg['rfname'], cfg['_cam']))

        with open(caldict_savepath, 'w') as f:
            json.dump(caldict, f)
    else:
        caldict = {}

    # save out the data
    ell_dict = {
        'theta': list(theta),
        'phi': list(phi),
        'longaxis': list(ellipse['long_axis'].values),
        'shortaxis': list(ellipse['short_axis'].values),
        'X0': list(ellipse['X0_in'].values),
        'Y0': list(ellipse['Y0_in'].values),
        'ellipse_rotation': list(ellipse['angle_to_x'].values),
        'camera_center': cam_center[:,0] # contains X and Y in zeroth dim
    }
    pos_dict = all_dlc_data.to_dict('list')

    return_dict = {
        'ellipse_data': ell_dict,
        'position_data': pos_dict,
        'pupil_calibration_self': caldict,
        'pupil_calibration_used': eyeparams
    }

    return return_dict
