"""body.py

Calculate movement and position properties of freely moving mice using
position data previously calculated.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys


def filter_likelihood(position_data, thresh=0.99):
    """Filter position data using the likleihood values.

    Parameters
    ----------
    position_data : dict
        Position data for the top cam.
    thresh : float, optional
        Threshold value (between 0 and 1), by default 0.99.

    Returns
    -------
    filt_positions : dict
        Position data, where timepoints below threshold are replaced with np.nan.
    useF : np.array
        Array with the shape (position, time) as bool values indicating every frame
        where a position was above or below the likleihood threshold.
    """

    # Get the column names
    pt_names = list(position_data.keys())
    x_cols = [i for i in pt_names if '_x' in i]
    y_cols = [i for i in pt_names if '_y' in i]
    l_cols = [i for i in pt_names if '_likelihood' in i]

    filt_positions = position_data.copy()
    useF = np.ones(len(x_cols),
                   len(position_data.index.values)).astype(int)

    # Itereate through the number of columns. Here, each colun is a position
    # label (e.g., nose_x).
    for i in range(len(x_cols)):
        
        # Get the data for the current label
        x = position_data[x_cols[i]]
        y = position_data[y_cols[i]]
        l = position_data[l_cols[i]]

        # Set x/y data to NaN if the likelihood is low
        x[l < thresh] = np.nan
        y[l < thresh] = np.nan
        useF[i, l < thresh] = 0

        filt_positions[x_cols[i]] = x.copy()
        filt_positions[y_cols[i]] = y.copy()

    return filt_positions, useF


def wide_smooth(x):
    """Helper function to median filter data and smooth it with a wide box
    convolution.

    Parameters
    ----------
    x : np.array
        1D array.

    Returns
    -------
    np.array
        Smoothed data.
    """
    return fmEphys.convfilt(fmEphys.nanmedfilt(x, 7).squeeze(), box_pts=20)


def narrow_smooth(x):
    """Helper function to median filter data and smooth it with a narrow box
    convolution.

    Parameters
    ----------
    x : np.array
        1D array.

    Returns
    -------
    np.array
        Smoothed data.
    """
    return fmEphys.convfilt(fmEphys.nanmedfilt(x, 7).squeeze())


def calc_speed(x, y, pxls2cm=1, fps=60):
    """Calculate speed from x/y coordinates of a single position.

    It is important to smooth position data before calculating speed with this
    function. Without smoothing, directionless jitter of positions from pose
    tracking would be appear as frame-to-frame locomotion.

    Without a conversion factor for video pixels to centimters, the speed will
    be returned in units of pxls/cm. If the

    Parameters
    ----------
    x : np.array
        x position or a single tracked point.
    y : np.array
        y position or a single tracked point.
    pxls2cm : int or float, optional
        Conversion factor from video pixels to centimeters. Should be a small
        float, by default 1.
    fps : int, optional
        Sample rate in units of frames per second, by default 60.

    Returns
    -------
    np.array
        Speed.
    """

    return np.sqrt(np.diff((x*fps) / pxls2cm)**2 + np.diff((y*fps) / pxls2cm)**2)

def calc_yaw(left_y, right_y, left_x, right_x, rotate=False):
    """Calculate yaw using four positions.

    If you are measuring head angle and using the two ears, set `rotate` to True
    and add 90 degrees so that you measure yaw along the correct axis.

    Parameters
    ----------
    left_y : np.aray
        1D position array.
    right_y : np.aray
        1D position array.
    left_x : np.aray
        1D position array.
    right_x : np.aray
        1D position array.
    rotate : bool, optional
        If the points are perpendicular to the angle you want to measure, you
        can rotate the angle by 90 degrees in the positive direction, by default
        False.

    Returns
    -------
    yaw_rad : np.array
        1D arrya of yaw over time in units of radians.
    yaw_deg : np.array
        1D arrya of yaw over time in units of degrees.
    """

    yaw_rad = np.arctan2((left_y - right_y), (left_x - right_x))

    if rotate is True:
        yaw_rad = yaw_rad + np.deg2rad(90)

    yaw_deg = np.rad2deg(yaw_rad % (2*np.pi))

    return yaw_rad, yaw_deg

def track_body(cfg, rpath=None):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    cfg : _type_
        _description_
    rpath : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """

    cfg = fmEphys.get_cfg(cfg)

    if rpath is not None:
        cfg['rpath'] = rpath

    if ('rname' not in cfg.keys()) or (cfg['rname'] is None):
        cfg['rname'] = fmEphys.get_rec_name(cfg['rpath'])

    pdf_savepath = os.path.join(cfg['rpath'],
            '{}_{}_diagnostics.pdf'.format(cfg['rname'], cfg['camname']))
    pdf = PdfPages(pdf_savepath)

    # Timestamps
    raw_camT_path = fmEphys.find('*{}*{}*BonsaiTS.csv'.format(cfg['rname'], cfg['camname']), cfg['rpath'], mr=True)
    topT = fmEphys.read_time(raw_camT_path)
    topT = topT - topT[0]

    # Position data
    dlc_path = fmEphys.find('*{}*DLC*.h5'.format(cfg['camname']), cfg['rpath'], mr=True)
    position_data = fmEphys.read_DLC_data(dlc_path)
    position_data, useF = filter_likelihood(cfg['lik_thresh'])

    x_pos, y_pos, l_vals = fmEphys.split_xyl(position_data)
    x_cols = x_pos.columns

    _names = []
    for i, _x_name in enumerate(x_cols):
        _names.append(_x_name[:-2])

    fracs_good = np.mean(useF, axis=1)

    # Plastic or metal corners?
    ctypes = ['m','p']
    frac_good = np.zeros(2)
    for ctypenum in range(2):
        ctype = ctypes[ctypenum]
        fg = 0
        for cpos in ['tl','tr','br','bl']:
            cname = cpos+'_'+ctype+'_corner_'
            x = x_pos[cname+'x']
            y = y_pos[cname+'y']
            fg += np.sum(~np.isnan(x) * ~np.isnan(y)) / len(x)
        frac_good[ctypenum] = fg
    metal_corners = np.argmax(frac_good)==0

    # Define cm using top or bottom corners?
    frac_good = np.zeros(2)
    m_or_p = ('m' if metal_corners else 'p')
    for c, poslet in enumerate(['t','b']):
        fg = 0
        for lrpos in ['l','r']:
            cname = poslet+lrpos+'_'+m_or_p+'_corner_'
            x = x_pos[cname+'x']
            y = y_pos[cname+'y']
            fg += np.sum(~np.isnan(x) * ~np.isnan(y)) / len(x)
        frac_good[c] = fg
    use_top_for_dist = np.argmax(frac_good)==0

    # conversion from pixels to cm
    tb = ('t' if use_top_for_dist else 'b')
    mp = ('m' if metal_corners else 'p')
    left = tb+'l_'+mp+'_corner_x'
    right = tb+'r_'+mp+'_corner_x'
    dist_pxls = np.nanmedian(x_pos[right]) - np.nanmedian(x_pos[left])
    pxls2cm = dist_pxls / cfg['top_arena_cm']
    if np.isnan(pxls2cm):
        pxls2cm = cfg['top_pxls2cm']

    ### Speed from neck point

    # Need to smooth position data w/ convlutional filter and median filter
    # otherwise, a small frame-to-frame jitter will be added to locomotion speed
    smooth_x = wide_smooth(position_data['center_neck_x'])
    smooth_y = wide_smooth(position_data['center_neck_y'])
    speed = calc_speed(smooth_x, smooth_y, pxls2cm=pxls2cm)
    # speeds above 25 cm/sec are not reasonable
    speed[speed>25] = np.nan

    ### Get head and body yaw

    # get head angle from ear points
    lear_x = narrow_smooth(position_data['left_ear_x'])
    lear_y = narrow_smooth(position_data['left_ear_y'])
    rear_x = narrow_smooth(position_data['right_ear_x'])
    rear_y = narrow_smooth(position_data['right_ear_y'])
    # Get angle
    # rotate 90deg because ears are perpendicular to head yaw
    head_yaw_rad, head_yaw_deg = calc_yaw(lear_y, rear_y, lear_x, rear_x,
                                          rotate=True)

    # body angle from neck and back points
    neck_x = narrow_smooth(position_data['center_neck_x'])
    neck_y = narrow_smooth(position_data['center_neck_y'])
    back_x = narrow_smooth(position_data['center_haunch_x'])
    back_y = narrow_smooth(position_data['center_haunch_y'])
    # rotate 90deg because ears are perpendicular to head yaw
    body_yaw_rad, body_yaw_deg = calc_yaw(neck_y, back_y, neck_x, back_x,
                                          rotate=True)

    # The difference betwen head and body yaw tells us if the head and body are
    # pointed in the same direction.
    body_head_diff = body_yaw_rad - head_yaw_rad
    body_head_diff[body_head_diff < -np.deg2rad(120)] = np.nan
    body_head_diff[body_head_diff > np.deg2rad(120)] = np.nan

    # Angle of body movement ("movement yaw")
    x_disp = np.diff((smooth_x*60) / pxls2cm)
    y_disp = np.diff((smooth_y*60) / pxls2cm)
    movement_yaw_rad = np.arctan2(y_disp, x_disp)
    movement_yaw_deg = np.rad2deg(movement_yaw_rad % (2*np.pi))

    ### Definitions of state

    frd_deg = cfg['forward_thresh'] # deg
    thresh_spd = cfg['running_thresh'] # cm/sec

    # How aligned are the body angle and the direction of locomotion?
    aligned = movement_yaw_rad - body_yaw_rad[:-1]

    is_forward = (np.abs(aligned) < np.deg2rad(frd_deg))
    is_backward = (np.abs(aligned + np.deg2rad(180) % (2*np.pi)) < np.deg2rad(frd_deg))
    is_running = (speed > thresh_spd)
    
    is_forward_run = is_running * is_forward
    is_backward_run = is_running * is_backward
    is_fine_motion = is_running * ~is_forward * ~is_backward
    is_stationary = ~is_running

    ### Diagnostic plots
    
    fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(2, 2, figsize=(11,8.5), dpi=300)
    
    ax0.bar(range(len(_names)), fracs_good, width=0.95)
    ax0.set_xticklabels(_names, rotatate=90)
    ax0.set_ylabel('fraction good frames')

    ax1.plot(topT[:3600], speed[:3600], linewidth=1, color='k')
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('speed (cm/sec)')
    ax1.set_xticks(np.linspace(0,60,6).astype(int))

    ax2.hist(speed, bins=40, density=True, color='k')
    ax2.set_xlabel('speed (cm/sec)')
    ax2.set_ylabel('fraction frames')

    ax3.plot(topT[:7200], head_yaw_deg[:7200], '.', markersize=1, color='k')
    ax3.set_ylabel('head yaw (deg)')
    ax3.set_xlabel('time (sec)')
    ax3.set_xticks(np.linspace(0,120,6).astype(int))

    fig.tight_layout()
    pdf.savefig()
    plt.close()

    fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(2, 2, figsize=(11,8.5), dpi=300)

    ax0.plot(topT[:7200], body_yaw_deg[:7200], '.', markersize=1, color='k')
    ax0.set_ylabel('body yaw (deg)')
    ax0.set_xlabel('time (sec)')
    ax0.set_xticks(np.linspace(0,120,6).astype(int))

    ax1.plot(topT[:7200], np.rad2deg(body_head_diff[:7200]), '.', markersize=1, color='k')
    ax1.set_ylabel('head yaw - body yaw (deg)')
    ax1.set_xlabel('time (sec)')
    ax1.set_xticks(np.linspace(0,120,6).astype(int))

    ax2.plot(topT[:7200], movement_yaw_deg[:7200], '.', markersize=1, color='k')
    ax2.set_ylabel('animal yaw (deg)')
    ax2.set_xlabel('time (sec)')

    ax3.plot(topT[:7200], (movement_yaw_deg - body_yaw_deg[:-1])[:7200], '.', markersize=1)
    ax3.set_ylabel('movement yaw - body yaw (deg)')
    ax3.set_xlabel('sec')
    ax3.set_xticks(np.linspace(0,120,6).astype(int))

    fig.tight_layout()
    pdf.savefig()
    plt.close()

    ### Diagnostic figures

    fig, [ax0,ax1] = plt.subplots(1, 2, figsize=(11,8.5), dpi=300)

    startF = 1000
    maxF = 3600
    cmap = plt.cm.jet(np.linspace(0,1,maxF))

    # plot arena borders
    xbounds = np.zeros(5)
    ybounds = np.zeros(5)
    for _i, _name in enumerate(['tl','tr','br','bl','tl']):
        
        xbounds[_i] = np.nanmedian(position_data['{}_{}_corner_x'.format(_name, mp)])
        ybounds[_i] = np.nanmedian(position_data['{}_{}_corner_y'.format(_name, mp)])

    ax0.plot(xbounds, ybounds, 'k-')


    ### New panel in the figure

    # Plot all necl positions.
    for f in range(startF, startF+maxF):

        ax1.plot(neck_x[f], neck_y[f], '.', color=cmap[f-startF])

    # Plot the arena corners.
    ax1.plot(xbounds, ybounds, 'k-')

    # Bint he speed data so that we can easily apply a colormap.
    speed_bins = np.linspace(0, int(np.ceil(np.nanmax(speed) / 3)))
    
    spdcolors = plt.cm.magma(speed_bins)

    for f in np.arange(startF, startF+maxF+10, 10):

        # Color the arrow by the speed of the animal using the magma colormap.
        if ~np.isnan(speed[f]):
            usecolor = spdcolors[np.argmin(np.abs(speed[f] - speed_bins))]
        else:
            continue

        # positions
        x0 = neck_x[f]
        y0 = neck_y[f]
        # displacement
        dX = 15 * np.cos(head_yaw_rad[f])
        dY = 15 * np.sin(head_yaw_rad[f])
        
        # draw an arrow showing the vector of movement
        ax1.arrow(x0, y0, dX, dY, facecolor=usecolor, width=7, edgecolor='k')

    fig.tight_layout()
    pdf.savefig()
    plt.close()

    # Close the pdf and write the file.
    pdf.close()

    # Collect movement data into a dictionary.
    movement_data = {
        'speed': speed,
        'head_yaw_rad': head_yaw_rad,
        'body_yaw_rad': body_yaw_rad,
        'body_head_diff': body_head_diff,
        'movement_yaw': movement_yaw_deg,
        'head_body_alignment': aligned,
        'is_forward_run': is_forward_run,
        'is_backward_run': is_backward_run,
        'is_fine_motion': is_fine_motion,
        'is_stationary': is_stationary
    }

    # Join the movment data with other data from this camera.
    return_dict = {
        'positions': position_data,
        'time': topT,
        'movement': movement_data
    }

    return return_dict

