"""
FreelyMovingEphys topdown cricket tracking utilities
track_cricket.py

Last modified August 23, 2020
"""

# package imports
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

def get_cricket_props(top_pts, mouse_theta, savepath, trial_name):
    # make save directory if it doesn't already exist
    fig_dir = savepath + '/' + trial_name + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # speed
    vx_c = np.diff(top_pts.sel(point_loc='cricket_Body_x').values)
    vy_c = np.diff(top_pts.sel(point_loc='cricket_Body_y').values)
    filt = np.ones([3]) / np.sum(np.ones([3]))
    vx_c = np.convolve(vx_c, filt, mode='same')
    vy_c = np.convolve(vx_c, filt, mode='same')
    cricket_speed = np.sqrt(vx_c**2, vy_c**2)

    # cricket range
    rx = top_pts.sel(point_loc='cricket_Body_x').values - top_pts.sel(point_loc='nose_x').values
    ry = top_pts.sel(point_loc='cricket_Body_y').values - top_pts.sel(point_loc='nose_y').values
    c_range = np.sqrt(rx**2, ry**2)

    # azimuth
    cricket_theta = np.arctan2(ry,rx)
    az = mouse_theta - cricket_theta

    # head angular velocity
    d_theta = np.diff(mouse_theta.values)
    d_theta = np.where(d_theta > np.pi, d_theta+2*np.pi, d_theta-2*np.pi)
    theta_fract = np.sum(~pd.isnull(mouse_theta.values))/len(mouse_theta.values)
    # long_theta_fract = np.sum(~pd.isnull(mouse_theta['mean_head_theta'].values))/len(mouse_theta['mean_head_theta'].values)

    # head velocity
    vx_m = np.diff(top_pts.sel(point_loc='nose_x').values) # currently use nose x/y -- is this different if we use the center of the head?
    vy_m = np.diff(top_pts.sel(point_loc='nose_x').values)
    vx_m = np.convolve(vx_m, filt, mode='same')
    vy_m = np.convolve(vy_m, filt, mode='same')
    mouse_speed = np.sqrt(vx_m**2, vy_m**2)

    # a very large plot of the cricket and mouse properties
    plt.subplots(2,3)
    plt.subplot(231)
    plt.plot(cricket_speed)
    plt.xlabel('frames')
    plt.ylabel('pixels/sec')
    plt.title('cricket speed')
    plt.subplot(232)
    plt.plot(c_range)
    plt.xlabel('frame')
    plt.ylabel('pixels')
    plt.title('range (cricket body to mouse nose)')
    plt.subplot(233)
    plt.plot(az)
    plt.xlabel('frame')
    plt.ylabel('radians')
    plt.title('azimuth')
    plt.subplot(234)
    plt.plot(d_theta)
    plt.xlabel('frame')
    plt.ylabel('radians/frame')
    plt.title('head angular velocity')
    plt.subplot(235)
    plt.plot(mouse_speed)
    plt.xlabel('frame')
    plt.ylabel('pixels/sec')
    plt.title('mouse speed')
    plt.savefig(fig_dir + 'cricket_props.png', dpi=300)
    plt.close()

    # print('data lengths: cspeed=' + str(len(cricket_speed)) + ' range=' + str(len(range)) + ' az=' + str(len(az)) + ' dtheta=' + str(len(d_theta)) + ' mspeed=' + str(len(mouse_speed)))
    props_out = pd.DataFrame({'cricket_speed':list(cricket_speed), 'range':list(c_range)[:-1], 'azimuth':list(az)[:-1], 'd_theta':list(d_theta), 'mouse_speed':list(mouse_speed)})
    # TO DO: Sort out why range and az end up longer than the rest of the data, and fix it. Reliably one value too long. Index to get it running for now
    prop_names = ['cricket_speed', 'range', 'azimuth', 'd_theta', 'mouse_speed']
    cricket_props = xr.DataArray(props_out, coords=[('frame',range(0,np.size(cricket_speed,0))), ('prop',prop_names)])

    return cricket_props
