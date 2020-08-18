"""
FreelyMovingEphys jump tracking utilities
analyze_jump.py

Last modified August 17, 2020
"""

def jump_cc(global_data_path, trial_name, REye_ds, LEye_ds, top_ds, save_path):

    while(1):
        side_ret, side_frame = sidevid.read()

        if not side_ret:
            break

        REye_now = REye.sel(frame=framenow)
        LEye_now = LEye.sel(frame=framenow)

        RTheta = REye_now.sel(ellipse_param='theta')
        RPhi = REye_now.sel(ellipse_param='phi')
        LTheta = LEye_now.sel(ellipse_param='theta')
        LPhi = LEye_now.sel(ellipse_param='phi')

def jump_tracking(datapath, savepath, trialname, REye, LEye):
    # setup the file to save out of this

    savepath = str(savepath) + '/' + str(trial_name) + '/' + str(trial_name) + '_' + vext + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (width, height))

    # set colors
    plot_color0 = (225, 255, 0)
    plot_color1 = (0, 255, 255)

    while(1):
        # read in videos
        SIDE_ret, SIDE_frame = sidevid.read()
        TOP_ret, TOP_frame = sidevid.read()
        REye_ret, REye_frame = sidevid.read()
        LEye_ret, LEye_frame = sidevid.read()

        if not SIDE_ret:
            break
        if not TOP_ret:
            break
        if not REye_ret:
            break
        if not LEye_ret:
            break

        # get current ellipse parameters
        REye_now = REye.sel(frame=framenow)
        LEye_now = LEye.sel(frame=framenow)

        # split apart parameters
        RTheta = REye_now.sel(ellipse_param='theta')
        RPhi = REye_now.sel(ellipse_param='phi')
        LTheta = LEye_now.sel(ellipse_param='theta')
        LPhi = LEye_now.sel(ellipse_param='phi')

        # scale factor Matlab code uses
        # R = R*Data(vid).scaleR/50;
        # L = L*Data(vid).scaleL/50;
