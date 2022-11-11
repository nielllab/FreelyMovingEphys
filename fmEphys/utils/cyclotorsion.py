
def sigmoid_curve(xval, a, b, c):
    """Sigmoid curve function."""

    return a + (b-a) / (1 + 10**( (c - xval) * 2))


def sigmoid_fit(d):
    """ Fit sigmoid.
    popt: fit
    ci: confidence interval
    """
    try:
        popt, pcov = optimize.curve_fit(sigmoid_curve,
                        xdata=range(1,len(d)+1),
                        ydata=d,
                        p0=[100.0,200.0,len(d)/2],
                        method='lm',
                        xtol=10**-3,
                        ftol=10**-3)
        ci = np.sqrt(np.diagonal(pcov))

    except RuntimeError:
        popt = np.nan*np.zeros(4)
        ci = np.nan*np.zeros(4)

    return (popt, ci)


def get_torsion_from_ridges(cfg, ell_dict, vidpath=None):
    """ Get torsion (omega) from rotation of ridges along the edge of the pupil.
    """

    if vidpath is None:
        vidpath = utils.path.find('{}*{}deinter.avi'.format(cfg['rfname'],  \
            cfg['dname']), cfg['rpath'])
        vidpath = utils.path.most_recent(vidpath)
    
    pdf_savepath = os.path.join(cfg['rpath'],
            '{}_{}_cyclotorsion.pdf'.format(cfg['rname'], cfg['cname']))
    pdf = PdfPages(pdf_savepath)
    
    # Set up range of degrees in radians
    rad_range = np.deg2rad(np.arange(360))

    # Get the ellipse parameters for this trial from the time-interpolated xarray
    eye_longaxis = ell_dict['longaxis']
    eye_shortaxis = ell_dict['shortaxis']
    eye_centX = ell_dict['X0']
    eye_centY = ell_dict['Y0']

    # Set up for the read-in video
    eyevid = cv2.VideoCapture(self.video_path)
    # this can be changed to a small number of frames for testing
    totalF = int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT))

    set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)),  \
                int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # set up for the multiprocessing for sigmoid fit
    n_proc = multiprocessing.cpu_count()
    print('Found {} as CPU count for multiprocessing'.format(n_proc))
    pool = multiprocessing.Pool(processes=n_proc)

    range_r = 10

    print('Calculating pupil cross-section and fitting sigmoid (slow)')
    errCount = 0

    rfit_out = np.zeros(totalF, 360)
    rfit_conv_out = np.zeros(totalF, 360)

    for f in tqdm(np.arange(totalF)):
        try:
            # Read frame
            ret, img = eyevid.read()
            if not ret:
                break

            # Convert to grey image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Mean radius
            meanr = 0.5 * (eye_longaxis[f] + eye_shortaxis[f])
            # Range of values over mean radius (meanr)
            r = range(int(meanr - range_r), int(meanr + range_r))
            # Empty array that the calculated edge of the pupil
            # will be put into
            pupil_edge = np.zeros([360, len(r)])

            # Get cross-section of pupil at each angle 1-360 and
            # fit to sigmoid
            rad_range = np.deg2rad(np.arange(360))
            
            for i in range(len(r)):
                pupil_edge[:,i] = img[(  \
                    (eye_centY[f] + r[i] * (np.sin(rad_range))).astype(int),  \
                    (eye_centY[f] + r[i] * (np.cos(rad_range))).astype(int)  \
                    )]

            d = pupil_edge[:,:]

            # Apply sigmoid fit with multiprocessing
            param_mp = [pool.apply_async(sigmoid_fit, args=(d[n,:],)) for n in range(360)]
            params_output = [result.get() for result in param_mp]

            # Unpack outputs of sigmoid fit
            params = []; ci = []
            for vals in params_output:
                params.append(vals[0])
                ci.append(vals[1])
            params = np.stack(params)
            ci = np.stack(ci)

            # Extract radius variable from parameters
            rfit = params[:,2] - 1

            # If confidence interval in estimate is > fit_thresh pix, set to to NaN
            ci_temp = (ci[:,0] > 5) | (ci[:,1] > 5)  | (ci[:,2]>0.75)
            rfit[ci_temp] = np.nan

            # Remove if luminance goes the wrong way (e.g. from reflectance)
            rfit[(params[:,1] - params[:,0]) < 10] = np.nan
            rfit[params[:,1] > 250] = np.nan

            try:
                # Median filter
                rfit_filt = utils.filter.nanmedfilt(rfit, 5)

                # Subtract baseline because our points aren't perfectly centered on ellipse
                filtsize = 31
                rfit_conv = rfit_filt - astropy.convolution.convolve(rfit_filt,  \
                                np.ones(filtsize)/filtsize, boundary='wrap')

            except ValueError as e:
                # In case every value in rfit is NaN
                rfit = np.nan*np.zeros(360)
                rfit_conv = np.nan*np.zeros(360)
                
        except (KeyError, ValueError):
            errCount += 1
            rfit = np.nan*np.zeros(360)
            rfit_conv = np.nan*np.zeros(360)

        # Get rid of outlier points
        rfit_conv[np.abs(rfit_conv) > 1.5] = np.nan

        # Save this out
        rfit_out[f,:] = rfit
        rfit_conv_out[f,:] = rfit_conv

    ##############

    # Save out pupil edge data
    edgedata_dict = {
        'rfit': rfit,
        'rfit_conv': rfit_conv
    }

    # Threshold out any frames with large or small rfit_conv distributions
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

    ### ***

        # calculate mean as template
        try:
            template_rfitconv_cc, template_rfit_cc_lags = nanxcorr(rfit_conv_xr[7].values, template, 30)
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
            rfit2times_cc, rfit2times_lags = nanxcorr(rfit_conv_xr.isel(frame=t0).values, rfit_conv_xr.isel(frame=t1).values, 10)
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
                    xc, lags = nanxcorr(template, pupil_update[frame_num,:], 20)
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
        shift_smooth = signal.medfilt(shift_nan,3)  # median filt to get rid of outliers
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


        if self.config['internals']['apply_gamma_to_eyecam']:
            self.auto_contrast()

        if self.config['main']['pose_estimation']:
            self.pose_estimation()
