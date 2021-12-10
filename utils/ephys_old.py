def make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units,
               this_unit, eyeInterp, worldInterp, top_vid, topT, topInterp, th, phi, top_speed, accT=None, gz=None, speedT=None, spd=None):
    """
    make a video (without sound)
    INPUTS
        file_dict: dict of files and options from function file_files
        eyeT: timestamps for eyecam
        worldT: timestamps for worldcam
        eye_vid: eyecam video as array
        worldvid: worldcam viedo as array
        contrast: contrast over time
        eye_params: xarray of eye parameters (e.g. theta, phi, X0, etc.)
        dEye: eye velocity over time
        goodcells: ephys dataframe
        units: indexes of all units
        this_unit: unit number to highlight
        eyeInterp: interpolator for eye video
        worldInterp: interpolator for world video
        accT: imu timestamps, if this is a freely moving recording (not plotted if it's not provided)
        gz: gyro z-axis (not plotted if it's not provided)
        speedT: ball optical mouse timestamps (used in place of accT if headfixed)
        spd: ball optical mouse speed (used in place of gz if headfixed)
    OUTPUTS
        vidfile: filepath to the generated video, which is saved out by the function
    """
    # set up figure
    fig = plt.figure(figsize = (10,16))
    gs = fig.add_gridspec(12,6)
    if top_vid is not None:
        axEye = fig.add_subplot(gs[0:2,0:2])
        axWorld = fig.add_subplot(gs[0:2,2:4])
        axTopdown = fig.add_subplot(gs[0:2,4:6])
    else:
        axEye = fig.add_subplot(gs[0:2,0:3])
        axWorld = fig.add_subplot(gs[0:2,3:6])
    axRad = fig.add_subplot(gs[2,:])
    axTh = fig.add_subplot(gs[3,:])
    if top_vid is not None:
        axGyro = fig.add_subplot(gs[4,:])
        axR = fig.add_subplot(gs[5:12,:])
    else:
        axR = fig.add_subplot(gs[4:12,:])

    # timerange and center frame (only)
    tr = [7, 7+15]
    fr = np.mean(tr) # time for frame
    eyeFr = np.abs(eyeT-fr).argmin(dim = "frame")
    worldFr = np.abs(worldT-fr).argmin(dim = "frame")
    if top_vid is not None:
        topFr = np.abs(topT-fr).argmin(dim = "frame")

    axEye.cla(); axEye.axis('off')
    axEye.imshow(eye_vid[eyeFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")

    axWorld.cla();  axWorld.axis('off'); 
    axWorld.imshow(world_vid[worldFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    
    if top_vid is not None:
        axTopdown.cla();  axTopdown.axis('off'); 
        axTopdown.imshow(top_vid[topFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    
    axTh.cla()
    axTh.plot(eyeT,th)
    axTh.set_xlim(tr[0],tr[1]); 
    axTh.set_ylabel('theta (deg)')#; axTh.set_ylim(-50,0)

    axRad.cla()
    axRad.plot(eyeT,eye_params.sel(ellipse_params='longaxis'))
    axRad.set_xlim(tr[0],tr[1])
    axRad.set_ylabel('pupil radius')
    
    if top_vid is not None:
        # plot gyro
        axGyro.plot(accT,gz)
        axGyro.set_xlim(tr[0],tr[1]); axGyro.set_ylim(-500,500)
        axGyro.set_ylabel('gyro z (deg/s)')    
  
    # plot spikes
    axR.fontsize = 20
    probe = file_dict['probe_name']
    if '64' in probe:
        sh_num = 2
    elif '128' in probe:
        sh_num = 4
    sh0 = np.arange(0,len(goodcells.index)+sh_num,sh_num)
    full_raster = np.array([]).astype(int)
    for sh in range(sh_num):
        full_raster = np.concatenate([full_raster, sh0+sh])
    for i, ind in enumerate(goodcells.index):
        i = full_raster[i]
        axR.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25,'k',linewidth=0.5)
    axR.vlines(goodcells.at[units[this_unit],'spikeT'], full_raster[this_unit+1]-0.25, full_raster[this_unit+1]+0.25,'b',linewidth=0.5) # this unit
    
    n_units = len(goodcells)
    axR.set_ylim(n_units,-.5)
    axR.set_xlim(tr[0],tr[1]); axR.set_xlabel('secs'); axR.set_ylabel('unit')
    axR.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)
    plt.tight_layout()

    vidfile = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'.mp4'))

    # animate
    writer = FFMpegWriter(fps=30, extra_args=['-vf','scale=800:-2'])
    with writer.saving(fig, vidfile, dpi=100):
        for t in np.arange(tr[0],tr[1],1/30):
            # show eye and world frames
            axEye.cla(); axEye.axis('off')
            axEye.imshow(eyeInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            axWorld.cla(); axWorld.axis('off'); 
            axWorld.imshow(worldInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            if top_vid is not None:
                axTopdown.cla(); axTopdown.axis('off')
                axTopdown.imshow(topInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            # plot line for time, then remove
            ln = axR.vlines(t,-0.5,n_units,'b')
            writer.grab_frame()
            ln.remove()
    return vidfile

def run_ephys_analysis(file_dict):
    
    # create interpolator for movie data so we can evaluate at same timebins are firing rate
    # img_norm[img_norm<-2] = -2
    sz = np.shape(img_norm); downsamp = 0.5
    img_norm_sm = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)))
    for f in range(sz[0]):
        img_norm_sm[f,:,:] = cv2.resize(img_norm[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
    movInterp = interp1d(worldT, img_norm_sm, axis=0, bounds_error=False)
    # get channel number
    if '16' in file_dict['probe_name']:
        ch_num = 16
    elif '64' in file_dict['probe_name']:
        ch_num = 64
    elif '128' in file_dict['probe_name']:
        ch_num = 128
    print('getting STA for single lag')
    # calculate spike-triggered average
    staAll, STA_single_lag_fig = plot_STA(goodcells, img_norm_sm, worldT, movInterp, ch_num, lag=2)
    detail_pdf.savefig()
    plt.close()
    print('getting STA for range in lags')
    # calculate spike-triggered average
    fig = plot_STA(goodcells, img_norm_sm, worldT, movInterp, ch_num, lag=np.arange(-2,8,2))
    detail_pdf.savefig()
    plt.close()
    print('getting STV')
    # calculate spike-triggered variance
    st_var, fig = plot_STV(goodcells, movInterp, img_norm_sm, worldT)
    detail_pdf.savefig()
    plt.close()

    if (free_move is True and file_dict['stim_type']=='light_arena') | (file_dict['stim_type'] == 'white_noise'):
        print('doing GLM receptive field estimate')
        # simplified setup for GLM
        # these are general parameters (spike rates, eye position)
        n_units = len(goodcells)
        print('get timing')
        model_dt = 0.025
        model_t = np.arange(0,np.max(worldT),model_dt)
        model_nsp = np.zeros((n_units,len(model_t)))
        # get spikes / rate
        print('get spikes')
        bins = np.append(model_t,model_t[-1]+model_dt)
        for i,ind in enumerate(goodcells.index):
            model_nsp[i,:],bins = np.histogram(goodcells.at[ind,'spikeT'],bins)
        # get eye position
        print('get eye')
        thInterp = interp1d(eyeT,th, bounds_error = False)
        phiInterp =interp1d(eyeT,phi, bounds_error = False)
        model_th = thInterp(model_t+model_dt/2)
        model_phi = phiInterp(model_t+model_dt/2)
        del thInterp, phiInterp
        # get active times
        if free_move:
            interp = interp1d(accT,(gz-np.mean(gz))*7.5,bounds_error=False)
            model_gz = interp(model_t)
            model_active = np.convolve(np.abs(model_gz),np.ones(np.int(1/model_dt)),'same')
            use = np.where((np.abs(model_th)<10) & (np.abs(model_phi)<10)& (model_active>40) )[0]
        else:
            use = np.array([True for i in range(len(model_th))])
        # get video ready for GLM
        print('setting up video')
        downsamp = 0.25
        testimg = img_norm[0,:,:]
        testimg = cv2.resize(testimg,(int(np.shape(testimg)[1]*downsamp), int(np.shape(testimg)[0]*downsamp)))
        testimg = testimg[5:-5,5:-5]; # remove area affected by eye movement correction
        resize_img_norm = np.zeros([np.size(img_norm,0), np.int(np.shape(testimg)[0]*np.shape(testimg)[1])])
        for i in tqdm(range(np.size(img_norm,0))):
            smallvid = cv2.resize(img_norm[i,:,:], (np.int(np.shape(img_norm)[2]*downsamp), np.int(np.shape(img_norm)[1]*downsamp)), interpolation=cv2.INTER_LINEAR_EXACT)
            smallvid = smallvid[5:-5,5:-5]
            resize_img_norm[i,:] = np.reshape(smallvid,np.shape(smallvid)[0]*np.shape(smallvid)[1])
        movInterp = interp1d(worldT, resize_img_norm, 'nearest', axis=0, bounds_error=False)
        model_vid_sm = movInterp(model_t)
        nks = np.shape(smallvid); nk = nks[0]*nks[1]
        model_vid_sm[np.isnan(model_vid_sm)]=0
        del movInterp
        gc.collect()
        glm_receptive_field, glm_cc, fig = fit_glm_vid(model_vid_sm,model_nsp,model_dt, use,nks)
        detail_pdf.savefig()
        plt.close()
        del model_vid_sm
        gc.collect()
    elif free_move is True and file_dict['stim_type']=='dark_arena':
        print('skipping GLM RFs; still getting active times')
        n_units = len(goodcells)
        model_dt = 0.025
        model_t = np.arange(0,np.max(worldT),model_dt)
        model_nsp = np.zeros((n_units,len(model_t)))
        interp = interp1d(accT,(gz-np.mean(gz))*7.5,bounds_error=False)
        model_gz = interp(model_t)
        model_active = np.convolve(np.abs(model_gz),np.ones(np.int(1/model_dt)),'same')

    print('plotting head and eye movements')
    # calculate saccade-locked psth
    spike_corr = 1 # correction factor for ephys timing drift
    plt.figure()
    plt.hist(dEye, bins=21, range=(-10,10), density=True)
    plt.xlabel('eye dtheta'); plt.ylabel('fraction')
    detail_pdf.savefig()
    plt.close()
    if free_move is True:
        dhead = interp1d(accT,(gz-np.mean(gz))*7.5, bounds_error=False)
        dgz = dEye + dhead(eyeT[0:-1])

        plt.figure()
        plt.hist(dhead(eyeT),bins=21,range = (-10,10))
        plt.xlabel('dhead')
        detail_pdf.savefig()
        plt.close()

        plt.figure()
        plt.hist(dgz,bins=21,range = (-10,10))
        plt.xlabel('dgaze')
        detail_pdf.savefig()
        plt.close()
        
        plt.figure()
        if len(dEye[0:-1:10]) == len(dhead(eyeT[0:-1:10])):
            plt.plot(dEye[0:-1:10],dhead(eyeT[0:-1:10]),'.')
        elif len(dEye[0:-1:10]) > len(dhead(eyeT[0:-1:10])):
            plt.plot(dEye[0:-1:10][:len(dhead(eyeT[0:-1:10]))],dhead(eyeT[0:-1:10]),'.')
        elif len(dEye[0:-1:10]) < len(dhead(eyeT[0:-1:10])):
            plt.plot(dEye[0:-1:10],dhead(eyeT[0:-1:10])[:len(dEye[0:-1:10])],'.')
        plt.xlabel('dEye'); plt.ylabel('dHead')
        plt.xlim((-10,10)); plt.ylim((-10,10))
        plt.plot([-10,10],[10,-10], 'r')
        detail_pdf.savefig()
        plt.close()
      
    print('plotting saccade-locked psths')
    trange = np.arange(-1,1.1,0.025)
    if free_move is True:
        sthresh = 5
        upsacc = eyeT[ (np.append(dEye,0)>sthresh)]
        downsacc = eyeT[ (np.append(dEye,0)<-sthresh)]
    else:
        sthresh = 3
        upsacc = eyeT[np.append(dEye,0)>sthresh]
        downsacc = eyeT[np.append(dEye,0)<-sthresh]   
    upsacc_avg, downsacc_avg, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
    plt.title('all dEye')
    detail_pdf.savefig()
    plt.close()

    if free_move is True:
        # plot gaze shifting eye movements
        sthresh = 5
        upsacc = eyeT[(np.append(dEye,0)>sthresh) & (np.append(dgz,0)>sthresh)]
        downsacc = eyeT[(np.append(dEye,0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
        upsacc_avg_gaze_shift_dEye, downsacc_avg_gaze_shift_dEye, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('gaze shift dEye');  detail_pdf.savefig() ;  plt.close()
        # plot compensatory eye movements    
        sthresh = 3
        upsacc = eyeT[(np.append(dEye,0)>sthresh) & (np.append(dgz,0)<1)]
        downsacc = eyeT[(np.append(dEye,0)<-sthresh) & (np.append(dgz,0)>-1)]
        upsacc_avg_comp_dEye, downsacc_avg_comp_dEye, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('comp dEye'); detail_pdf.savefig() ;  plt.close()
        # plot gaze shifting head movements
        sthresh = 3
        upsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)>sthresh) & (np.append(dgz,0)>sthresh)]
        downsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
        upsacc_avg_gaze_shift_dHead, downsacc_avg_gaze_shift_dHead, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('gaze shift dhead') ; detail_pdf.savefig() ;  plt.close()
        # plot compensatory head movements
        sthresh = 3
        upsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)>sthresh) & (np.append(dgz,0)<1)]
        downsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)<-sthresh) & (np.append(dgz,0)>-1)]
        upsacc_avg_comp_dHead, downsacc_avg_comp_dHead, saccade_lock_fig = plot_saccade_locked(goodcells, upsacc,  downsacc, trange)
        plt.title('comp dhead') ; detail_pdf.savefig() ;  plt.close()

    # normalize and plot eye radius
    eyeR = eye_params.sel(ellipse_params='longaxis').copy()
    Rnorm = (eyeR-np.mean(eyeR)) / np.std(eyeR)
    plt.figure()
    plt.plot(eyeT,Rnorm)
    plt.xlabel('secs')
    plt.ylabel('normalized pupil R')
    diagnostic_pdf.savefig()
    plt.close()

    print('plotting spike rate vs pupil radius and position')
    # plot rate vs pupil
    R_range = np.linspace(10,50,10)
    spike_rate_vs_pupil_radius_cent, spike_rate_vs_pupil_radius_tuning, spike_rate_vs_pupil_radius_err, spike_rate_vs_pupil_radius_fig = plot_spike_rate_vs_var(eyeR, R_range, goodcells, eyeT, t, 'pupil radius')
    detail_pdf.savefig()
    plt.close()

    # normalize eye position
    eyeTheta = eye_params.sel(ellipse_params = 'theta').copy()
    thetaNorm = (eyeTheta - np.mean(eyeTheta))/np.std(eyeTheta)
    plt.plot(eyeT[0:3600],thetaNorm[0:3600])
    plt.xlabel('secs'); plt.ylabel('normalized eye theta')
    diagnostic_pdf.savefig()
    plt.close()

    eyePhi = eye_params.sel(ellipse_params='phi').copy()
    phiNorm = (eyePhi-np.mean(eyePhi)) / np.std(eyePhi)

    print('plotting spike rate vs theta/phi')
    # plot rate vs theta
    th_range = np.linspace(-30,30,10)
    spike_rate_vs_theta_cent, spike_rate_vs_theta_tuning, spike_rate_vs_theta_err, spike_rate_vs_theta_fig = plot_spike_rate_vs_var(th, th_range, goodcells, eyeT, t, 'eye theta')
    detail_pdf.savefig()
    plt.close()
    phi_range = np.linspace(-30,30,10)
    spike_rate_vs_phi_cent, spike_rate_vs_phi_tuning, spike_rate_vs_phi_err, spike_rate_vs_phi_fig = plot_spike_rate_vs_var(phi, phi_range, goodcells, eyeT, t, 'eye phi')
    detail_pdf.savefig()
    plt.close()
    
    if free_move is True:
        print('plotting spike rate vs gyro and speed')
        # get active times only
        active_interp = interp1d(model_t, model_active, bounds_error=False)
        active_accT = active_interp(accT.values)
        use = np.where(active_accT > 40)
        # spike rate vs gyro x
        gx_range = np.linspace(-5,5,10)
        active_gx = ((gx-np.mean(gx))*7.5)[use]
        spike_rate_vs_gx_cent, spike_rate_vs_gx_tuning, spike_rate_vs_gx_err, spike_rate_vs_gx_fig = plot_spike_rate_vs_var(active_gx, gx_range, goodcells, accT[use], t, 'gyro x')
        detail_pdf.savefig()
        plt.close()
        # spike rate vs gyro y
        gy_range = np.linspace(-5,5,10)
        active_gy = ((gy-np.mean(gy))*7.5)[use]
        spike_rate_vs_gy_cent, spike_rate_vs_gy_tuning, spike_rate_vs_gy_err, spike_rate_vs_gy_fig = plot_spike_rate_vs_var(active_gy, gy_range, goodcells, accT[use], t, 'gyro y')
        detail_pdf.savefig()
        plt.close()
        # spike rate vs gyro z
        gz_range = np.linspace(-7,7,10)
        active_gz = ((gz-np.mean(gz))*7.5)[use]
        spike_rate_vs_gz_cent, spike_rate_vs_gz_tuning, spike_rate_vs_gz_err, spike_rate_vs_gz_fig = plot_spike_rate_vs_var(active_gz, gz_range, goodcells, accT[use], t, 'gyro z')
        detail_pdf.savefig()
        plt.close()

    if free_move is False and has_mouse is True:
        print('plotting spike rate vs speed')
        spd_range = [0, 0.01, 0.1, 0.2, 0.5, 1.0]
        spike_rate_vs_spd_cent, spike_rate_vs_spd_tuning, spike_rate_vs_spd_err, spike_rate_vs_spd_fig = plot_spike_rate_vs_var(spd, spd_range, goodcells, speedT, t, 'speed')
        detail_pdf.savefig()
        plt.close()

    if free_move is True:
        print('plotting spike rate vs pitch/roll')
        # roll vs spike rate
        roll_range = np.linspace(-30,30,10)
        spike_rate_vs_roll_cent, spike_rate_vs_roll_tuning, spike_rate_vs_roll_err, spike_rate_vs_roll_fig = plot_spike_rate_vs_var(groll[use], roll_range, goodcells, accT[use], t, 'roll')
        detail_pdf.savefig()
        plt.close()
        # pitch vs spike rate
        pitch_range = np.linspace(-30,30,10)
        spike_rate_vs_pitch_cent, spike_rate_vs_pitch_tuning, spike_rate_vs_pitch_err, spike_rate_vs_pitch_fig = plot_spike_rate_vs_var(gpitch[use], pitch_range, goodcells, accT[use], t, 'pitch')
        detail_pdf.savefig()
        plt.close()
        print('plotting pitch/roll vs th/phi')
        # subtract mean from roll and pitch to center around zero
        pitch = gpitch - np.mean(gpitch)
        roll = groll - np.mean(groll)
        # pitch vs theta
        pitchi1d = interp1d(accT, pitch, bounds_error=False)
        pitch_interp = pitchi1d(eyeT)
        plt.figure()
        plt.plot(pitch_interp[::100], th[::100], '.'); plt.xlabel('pitch'); plt.ylabel('theta')
        plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
        diagnostic_pdf.savefig()
        plt.close()
        # roll vs phi
        rolli1d = interp1d(accT, roll, bounds_error=False)
        roll_interp = rolli1d(eyeT)
        plt.figure()
        plt.plot(roll_interp[::100], phi[::100], '.'); plt.xlabel('roll'); plt.ylabel('phi')
        plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')
        diagnostic_pdf.savefig()
        plt.close()
        # roll vs theta
        plt.figure()
        plt.plot(roll_interp[::100], th[::100], '.'); plt.xlabel('roll'); plt.ylabel('theta')
        plt.ylim([-60,60]); plt.xlim([-60,60])
        diagnostic_pdf.savefig()
        plt.close()
        # pitch vs phi
        plt.figure()
        plt.plot(pitch_interp[::100], phi[::100], '.'); plt.xlabel('pitch'); plt.ylabel('phi')
        plt.ylim([-60,60]); plt.xlim([-60,60])
        diagnostic_pdf.savefig()
        plt.close()
        # histogram of pitch values
        plt.figure()
        plt.hist(pitch, bins=50); plt.xlabel('pitch')
        diagnostic_pdf.savefig()
        plt.close()
        # histogram of pitch values
        plt.figure()
        plt.hist(roll, bins=50); plt.xlabel('roll')
        diagnostic_pdf.savefig()
        plt.close()
        # histogram of th values
        plt.figure()
        plt.hist(th, bins=50); plt.xlabel('theta')
        diagnostic_pdf.savefig()
        plt.close()
        # histogram of pitch values
        plt.figure()
        plt.hist(phi, bins=50); plt.xlabel('phi')
        diagnostic_pdf.savefig()
        plt.close()

    print('making overview plot')
    if file_dict['stim_type'] == 'grat':
        summary_fig = plot_overview(goodcells, crange, resp, file_dict, staAll, trange, upsacc_avg, downsacc_avg, ori_tuning=ori_tuning, drift_spont=drift_spont, grating_ori=grating_ori, sf_cat=sf_cat, grating_rate=grating_rate, spont_rate=spont_rate)
    else:
        summary_fig = plot_overview(goodcells, crange, resp, file_dict, staAll, trange, upsacc_avg, downsacc_avg)
    overview_pdf.savefig()
    plt.close()

    print('making summary plot')
    hist_dt = 1
    hist_t = np.arange(0, np.max(worldT),hist_dt)
    plt.subplots(n_units+3,1,figsize=(8,int(np.ceil(n_units/3))))
    plt.tight_layout()
    # either gyro or optical mouse reading
    plt.subplot(n_units+3,1,1)
    if has_imu:
        plt.plot(accT,gz)
        plt.xlim(0, np.max(worldT)); plt.ylabel('gz'); plt.title('gyro')
    elif has_mouse:
        plt.plot(speedT,spd)
        plt.xlim(0, np.max(worldT)); plt.ylabel('cm/sec'); plt.title('mouse speed')  
    # pupil diameter
    plt.subplot(n_units+3,1,2)
    plt.plot(eyeT,eye_params.sel(ellipse_params = 'longaxis'))
    plt.xlim(0, np.max(worldT)); plt.ylabel('rad (pix)'); plt.title('pupil diameter')
    # worldcam contrast
    plt.subplot(n_units+3,1,3)
    plt.plot(worldT,contrast)
    plt.xlim(0, np.max(worldT)); plt.ylabel('contrast a.u.'); plt.title('contrast')
    # raster
    for i,ind in enumerate(goodcells.index):
        rate,bins = np.histogram(ephys_data.at[ind,'spikeT'],hist_t)
        plt.subplot(n_units+3,1,i+4)
        plt.plot(bins[0:-1],rate)
        plt.xlim(bins[0],bins[-1]); plt.ylabel('unit ' + str(ind))
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()
    # clear up space in memory
    del ephys_data
    gc.collect()

    print('closing pdfs')
    overview_pdf.close(); detail_pdf.close(); diagnostic_pdf.close()

    print('organizing data and saving .h5')
    split_base_name = file_dict['name'].split('_')
    date = split_base_name[0]; mouse = split_base_name[1]; exp = split_base_name[2]; rig = split_base_name[3]
    try:
        stim = '_'.join(split_base_name[4:])
    except:
        stim = split_base_name[4:]
    session_name = date+'_'+mouse+'_'+exp+'_'+rig
    unit_data = pd.DataFrame([])

    elif free_move is True and file_dict['stim_type'] == 'light_arena':
        
    elif free_move is True and file_dict['stim_type'] == 'dark_arena':
        
    print('clearing memory')
    del data_out
    gc.collect()
    print('done')


def session_ephys_analysis(config):
    """
    run ephys analysis on a full session, finding all recordings in that session and organizing analysis options
    INPUTS
        config: options dict
    OUTPUTS
        None
    """
    # get options out
    data_path = config['animal_dir']
    unit = config['ephys_analysis']['unit_to_highlight']
    probe_name = config['ephys_analysis']['probe_type']
    # get subdirectories (i.e. name of each recording for this session)
    dirnames = list_subdirs(data_path)
    recording_names = sorted([i for i in dirnames if 'hf' in i or 'fm' in i])
    if config['ephys_analysis']['recording_list'] != []:
        recording_names = [i for i in recording_names if i in config['ephys_analysis']['recording_list']]
    # iterate through each recording's name
    for recording_name in recording_names:
        try:
            print('starting ephys analysis for',recording_name,'in path',data_path)
            if 'fm' in recording_name:
                fm = True
            elif 'fm' not in recording_name:
                fm = False
            this_unit = int(unit)
            if fm == True and 'light' in recording_name:
                stim_type = 'light_arena'
            elif fm == True and 'dark' in recording_name:
                stim_type = 'dark_arena'
            elif fm == True and 'light' not in recording_name and 'dark' not in recording_name:
                stim_type = 'light_arena'
            elif 'wn' in recording_name:
                stim_type = 'white_noise'
            elif 'grat' in recording_name:
                stim_type = 'gratings'
            elif 'noise' in recording_name:
                stim_type = 'sparse_noise'
            elif 'revchecker' in recording_name:
                stim_type = 'revchecker'
            recording_path = os.path.join(data_path, recording_name)
            norm_recording_path = os.path.normpath(recording_path).replace('\\', '/')
            full_recording_name = '_'.join(norm_recording_path.split('/')[-3:-1])+'_control_Rig2_'+os.path.split(norm_recording_path)[1].split('/')[-1]
            mp4 = config['ephys_analysis']['write_videos']
            drop_slow_frames = config['parameters']['drop_slow_frames']
            file_dict = find_files(recording_path, full_recording_name, fm, this_unit, stim_type, mp4, probe_name, drop_slow_frames)
            run_ephys_analysis(file_dict)
        except:
            print(traceback.format_exc())