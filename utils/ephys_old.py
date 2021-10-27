def plot_overview(goodcells, crange, resp, file_dict, staAll, trange, upsacc_avg, downsacc_avg,
                 ori_tuning=None, drift_spont=None, grating_ori=None, sf_cat=None, grating_rate=None,
                 spont_rate=None):
    """
    overview figure of ephys analysis
    INPUTS
        goodcells: ephys dataframe
        crange: range of contrast bins used
        resp: contrast reponse for each bin in crange
        file_dict: dict of files and options (comes from function find_files)
        staAll: STA for all units
        trange: range of time values used in head/eye movement plots
        upsacc_avg: average eye movement reponse
        downsacc_avg: average eye movement reponse
        ori_tuning: orientation reponse
        drift_spont: grating orientation spont
        grating_ori: grating response
        sf_cat: spatial frequency categories
        grating_rate: sp/sec at each sf
        spont_rate: grating spontanious rate
    OUTPUTS
        fig: figure
    """
    n_units = len(goodcells)
    samprate = 30000  # ephys sample rate
    fig = plt.figure(figsize=(5,np.int(np.ceil(n_units/3))),dpi=50)
    for i, ind in enumerate(goodcells.index): 
        # plot waveform
        plt.subplot(n_units,4,i*4 + 1)
        wv = goodcells.at[ind,'waveform']
        plt.plot(np.arange(len(wv))*1000/samprate,goodcells.at[ind,'waveform'])
        plt.xlabel('msec')
        plt.title(str(ind)+' '+goodcells.at[ind,'KSLabel']+' cont='+str(goodcells.at[ind,'ContamPct']))
        # plot CRF
        if grating_ori is not None:
            # for gratings stim, plot orientation tuning curve
            plt.subplot(n_units,4,i*4 + 2)
            plt.scatter(grating_ori,grating_rate[i,:],c=sf_cat)
            plt.plot(3*np.ones(len(spont_rate[i,:])),spont_rate[i,:],'r.')
        if file_dict['stim_type'] == 'dark_arena':
            # dark arena will have no values for contrast response function
            # skip this panel for now
            plt.subplot(n_units,4,i*4 + 2)
            plt.axis('off')
        else:
            # for light fm and all hf besides gratings, plot CRF
            plt.subplot(n_units,4,i*4 + 2)
            plt.plot(crange[2:-1],resp[i,2:-1])
            plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec')
            try:
                plt.ylim([0,np.nanmax(resp[i,2:-1])])
            except ValueError:
                plt.ylim(0,1)
        # plot STA or tuning curve
        plt.subplot(n_units,4,i*4 + 3)
        if ori_tuning is not None:
            plt.plot(np.arange(8)*45, ori_tuning[i,:,0], label = 'low sf')
            plt.plot(np.arange(8)*45, ori_tuning[i,:,1], label = 'mid sf')
            plt.plot(np.arange(8)*45, ori_tuning[i,:,2], label = 'hi sf')
            plt.plot([0,315],[drift_spont[i],drift_spont[i]],'r:', label = 'spont')
            try:
                plt.ylim(0,np.nanmax(ori_tuning[i,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.xlabel('orientation (deg)')
        else:
            sta = staAll[i,:,:]
            staRange = np.max(np.abs(sta))*1.2
            if staRange<0.25:
                staRange=0.25
            plt.imshow(staAll[i,:,:],vmin = -staRange, vmax= staRange, cmap = 'jet')    
        # plot eye movements
        plt.subplot(n_units,4,i*4 + 4)
        plt.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[i,:])
        plt.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[i,:],'r')
        plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2),'r')
        plt.ylim([0, np.max(upsacc_avg[i,:])*1.8])
        plt.ylabel('sp/sec')
    plt.tight_layout()
    return fig

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

def make_summary_panels(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params,
			dEye, goodcells, units, this_unit, eyeInterp, worldInterp, top_vid,
			topT, topInterp, th, phi, top_speed, accT=None, gz=None, speedT=None, spd=None):
    # set up figure
    fig = plt.figure(figsize = (10,16))
    gs = fig.add_gridspec(12,6)
    axEye = fig.add_subplot(gs[0:2,0:2])
    axWorld = fig.add_subplot(gs[0:2,2:4])
    axTopdown = fig.add_subplot(gs[0:2,4:6])
    axRad = fig.add_subplot(gs[2,:])
    axTh = fig.add_subplot(gs[3,:])
    axGyro = fig.add_subplot(gs[4,:])
    axR = fig.add_subplot(gs[5:12,:])
    
    #timerange and center frame (only)
    tr = [7, 7+15]
    fr = np.mean(tr) # time for frame
    eyeFr = np.abs(eyeT-fr).argmin(dim = "frame")
    worldFr = np.abs(worldT-fr).argmin(dim = "frame")
    topFr = np.abs(topT-fr).argmin(dim = "frame")

    axEye.cla(); axEye.axis('off')
    axEye.imshow(eye_vid[eyeFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")

    axWorld.cla();  axWorld.axis('off'); 
    axWorld.imshow(world_vid[worldFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")

    axTopdown.cla();  axTopdown.axis('off'); 
    axTopdown.imshow(top_vid[topFr,:,:],'gray',vmin=0,vmax=255,aspect = "equal")
    
    axTh.cla()
    axTh.plot(eyeT,th)
    axTh.set_xlim(tr[0],tr[1]); 
    axTh.set_ylabel('theta (deg)'); axTh.set_ylim(-50,0)

    axRad.cla()
    axRad.plot(eyeT,eye_params.sel(ellipse_params='longaxis'))
    axRad.set_xlim(tr[0],tr[1])
    axRad.set_ylabel('pupil radius')
    
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
    elif '16' in probe:
        sh_num = 16
    even_raster = np.arange(0,len(goodcells.index),sh_num)
    for i,ind in enumerate(goodcells.index):
        i = (even_raster+(i%32))[int(np.floor(i/32))]
        axR.vlines(goodcells.at[ind,'spikeT'],i-0.25,i+0.25,'k',linewidth=0.5) # all units
    axR.vlines(goodcells.at[units[this_unit],'spikeT'],this_unit-0.25,this_unit+0.25,'k',linewidth=0.5) # this unit
    
    n_units = len(goodcells)
    axR.set_ylim(n_units,-.5)
    axR.set_xlim(tr[0],tr[1]); axR.set_xlabel('secs'); axR.set_ylabel('unit')
    axR.spines['right'].set_visible(False)
    axR.spines['top'].set_visible(False)
    plt.tight_layout()
    return fig

def run_ephys_analysis(file_dict):
    """
    ephys analysis bringing together eyecam, worldcam, ephys data, imu data, and running ball optical mouse data
    runs on one recording at a time
    saves out an .h5 file for the rec structured as a dict of 
    h5 file is  best read in with pandas, or if pooling data across recordings, and then across sessions, with load_ephys func in /project_analysis/ephys/ephys_utils.py
    INPUTS
        file_dict: dictionary saved out from func find_files
    OUTPUTS
        None
    """
    # set up recording properties
    if file_dict['speed'] is None:
        free_move = True; has_imu = True; has_mouse = False
    else:
        free_move = False; has_imu = False; has_mouse = True
    # delete the existing h5 file, so that a new one can be written
    if os.path.isfile(os.path.join(file_dict['save'], (file_dict['name']+'_ephys_props.h5'))):
        os.remove(os.path.join(file_dict['save'], (file_dict['name']+'_ephys_props.h5')))
    # open three pdfs
    print('opening pdfs')
    overview_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_overview_analysis_figures.pdf')))
    detail_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_detailed_analysis_figures.pdf')))
    diagnostic_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_diagnostic_analysis_figures.pdf')))
    print('opening and resizing worldcam data')
    # open worldcam
    world_data = xr.open_dataset(file_dict['world'])
    world_vid_raw = np.uint8(world_data['WORLD_video'])
    # resize worldcam
    sz = world_vid_raw.shape # raw video size
    # if size is larger than the target 60x80, resize by 0.5
    if sz[1]>160:
        downsamp = 0.5
        world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
        for f in range(sz[0]):
            world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
    else:
        # if the worldcam has already been resized when the nc file was written in preprocessing, don't resize
        world_vid = world_vid_raw.copy()
    # world timestamps
    worldT = world_data.timestamps.copy()
    # plot worldcam timing
    fig, axs = plt.subplots(1,2)
    axs[0].plot(np.diff(worldT)[0:-1:10]); axs[0].set_xlabel('every 10th frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('worldcam timing')
    axs[1].hist(np.diff(worldT),100);axs[1].set_xlabel('deltaT')
    diagnostic_pdf.savefig()
    plt.close()
    # plot mean world image
    plt.figure()
    plt.imshow(np.mean(world_vid,axis=0)); plt.title('mean world image')
    diagnostic_pdf.savefig()
    plt.close()
    if free_move is True:
        print('opening top data')
        # open the topdown camera nc file
        top_data = xr.open_dataset(file_dict['top'])
        # get the speed of the base of the animal's tail in the topdown tracking
        # most points don't track well enough for this to be done with other parts of animal (e.g. head points)
        topx = top_data.TOP1_pts.sel(point_loc='tailbase_x').values; topy = top_data.TOP1_pts.sel(point_loc='tailbase_y').values
        topdX = np.diff(topx); topdY = np.diff(topy)
        top_speed = np.sqrt(topdX**2, topdY**2) # speed of tailbase in topdown camera
        topT = top_data.timestamps.copy() # read in time timestamps
        top_vid = np.uint8(top_data['TOP1_video']) # read in top video
        # clear from memory
        del top_data
        gc.collect()
    # load IMU data
    if file_dict['imu'] is not None:
        print('opening imu data')
        imu_data = xr.open_dataset(file_dict['imu'])
        accT = imu_data.IMU_data.sample # imu timestamps
        acc_chans = imu_data.IMU_data # imu dample data
        # raw gyro values
        gx = np.array(acc_chans.sel(channel='gyro_x_raw'))
        gy = np.array(acc_chans.sel(channel='gyro_y_raw'))
        gz = np.array(acc_chans.sel(channel='gyro_z_raw'))
        # gyro values in degrees
        gx_deg = np.array(acc_chans.sel(channel='gyro_x'))
        gy_deg = np.array(acc_chans.sel(channel='gyro_y'))
        gz_deg = np.array(acc_chans.sel(channel='gyro_z'))
        # pitch and roll in deg
        groll = np.array(acc_chans.sel(channel='roll'))
        gpitch = np.array(acc_chans.sel(channel='pitch'))
        # figure of gyro z
        plt.figure()
        plt.plot(gz_deg[0:100*60])
        plt.title('gyro z (deg)')
        plt.xlabel('frame')
        diagnostic_pdf.savefig()
        plt.close()
    # load optical mouse nc file from running ball
    if file_dict['speed'] is not None:
        print('opening speed data')
        speed_data = xr.open_dataset(file_dict['speed'])
        spdVals = speed_data.BALL_data
        try:
            spd = spdVals.sel(move_params = 'speed_cmpersec')
            spd_tstamps = spdVals.sel(move_params = 'timestamps')
        except:
            spd = spdVals.sel(frame = 'speed_cmpersec')
            spd_tstamps = spdVals.sel(frame = 'timestamps')
    print('opening ephys data')
    # ephys data for this individual recording
    ephys_data = pd.read_json(file_dict['ephys'])
    # sort units by shank and site order
    ephys_data = ephys_data.sort_values(by='ch', axis=0, ascending=True)
    ephys_data = ephys_data.reset_index()
    ephys_data = ephys_data.drop('index', axis=1)
    # spike times
    ephys_data['spikeTraw'] = ephys_data['spikeT']
    print('getting good cells')
    # select good cells from phy2
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    units = goodcells.index.values
    # get number of good units
    n_units = len(goodcells)
    # plot spike raster
    spikeraster_fig = plot_spike_raster(goodcells)
    detail_pdf.savefig()
    plt.close()
    print('opening eyecam data')
    # load eye data
    eye_data = xr.open_dataset(file_dict['eye'])
    eye_vid = np.uint8(eye_data['REYE_video'])
    eyeT = eye_data.timestamps.copy()
    # plot eye timestamps
    fig, axs = plt.subplots(1,2)
    axs[0].plot(np.diff(eyeT)[0:-1:10]); axs[0].set_xlabel('every 10th frame'); axs[0].set_ylabel('deltaT'); axs[0].set_title('eyecam timing')
    axs[1].hist(np.diff(eyeT),100);axs[1].set_xlabel('deltaT')
    diagnostic_pdf.savefig()
    plt.close()
    # plot eye postion across recording
    eye_params = eye_data['REYE_ellipse_params']
    eyepos_fig = plot_eye_pos(eye_params)
    detail_pdf.savefig()
    plt.close()
    # define theta, phi and zero-center
    th = np.array((eye_params.sel(ellipse_params = 'theta')-np.nanmean(eye_params.sel(ellipse_params = 'theta')))*180/3.14159)
    phi = np.array((eye_params.sel(ellipse_params = 'phi')-np.nanmean(eye_params.sel(ellipse_params = 'phi')))*180/3.14159)
    # plot optical mouse speeds
    if file_dict['speed'] is not None:
        fig = plt.figure()
        plt.plot(spd_tstamps,spd)
        plt.xlabel('sec'); plt.ylabel('running speed cm/sec')
        detail_pdf.savefig()
        plt.close()
    print('adjusting camera times to match ephys')
    # adjust eye/world/top times relative to ephys
    ephysT0 = ephys_data.iloc[0,12]
    eyeT = eye_data.timestamps  - ephysT0
    if eyeT[0]<-600:
        eyeT = eyeT + 8*60*60 # 8hr offset for some data
    worldT = world_data.timestamps - ephysT0
    if worldT[0]<-600:
        worldT = worldT + 8*60*60
    if free_move is True and has_imu is True:
        accTraw = imu_data.IMU_data.sample - ephysT0
    if free_move is False and has_mouse is True:
        speedT = spd_tstamps - ephysT0
    if free_move is True:
        topT = topT - ephysT0
    # make space in memory
    del eye_data
    gc.collect()
    if file_dict['drop_slow_frames'] is True:
        # in the case that the recording has long time lags, drop data in a window +/- 3 frames around these slow frames
        isfast = np.diff(eyeT)<=0.03
        isslow = sorted(list(set(chain.from_iterable([list(range(int(i)-3,int(i)+4)) for i in np.where(isfast==False)[0]]))))
        th[isslow] = np.nan
        phi[isslow] = np.nan
    # check that deinterlacing worked correctly
    # plot theta and theta_switch
    # want theta_switch to be jagged, theta to be smooth
    theta_switch_fig, th_switch = plot_param_switch_check(eye_params)
    diagnostic_pdf.savefig()
    plt.close()
    # plot eye variables
    fig, axs = plt.subplots(4,1)
    for i,val in enumerate(eye_params.ellipse_params[0:4]):
        axs[i].plot(eyeT[0:-1:10],eye_params.sel(ellipse_params = val)[0:-1:10])
        axs[i].set_ylabel(val.values)
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()
    # calculate eye veloctiy
    dEye = np.diff(th)
    # check accelerometer / eye temporal alignment
    if file_dict['imu'] is not None:
        print('checking accelerometer / eye temporal alignment')
        # plot eye velocity against head movements
        plt.figure
        plt.plot(eyeT[0:-1],-dEye,label='-dEye')
        plt.plot(accTraw,gz_deg,label='gz')
        plt.legend()
        plt.xlim(0,10); plt.xlabel('secs'); plt.ylabel('gyro (deg)')
        diagnostic_pdf.savefig()
        plt.close()
        lag_range = np.arange(-0.2,0.2,0.002)
        cc = np.zeros(np.shape(lag_range))
        t1 = np.arange(5,len(dEye)/60-120,20).astype(int) # was np.arange(5,1600,20), changed for shorter videos
        t2 = t1 + 60
        offset = np.zeros(np.shape(t1))
        ccmax = np.zeros(np.shape(t1))
        acc_interp = interp1d(accTraw, (gz-3)*7.5)
        for tstart in tqdm(range(len(t1))):
            for l in range(len(lag_range)):
                try:
                    c, lag= nanxcorr(-dEye[t1[tstart]*60 : t2[tstart]*60] , acc_interp(eyeT[t1[tstart]*60:t2[tstart]*60]+lag_range[l]),1)
                    cc[l] = c[1]
                except: # occasional problem with operands that cannot be broadcast togther because of different shapes
                    cc[l] = np.nan
            offset[tstart] = lag_range[np.argmax(cc)]    
            ccmax[tstart] = np.max(cc)
        offset[ccmax<0.1] = np.nan
        acc_eyetime_alligment_fig = plot_acc_eyetime_alignment(eyeT, t1, offset, ccmax)
        diagnostic_pdf.savefig()
        plt.close()
        del ccmax
        gc.collect()
    if file_dict['imu'] is not None:
        print('fitting regression to timing drift')
        # fit regression to timing drift
        model = LinearRegression()
        dataT = np.array(eyeT[t1*60 + 30*60])
        model.fit(dataT[~np.isnan(offset)].reshape(-1,1),offset[~np.isnan(offset)]) 
        offset0 = model.intercept_
        drift_rate = model.coef_
        plot_regression_timing_fit_fig = plot_regression_timing_fit(dataT[~np.isnan(dataT)], offset[~np.isnan(dataT)], offset0, drift_rate)
        diagnostic_pdf.savefig()
        plt.close()
        del dataT
        gc.collect()
    elif file_dict['speed'] is not None:
        offset0 = 0.1
        drift_rate = -0.000114
    if file_dict['imu'] is not None:
        accT = accTraw - (offset0 + accTraw*drift_rate)
        del accTraw
    print('correcting ephys spike times for offset and timing drift')
    for i in ephys_data.index:
        ephys_data.at[i,'spikeT'] = np.array(ephys_data.at[i,'spikeTraw']) - (offset0 + np.array(ephys_data.at[i,'spikeTraw']) *drift_rate)
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    print('preparing worldcam video')
    if free_move and file_dict['stim_type'] != 'dark_arena':
        print('estimating eye-world calibration')
        fig, xmap, ymap = eye_shift_estimation(th, phi, eyeT, world_vid, worldT, 60*60)
        xcorrection = xmap.copy()
        ycorrection = ymap.copy()
        print('shifting worldcam for eyes')
        thInterp =interp1d(eyeT,th, bounds_error = False)
        phiInterp =interp1d(eyeT,phi, bounds_error = False)
        thWorld = thInterp(worldT)
        phiWorld = phiInterp(worldT)
        for f in tqdm(range(np.shape(world_vid)[0])):
            world_vid[f,:,:] = imshift(world_vid[f,:,:],(-np.int8(thInterp(worldT[f])*ycorrection[0] + phiInterp(worldT[f])*ycorrection[1]),
                                                         -np.int8(thInterp(worldT[f])*xcorrection[0] + phiInterp(worldT[f])*xcorrection[1])))
        # print('saving worldcam video corrected for eye movements')
        # np.save(file=os.path.join(file_dict['save'], 'corrected_worldcam.npy'), arr=world_vid)
    std_im = np.std(world_vid,axis=0)
    img_norm = (world_vid-np.mean(world_vid,axis=0))/std_im
    std_im[std_im<20] = 0
    img_norm = img_norm * (std_im>0)
    # worldcam contrast
    contrast = np.empty(worldT.size)
    for i in range(worldT.size):
        contrast[i] = np.nanstd(img_norm[i,:,:])
    plt.plot(contrast[2000:3000])
    plt.xlabel('time')
    plt.ylabel('contrast')
    diagnostic_pdf.savefig()
    plt.close()
    # std of worldcam image
    fig = plt.figure()
    plt.imshow(std_im)
    plt.colorbar(); plt.title('std img')
    diagnostic_pdf.savefig()
    plt.close()
    # make movie and sound
    this_unit = file_dict['cell']
    # set up interpolators for eye and world videos
    eyeInterp = interp1d(eyeT, eye_vid, axis=0, bounds_error=False)
    worldInterp = interp1d(worldT, world_vid_raw, axis=0, bounds_error=False)
    if free_move:
        topInterp = interp1d(topT, top_vid, axis=0,bounds_error=False)
    if file_dict['imu'] is not None:
        fig = make_summary_panels(file_dict, eyeT, worldT, eye_vid, world_vid_raw, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, top_vid, topT, topInterp, th, phi, top_speed, accT=accT, gz=gz_deg)
        detail_pdf.savefig()
        plt.close()
    if file_dict['mp4']:
        if file_dict['imu'] is not None:
            print('making video figure')
            vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid_raw, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, top_vid, topT, topInterp, th, phi, top_speed, accT=accT, gz=gz_deg)
            audfile = make_sound(file_dict, ephys_data, units, this_unit)
            merge_mp4_name = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'_merge.mp4'))
            subprocess.call(['ffmpeg', '-i', vidfile, '-i', audfile, '-c:v', 'copy', '-c:a', 'aac', '-y', merge_mp4_name])
        elif file_dict['speed'] is not None:
            print('making video figure')
            vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid_raw, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, top_vid=None, topT=None, topInterp=None, th=th, phi=phi, top_speed=None, speedT=speedT, spd=spd)
            audfile = make_sound(file_dict, ephys_data, units, this_unit)
            merge_mp4_name = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'_merge.mp4'))
            subprocess.call(['ffmpeg', '-i', vidfile, '-i', audfile, '-c:v', 'copy', '-c:a', 'aac', '-y', merge_mp4_name])

    if free_move is True and file_dict['imu'] is not None:
        plt.figure()
        plt.plot(eyeT[0:-1],np.diff(th),label = 'dTheta')
        plt.plot(accT-0.1,(gz-3)*10, label = 'gyro')
        plt.xlim(30,40); plt.ylim(-12,12); plt.legend(); plt.xlabel('secs')
        diagnostic_pdf.savefig()
        plt.close()
    print('plot eye and gaze (i.e. saccade and fixate)')
    if free_move and file_dict['imu'] is not None:
        gInterp = interp1d(accT,(gz-np.nanmean(gz))*7.5 , bounds_error = False)
        plt.figure(figsize = (8,4))
        plot_saccade_and_fixate_fig = plot_saccade_and_fixate(eyeT, dEye, gInterp, th)
        diagnostic_pdf.savefig()
        plt.close()
    plt.subplot(1,2,1)
    plt.imshow(std_im)
    plt.title('std dev of image')
    plt.subplot(1,2,2)
    plt.imshow(np.mean(world_vid, axis=0), vmin=0, vmax=255)
    plt.title('mean of image')
    diagnostic_pdf.savefig()
    plt.close()
    # set up timebase for subsequent analysis
    dt = 0.025
    t = np.arange(0, np.max(worldT),dt)
    # interpolate and plot contrast
    newc = interp1d(worldT,contrast)
    contrast_interp = newc(t[0:-1])
    plt.plot(t[0:600],contrast_interp[0:600])
    plt.xlabel('secs'); plt.ylabel('world contrast')
    diagnostic_pdf.savefig()
    plt.close()
    print('calculating firing rate')
    # calculate firing rate at new timebase
    ephys_data['rate'] = nan
    ephys_data['rate'] = ephys_data['rate'].astype(object)
    for i,ind in enumerate(ephys_data.index):
        ephys_data.at[ind,'rate'],bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
    ephys_data['rate']= ephys_data['rate']/dt
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    print('calculating contrast reponse functions')
    # mean firing rate in timebins correponding to contrast ranges
    resp = np.empty((n_units,12))
    crange = np.arange(0,1.2,0.1)
    for i, ind in enumerate(goodcells.index):
        for c,cont in enumerate(crange):
            resp[i,c] = np.mean(goodcells.at[ind,'rate'][(contrast_interp>cont) & (contrast_interp<(cont+0.1))])
    # plot individual contrast response functions in subplots
    crf_cent, crf_tuning, crf_err, crf_fig = plot_spike_rate_vs_var(contrast, crange, goodcells, worldT, t, 'contrast')
    detail_pdf.savefig()
    plt.close()
    eyeR = eye_params.sel(ellipse_params = 'longaxis').copy()
    Rnorm = (eyeR - np.mean(eyeR))/np.std(eyeR)
    try:
        plt.figure()
        plt.plot(eyeT,Rnorm)
        #plt.xlim([0,60])
        plt.xlabel('secs')
        plt.ylabel('normalized pupil R')
        diagnostic_pdf.savefig()
        plt.close()
    except:
        pass

    if not free_move:
        # don't run for freely moving, at least for now, because recordings can be too long to fit ephys binary into memory
        # was only a problem for a 128ch recording
        # but hf recordings should be sufficient length to get good estimate
        print('starting continuous LFP laminar depth estimation')
        print('loading ephys binary file')
        # read in ephys binary
        lfp_ephys = read_ephys_bin(file_dict['ephys_bin'], file_dict['probe_name'], do_remap=True, mapping_json=file_dict['mapping_json'])
        print('applying bandpass filter')
        # subtract mean in time dim and apply bandpass filter
        ephys_center_sub = lfp_ephys - np.mean(lfp_ephys,0)
        filt_ephys = butter_bandpass(ephys_center_sub, lowcut=600, highcut=6000, fs=30000, order=6)
        print('getting lfp power profile across channels')
        # get lfp power profile for each channel
        ch_num = np.size(filt_ephys,1)
        lfp_power_profiles = np.zeros([ch_num])
        for ch in range(ch_num):
            lfp_power_profiles[ch] = np.sqrt(np.mean(filt_ephys[:,ch]**2)) # multiunit LFP power profile
        # median filter
        print('applying median filter')
        lfp_power_profiles_filt = medfilt(lfp_power_profiles)
        if file_dict['probe_name'] == 'DB_P64-8':
            ch_spacing = 25/2
        else:
            ch_spacing = 25
        print('making figures')
        if ch_num == 64:
            norm_profile_sh0 = lfp_power_profiles_filt[:32]/np.max(lfp_power_profiles_filt[:32])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            norm_profile_sh1 = lfp_power_profiles_filt[32:64]/np.max(lfp_power_profiles_filt[32:64])
            layer5_cent_sh1 = np.argmax(norm_profile_sh1)
            lfp_power_profiles = [norm_profile_sh0, norm_profile_sh1]
            lfp_layer5_centers = [layer5_cent_sh0, layer5_cent_sh1]
            plt.subplots(1,2)
            plt.tight_layout()
            plt.subplot(1,2,1)
            plt.plot(norm_profile_sh0,range(0,32))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            plt.subplot(1,2,2)
            plt.plot(norm_profile_sh1,range(0,32))
            plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
            plt.title('shank1')
            detail_pdf.savefig(); plt.close()
        elif ch_num == 16:
            norm_profile_sh0 = lfp_power_profiles_filt[:16]/np.max(lfp_power_profiles_filt[:16])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            lfp_power_profiles = [norm_profile_sh0]
            lfp_layer5_centers = [layer5_cent_sh0]
            plt.figure()
            plt.tight_layout()
            plt.plot(norm_profile_sh0,range(0,16))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([17,-1]); plt.yticks(ticks=list(range(-1,17)),labels=(ch_spacing*np.arange(18)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            detail_pdf.savefig(); plt.close()
        elif ch_num == 128:
            norm_profile_sh0 = lfp_power_profiles_filt[:32]/np.max(lfp_power_profiles_filt[:32])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            norm_profile_sh1 = lfp_power_profiles_filt[32:64]/np.max(lfp_power_profiles_filt[32:64])
            layer5_cent_sh1 = np.argmax(norm_profile_sh1)
            norm_profile_sh2 = lfp_power_profiles_filt[64:96]/np.max(lfp_power_profiles_filt[64:96])
            layer5_cent_sh2 = np.argmax(norm_profile_sh2)
            norm_profile_sh3 = lfp_power_profiles_filt[96:128]/np.max(lfp_power_profiles_filt[96:128])
            layer5_cent_sh3 = np.argmax(norm_profile_sh3)
            lfp_power_profiles = [norm_profile_sh0, norm_profile_sh1, norm_profile_sh2, norm_profile_sh3]
            lfp_layer5_centers = [layer5_cent_sh0, layer5_cent_sh1, layer5_cent_sh2, layer5_cent_sh3]
            plt.subplots(1,4)
            plt.tight_layout()
            plt.subplot(1,4,1)
            plt.plot(norm_profile_sh0,range(0,32))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            plt.subplot(1,4,2)
            plt.plot(norm_profile_sh1,range(0,32))
            plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
            plt.title('shank1')
            plt.subplot(1,4,3)
            plt.plot(norm_profile_sh2,range(0,32))
            plt.plot(norm_profile_sh2[layer5_cent_sh2]+0.01,layer5_cent_sh2,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh2*ch_spacing)))
            plt.title('shank2')
            plt.subplot(1,4,4)
            plt.plot(norm_profile_sh3,range(0,32))
            plt.plot(norm_profile_sh3[layer5_cent_sh3]+0.01,layer5_cent_sh3,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh3*ch_spacing)))
            plt.title('shank3')
            detail_pdf.savefig(); plt.close()

    
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
        for unit_num, ind in enumerate(goodcells.index):
            cols = [stim+'_'+i for i in ['c_range',
                                        'crf_cent',
                                        'crf_tuning',
                                        'crf_err',
                                        'spike_triggered_average',
                                        'sta_shape',
                                        'spike_triggered_variance',
                                        'upsacc_avg',
                                        'downsacc_avg',
                                        'upsacc_avg_gaze_shift_dEye',
                                        'downsacc_avg_gaze_shift_dEye',
                                        'upsacc_avg_comp_dEye',
                                        'downsacc_avg_comp_dEye',
                                        'upsacc_avg_gaze_shift_dHead',
                                        'downsacc_avg_gaze_shift_dHead',
                                        'upsacc_avg_comp_dHead',
                                        'downsacc_avg_comp_dHead',
                                        'spike_rate_vs_pupil_radius_cent',
                                        'spike_rate_vs_pupil_radius_tuning',
                                        'spike_rate_vs_pupil_radius_err',
                                        'spike_rate_vs_theta_cent',
                                        'spike_rate_vs_theta_tuning',
                                        'spike_rate_vs_theta_err',
                                        'spike_rate_vs_gz_cent',
                                        'spike_rate_vs_gz_tuning',
                                        'spike_rate_vs_gz_err',
                                        'spike_rate_vs_gx_cent',
                                        'spike_rate_vs_gx_tuning',
                                        'spike_rate_vs_gx_err',
                                        'spike_rate_vs_gy_cent',
                                        'spike_rate_vs_gy_tuning',
                                        'spike_rate_vs_gy_err',
                                        'trange',
                                        'dHead',
                                        'dEye',
                                        'eyeT',
                                        'theta',
                                        'phi',
                                        'gz',
                                        'spike_rate_vs_roll_cent',
                                        'spike_rate_vs_roll_tuning',
                                        'spike_rate_vs_roll_err',
                                        'spike_rate_vs_pitch_cent',
                                        'spike_rate_vs_pitch_tuning',
                                        'spike_rate_vs_pitch_err',
                                        'spike_rate_vs_phi_cent',
                                        'spike_rate_vs_phi_tuning',
                                        'spike_rate_vs_phi_err',
                                        'accT',
                                        'roll',
                                        'pitch',
                                        'roll_interp',
                                        'pitch_interp']]
            unit_df = pd.DataFrame(pd.Series([crange,
                                    crf_cent,
                                    crf_tuning[unit_num],
                                    crf_err[unit_num],
                                    np.ndarray.flatten(staAll[unit_num]),
                                    np.shape(staAll[unit_num]),
                                    np.ndarray.flatten(st_var[unit_num]),
                                    upsacc_avg[unit_num],
                                    downsacc_avg[unit_num],
                                    upsacc_avg_gaze_shift_dEye[unit_num],
                                    downsacc_avg_gaze_shift_dEye[unit_num],
                                    upsacc_avg_comp_dEye[unit_num],
                                    downsacc_avg_comp_dEye[unit_num],
                                    upsacc_avg_gaze_shift_dHead[unit_num],
                                    downsacc_avg_gaze_shift_dHead[unit_num],
                                    upsacc_avg_comp_dHead[unit_num],
                                    downsacc_avg_comp_dHead[unit_num],
                                    spike_rate_vs_pupil_radius_cent,
                                    spike_rate_vs_pupil_radius_tuning[unit_num],
                                    spike_rate_vs_pupil_radius_err[unit_num],
                                    spike_rate_vs_theta_cent,
                                    spike_rate_vs_theta_tuning[unit_num],
                                    spike_rate_vs_theta_err[unit_num],
                                    spike_rate_vs_gz_cent,
                                    spike_rate_vs_gz_tuning[unit_num],
                                    spike_rate_vs_gz_err[unit_num],
                                    spike_rate_vs_gx_cent,
                                    spike_rate_vs_gx_tuning[unit_num],
                                    spike_rate_vs_gx_err[unit_num],
                                    spike_rate_vs_gy_cent,
                                    spike_rate_vs_gy_tuning[unit_num],
                                    spike_rate_vs_gy_err[unit_num],
                                    trange,
                                    dhead,
                                    dEye,
                                    eyeT,
                                    th,
                                    phi,
                                    gz,
                                    spike_rate_vs_roll_cent,
                                    spike_rate_vs_roll_tuning[unit_num],
                                    spike_rate_vs_roll_err[unit_num],
                                    spike_rate_vs_pitch_cent,
                                    spike_rate_vs_pitch_tuning[unit_num],
                                    spike_rate_vs_pitch_err[unit_num],
                                    spike_rate_vs_phi_cent,
                                    spike_rate_vs_phi_tuning[unit_num],
                                    spike_rate_vs_phi_err[unit_num],
                                    accT,
                                    roll,
                                    pitch,
                                    roll_interp,
                                    pitch_interp]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
    data_out = pd.concat([goodcells, unit_data],axis=1)
    data_out.to_hdf(os.path.join(file_dict['save'], (file_dict['name']+'_ephys_props.h5')), 'w')
    print('clearing memory')
    del data_out
    gc.collect()
    print('done')

def load_ephys(csv_filepath):
    """
    using a .csv file of metadata identical to the one used to run batch analysis, pool experiments marked for inclusion and orgainze properties
    saved out from ephys analysis into .h5 files as columns and each unit as an index
    also reads in the .json of calibration properties saved out from fm recording eyecam analysis so that filtering can be done based on how well the eye tracking worked
    INPUTS
        csv_filepath: path to csv file used for batch analysis
    OUTPUTS
        all_data: DataFrame of all units marked for pooled analysis, with each index representing a unit across all recordings of a session
    """
    # open the csv file of metadata and pull out all of the desired data paths
    if type(csv_filepath) == str:
        csv = pd.read_csv(csv_filepath)
        for_data_pool = csv[csv['load_for_data_pool'] == any(['TRUE' or True or 'True'])]
    elif type(csv_filepath) == pd.Series:
        for_data_pool = csv_filepath
    goodsessions = []
    probenames_for_goodsessions = []
    layer5_depth_for_goodsessions = []
    # get all of the best freely moving recordings of a session into a dictionary
    goodlightrecs = dict(zip(list([j+'_'+i for i in [i.split('\\')[-1] for i in for_data_pool['animal_dirpath']] for j in [datetime.strptime(i,'%m/%d/%y').strftime('%m%d%y') for i in list(for_data_pool['experiment_date'])]]),[i if i !='' else 'fm1' for i in for_data_pool['best_light_fm']]))
    gooddarkrecs = dict(zip(list([j+'_'+i for i in [i.split('\\')[-1] for i in for_data_pool['animal_dirpath']] for j in [datetime.strptime(i,'%m/%d/%y').strftime('%m%d%y') for i in list(for_data_pool['experiment_date'])]]),[i if i !='' else None for i in for_data_pool['best_dark_fm']]))
    # change paths to work with linux
    if platform.system() == 'Linux':
        for ind, row in for_data_pool.iterrows():
            drive = [row['drive'] if row['drive'] == 'nlab-nas' else row['drive'].capitalize()][0]
            for_data_pool.loc[ind,'animal_dirpath'] = os.path.expanduser('~/'+('/'.join([row['computer'].title(), drive] + list(filter(None, row['animal_dirpath'].replace('\\','/').split('/')))[2:])))
    for ind, row in for_data_pool.iterrows():
        goodsessions.append(row['animal_dirpath'])
        probenames_for_goodsessions.append(row['probe_name'])
        layer5_depth_for_goodsessions.append(row['overwrite_layer5center'])
    # get the .h5 files from each day
    # this will be a list of lists, where each list inside of the main list has all the data of a single session
    sessions = [find('*_ephys_props.h5',session) for session in goodsessions]
    # read the data in and append them into one shared df
    all_data = pd.DataFrame([])
    ind = 0
    sessions = [i for i in sessions if i != []]
    for session in tqdm(sessions):
        session_data = pd.DataFrame([])
        for recording in session:
            rec_data = pd.read_hdf(recording)
            # get name of the current recording (i.e. 'fm' or 'hf1_wn')
            rec_type = '_'.join(([col for col in rec_data.columns.values if 'trange' in col][0]).split('_')[:-1])
            # rename spike time columns so that data is retained for each of the seperate trials
            rec_data = rec_data.rename(columns={'spikeT':rec_type+'_spikeT', 'spikeTraw':rec_type+'_spikeTraw','rate':rec_type+'_rate','n_spikes':rec_type+'_n_spikes'})
            # add a column for which fm recording should be prefered
            for key,val in goodlightrecs.items():
                if key in rec_data['session'].iloc[0]:
                    rec_data['best_light_fm'] = val
            for key,val in gooddarkrecs.items():
                if key in rec_data['session'].iloc[0]:
                    rec_data['best_dark_fm'] = val
            # get column names
            column_names = list(session_data.columns.values) + list(rec_data.columns.values)
            # new columns for same unit within a session
            session_data = pd.concat([session_data, rec_data],axis=1,ignore_index=True)
            # add the list of column names from all sessions plus the current recording
            session_data.columns = column_names
            # remove duplicate columns (i.e. shared metadata)
            session_data = session_data.loc[:,~session_data.columns.duplicated()]
        # add probe name as new col
        animal = goodsessions[ind]
        ellipse_json_path = find('*fm_eyecameracalc_props.json', animal)
        if ellipse_json_path != []:
            with open(ellipse_json_path[0]) as f:
                ellipse_fit_params = json.load(f)
            session_data['best_ellipse_fit_m'] = ellipse_fit_params['regression_m']
            session_data['best_ellipse_fit_r'] = ellipse_fit_params['regression_r']
        else:
            pass
        # add probe name
        session_data['probe_name'] = probenames_for_goodsessions[ind]
        # replace LFP power profile estimate of laminar depth with value entered into spreadsheet
        manual_depth_entry = layer5_depth_for_goodsessions[ind]
        if 'hf1_wn_lfp_layer5_centers' in session_data.columns.values:
            if type(session_data['hf1_wn_lfp_layer5_centers'].iloc[0]) != float and type(manual_depth_entry) != float and manual_depth_entry not in ['?','','FALSE',False]:
                num_sh = len(session_data['hf1_wn_lfp_layer5_centers'].iloc[0])
                for i, row in session_data.iterrows():
                    session_data.at[i, 'hf1_wn_lfp_layer5_centers'] = list(np.ones([num_sh]).astype(int)*int(manual_depth_entry))
        ind += 1
        # new rows for units from different mice or sessions
        all_data = pd.concat([all_data, session_data], axis=0)
    fm2_light = [c for c in all_data.columns.values if 'fm2_light' in c]
    fm1_dark = [c for c in all_data.columns.values if 'fm1_dark' in c]
    dark_dict = dict(zip(fm1_dark, [i.replace('fm1_dark', 'fm_dark') for i in fm1_dark]))
    light_dict = dict(zip(fm2_light, [i.replace('fm2_light_', 'fm1_') for i in fm2_light]))
    all_data = all_data.rename(dark_dict, axis=1).rename(light_dict, axis=1)
    # drop empty data without session name
    for ind, row in all_data.iterrows():
        if type(row['session']) != str:
            all_data = all_data.drop(ind, axis=0)
    # combine columns where one property of the unit is spread across multiple columns because of renaming scheme
    for col in list(all_data.loc[:,all_data.columns.duplicated()].columns.values):
        all_data[col] = all_data[col].iloc[:,0].combine_first(all_data[col].iloc[:,1])
    # and drop the duplicates that have only partial data (all the data will now be in another column)
    all_data = all_data.loc[:,~all_data.columns.duplicated()]
    return all_data

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