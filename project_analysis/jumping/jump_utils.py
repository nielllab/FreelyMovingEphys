"""
jump_utils.py
"""
import argparse, json, sys, os, shutil
import xarray as xr
import pandas as pd
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import traceback

from util.log import log
from util.paths import find
from util.aux_funcs import nanxcorr
from project_analysis.jumping.Dec2020.analyze_jump import jump_gaze_trace, animated_gaze_plot

def organize_dirs(jump_config_path):
    """
    first func to run
    organize directories to match expected structure of freely moving ephys preprocessing
    """
    # open config file
    with open(jump_config_path, 'r') as fp:
        config = json.load(fp)
    # set arguments as variables
    path_in = config['path_to_raw_data']
    path_out = config['analysis_save_dir']
    # get all the video and timestamp paths
    file_list = find('*.avi', path_in) + find('*.csv', path_in)
    # find all the text files that contain recording metadata
    text_file_list = find('*.txt', path_in)
    #iterate through videos
    for file in file_list:
        print('copying ' + file)
        # get the name without the path
        file_name = os.path.split(file)[-1]
        # for videos...
        if 'BonsaiTS' not in file:
            base = '_'.join(file_name.split('_')[:-2]) # file name without camera type, etc.
            cam = file_name.split('_')[-2:-1][0] # camera type (i.e. REYE, LEYE, TOP, etc.)
            num = (file_name.split('_')[-1]).split('.')[0] # jump recording number
            
            new_dir = '_'.join([base, num]) # save directory for this video
            new_name = '_'.join([base, num, cam+'.avi']) # file name to use when saving copy
            save_path = os.path.join(path_out, new_dir) # full path to save
            # create direcory if it doesn't already exist
            if os.path.exists(save_path) is False:
                os.makedirs(save_path)
            # for txt metadata
            for meta in text_file_list:
                # only handle txt files associated with this video
                if base in meta:
                    old_meta_name = os.path.split(meta)[-1]# file name that it used before
                    old_meta_noext = os.path.splitext(old_meta_name)[0] # above, but without the file extension
                    # there are two types of txt files: one with metadata alone, and one with timestamp info
                    # first, find the timestamp txt files and create a new 'ready_meta' file name
                    if 'vidclip_ts' in old_meta_noext:
                        base_meta = '_'.join(old_meta_noext.split('_')[:-3])
                        ready_meta = '_'.join([base_meta, num, '_'.join(old_meta_noext.split('_')[-3:-1])])
                    # next, find the plain metadata txt file and create it's 'ready_meta' file name
                    elif 'vidclip_ts' not in old_meta_noext:
                        base_meta = '_'.join(old_meta_noext.split('_')[:-1])
                        ready_meta = '_'.join([base_meta, num, ''.join(old_meta_noext.split('_')[-1])])
                    # add extension; turn into a full path
                    new_meta_name = ready_meta + '.txt'
                    metadata_save_full = os.path.join(save_path, new_meta_name)
                    # make a copy of the metadata file
                    shutil.copyfile(meta, metadata_save_full)
                    print('saved ' + meta)
            # make the full save path for the video file
            save_path_full = os.path.join(save_path, new_name)
            # save a copy of that video in the new directory
            shutil.copyfile(file, save_path_full)
            print('saved ' + file)
        # for timestamps...
        if 'BonsaiTS' in file:
            # get out filename info
            base = '_'.join(file_name.split('_')[:-3]) # file name without camera type, etc.
            cam = file_name.split('_')[-3:-2][0] # camera type
            num = (file_name.split('_')[-2]) # jump recording number
            bon = (file_name.split('_')[-1]).split('.')[0] # bonsai label
            # set up the new directory
            new_dir = '_'.join([base, num]) # save directory
            new_name = '_'.join([base, num, cam, bon+'.csv']) # file name
            save_path = os.path.join(path_out, new_dir, new_name) # full path
            # copy the file with new name
            shutil.copyfile(file, save_path)
            print('saved ' + file)

def jump_cc(REye_ds, LEye_ds, top_ds, side_ds, time, meta, config, timebin):
    """
    get figures and process data for individual jump recordings
    """
    # handle jump timing metadata
    jump_num = config['recording_name'].split('_')[-1].lstrip('0') # get the jump number without preceding 0
    vals = [] # the values in the dictionary for this jump
    cam_points = [] # the entries in time metadata dictionary
    for cam_point in time:
        cam_values = time[cam_point]
        vals.append(cam_values[str(int(jump_num)-1)])
        cam_points.append(cam_point)
    time_dict = {cam_points[i] : vals[i] for i in range(len(cam_points))} # make the dictionary for only this jump
    # then do it for the recording's general metadata and merge the time metadata in
    trial_info = meta['trial_info']
    jump_info = trial_info[int(jump_num)-1]
    jump_info.update(time_dict)

    # open pdf file to save plots in
    pdf = PdfPages(os.path.join(config['trial_head'], (config['recording_name'] + '_' + timebin + '_jump_cc.pdf')))

    # organize data
    REye = REye_ds.REYE_ellipse_params
    LEye = LEye_ds.LEYE_ellipse_params
    head_pitch = side_ds.SIDE_theta

    # zero-center theta and phi for each eye, and flip sign of phi
    RTheta = np.rad2deg(REye.sel(ellipse_params='theta')) - np.rad2deg(np.nanmedian(REye.sel(ellipse_params='theta')))
    RPhi = (np.rad2deg(REye.sel(ellipse_params='phi')) - np.rad2deg(np.nanmedian(REye.sel(ellipse_params='phi'))))
    LTheta = np.rad2deg(LEye.sel(ellipse_params='theta')) -  np.rad2deg(np.nanmedian(LEye.sel(ellipse_params='theta')))
    LPhi = (np.rad2deg(LEye.sel(ellipse_params='phi')) - np.rad2deg(np.nanmedian(LEye.sel(ellipse_params='phi'))))

    # zero-center head pitch, and get rid of wrap-around effect (mod 360)
    pitch = np.rad2deg(head_pitch)
    pitch = ((pitch+360) % 360)
    pitch = pitch - np.nanmean(pitch) # have to mean center as last step so that it's comparable on plots--is it okay for this to be different in video plotting? I would think so

    # interpolate over eye paramters to match head pitch
    RTheta_interp = RTheta.interp_like(pitch, method='linear')
    RPhi_interp = RPhi.interp_like(pitch, method='linear')
    LTheta_interp = LTheta.interp_like(pitch, method='linear')
    LPhi_interp = LPhi.interp_like(pitch, method='linear')

    # eye divergence (theta)
    div = (RTheta_interp - LTheta_interp) * 0.5
    # gaze (mean theta of eyes)
    gaze_th = (RTheta_interp + LTheta_interp) * 0.5
    # gaze (mean phi of eyes)
    gaze_phi = (RPhi_interp + LPhi_interp) * 0.5

    # correct lengths when off
    pitch_len = len(pitch.values); gaze_th_len = len(gaze_th.values); div_len = len(div.values); gaze_phi_len = len(gaze_phi.values)
    min_len = np.min([pitch_len, gaze_th_len, div_len, gaze_phi_len])
    max_len = np.max([pitch_len, gaze_th_len, div_len, gaze_phi_len])
    if max_len != min_len:
        pitch = pitch.isel(frame=range(0,min_len))
        gaze_th = gaze_th.isel(frame=range(0,min_len))
        div = div.isel(frame=range(0,min_len))
        gaze_phi = gaze_phi.isel(frame=range(0,min_len))

    # calculate xcorrs
    th_gaze, lags = nanxcorr(pitch.values, gaze_th.values, 30)
    th_div, lags = nanxcorr(pitch.values, div.values, 30)
    th_phi, lags = nanxcorr(pitch.values, gaze_phi.values, 30)

    # make an xarray of this trial's data to be used in pooled analysis
    trial_outputs = pd.DataFrame([pitch, gaze_th, div, gaze_phi,th_gaze,th_div,th_phi]).T
    trial_outputs.columns = ['head_pitch','mean_eye_th','eye_th_div','mean_eye_phi','th_gaze','th_div','th_phi']
    trial_xr = xr.DataArray(trial_outputs, dims=['frame','jump_params'])
    trial_xr.attrs['jump_metadata'] = str(jump_info)

    # plots
    plt.figure()
    plt.title(config['recording_name'])
    plt.ylabel('deg'); plt.xlabel('frames')
    plt.plot(pitch); plt.plot(gaze_th); plt.plot(div); plt.plot(gaze_phi)
    plt.legend(['head_pitch', 'eye_theta','eye_divergence','eye_phi'])
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.subplots(2,1)
    plt.subplot(211)
    plt.plot(LTheta_interp)
    plt.plot(RTheta_interp)
    plt.plot(gaze_th)
    plt.title('theta interp_like pitch')
    plt.legend(['left','right','mean'])
    plt.subplot(212)
    plt.plot(LPhi_interp)
    plt.plot(RPhi_interp)
    plt.plot(gaze_phi)
    plt.title('phi interp_like pitch')
    plt.legend(['left','right','mean'])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.title('head_pitch xcorr')
    plt.plot(lags, th_gaze); plt.plot(lags, th_div); plt.plot(lags, th_phi)
    plt.legend(['gaze', 'div', 'phi'])
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.ylabel('eye div deg'); plt.xlabel('head pitch deg')
    plt.plot([-30,30],[30,-30], 'r:')
    plt.xlim([-30,30]); plt.ylim([-30,30])
    plt.scatter(pitch, div)
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.ylabel('eye phi deg'); plt.xlabel('head pitch deg')
    plt.plot([-30,30],[-30,30], 'r:')
    plt.xlim([-30,30]); plt.ylim([-30,30])
    plt.scatter(pitch, gaze_phi)
    pdf.savefig()
    plt.close()

    # if config['save_avi_vids'] is True:

    # make an animated plot of these parameters
    if config['plot_avi_vids'] is True:
        print('saving animated plots')
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
        ax1.plot(pitch, 'b-'); ax1.set_title('pitch')
        ax2.plot(gaze_th, 'b-'); ax2.set_title('mean th')
        ax3.plot(gaze_phi, 'b-'); ax3.set_title('mean phi')
        ax4.plot(div, 'b-'); ax4.set_title('th div')
        plt.tight_layout()
        pdf.savefig()

        ani_save_path = os.path.join(config['trial_head'], (config['recording_name'] + '_params_animation.avi'))
        writer = FFMpegWriter(fps=60, bitrate=-1)
        with writer.saving(fig, ani_save_path, 150):
            for t in tqdm(range(0,len(pitch))):
                ln1 = ax1.vlines(t,-40,40)
                ln2 = ax2.vlines(t,-40,40)
                ln3 = ax3.vlines(t,-40,40)
                ln4 = ax4.vlines(t,-40,40)
                writer.grab_frame()
                ln1.remove(); ln2.remove(); ln3.remove(); ln4.remove()

            # plt.close('all')
    pdf.close()
    return trial_xr

def mean_within_animal(data):
    time_sel = data.sel(frame=range(60))
    data_mean = time_sel.mean(dim='variable',skipna=True)
    out = []
    for var in data_mean:
        out.append(list(data_mean[var].values))
    return np.array(out)

def jump_hist(div):
    all_bins = np.arange(-25,26)
    all_hist = np.zeros([np.size(div, 0), len(all_bins)-1])
    for jump_num in range(np.size(div, 0)):
        hist, bin_edges = np.histogram(div[jump_num].dropna(dim='frame'), bins=np.arange(-25,26), density=True)
        all_hist[jump_num] = hist
    return all_hist

def jump_hist1(div, axis=0):
    all_bins = np.arange(-25,26)
    all_hist = np.zeros([np.size(div, axis), len(all_bins)-1])
    for jump_num in range(np.size(div, axis)):
        hist, bin_edges = np.histogram(div[jump_num], bins=np.arange(-25,26), density=True)
        all_hist[jump_num] = hist
    return all_hist

def mean_within_animal1(data):
    data_mean = data.mean(dim='jump',skipna=True)
    out = []
    for var in range(np.size(data_mean,0)):
        out.append(list(data_mean[var]))
    return np.array(out)

# make plots using the pooled jumping data
def pooled_jump_analysis(pooled, config, bin_name):

    pdf = PdfPages(os.path.join(config['analysis_save_dir'], bin_name+'_pooled_jump_plots.pdf'))
    
    # convert to dataarray so that indexing can be done accross recordings
    # this is only needed if there's more than one trial read in, so a try/except is used
    try:
        pooled_da = pooled.to_array()
    except AttributeError:
        pooled_da = pooled
    # then, get data out for each parameter
    all_pitch = pooled_da.sel(jump_params='head_pitch').values
    all_phi = - pooled_da.sel(jump_params='mean_eye_phi').values
    all_div = pooled_da.sel(jump_params='eye_th_div').values
    all_th_gaze = pooled_da.sel(jump_params='th_gaze', frame=range(60)).values
    all_th_div = pooled_da.sel(jump_params='th_div', frame=range(60)).values
    all_th_phi = - pooled_da.sel(jump_params='th_phi', frame=range(60)).values
    lags = range(-30, 30)
    
    dwnspl = 100
    
    # head theta, phi
    plt.figure(figsize=(5,5))
    plt.plot(all_pitch[::dwnspl], all_phi[::dwnspl], 'k.')
    plt.xlabel('head pitch'); plt.ylabel('phi')
    plt.xlim([-60,60]); plt.ylim([-30,30])
    plt.plot([-60,60],[60,-60],':',color=[0.5,0.5,0.5])
    pdf.savefig()
    plt.close()
    # head theta, eye theta divergence
    plt.figure(figsize=(5,5))
    plt.plot(all_pitch[::dwnspl], all_div[::dwnspl], 'k.')
    plt.xlabel('head pitch'); plt.ylabel('eye theta div')
    plt.xlim([-60,60]); plt.ylim([-30,30])
    plt.plot([-60,60],[60,-60],':',color=[0.5,0.5,0.5])
    pdf.savefig()
    plt.close()

    # organize data for head xcorr with th, divergence, and phi (grouped within each animal's jumps)
    try:
        animal_names = list(set([i.split('_')[1] for i in list(pooled_da.variables.values)]))
    except:
        animal_names = list(set([i.split('_')[1] for i in list(pooled_da['variable'].values)]))
    for var in pooled:
        pooled[var].attrs['animal'] = var.split('_')[1]

    pool_by_animal = xr.Dataset()
    for animal_name in animal_names:
        this_animal = pooled.filter_by_attrs(animal=animal_name).to_array()
        this_animal.name = animal_name
        pool_by_animal = xr.merge([pool_by_animal, this_animal])

    ani_th_gaze = mean_within_animal(pool_by_animal.sel(jump_params='th_gaze'))
    ani_th_div = mean_within_animal(pool_by_animal.sel(jump_params='th_div'))
    ani_th_phi = - mean_within_animal(pool_by_animal.sel(jump_params='th_phi'))

    # plot head xcorr
    y1 = np.mean(ani_th_gaze,0)
    err1 = np.std(np.array(ani_th_gaze,dtype=np.float64),0)/np.sqrt(np.size(ani_th_gaze,0))
    y2 = np.mean(ani_th_div,0)
    err2 = np.std(np.array(ani_th_div,dtype=np.float64),0)/np.sqrt(np.size(ani_th_div,0))
    y3 = np.mean(ani_th_phi,0)
    err3 = np.std(np.array(ani_th_phi,dtype=np.float64),0)/np.sqrt(np.size(ani_th_phi,0))
    plt.figure(figsize=(6,6))
    plt.plot(lags, y1, 'C1')
    plt.fill_between(lags, y1-err1, y1+err1, color='C1', alpha=0.3)
    plt.plot(lags, y2, 'C2')
    plt.fill_between(lags, y2-err2, y2+err2, color='C2', alpha=0.3)
    plt.plot(lags, y3, 'k')
    plt.fill_between(lags, y3-err3, y3+err3, color='k', alpha=0.3)
    plt.ylim([-1,1]); plt.ylabel('correlation'); plt.title('xcorr with head pitch')
    plt.legend(['mean theta', 'theta divergence', 'mean phi'])
    pdf.savefig()
    plt.close()

    pdf.close()

    return pool_by_animal

def jump_analysis(config):
    # initialize logger
    logf = log(os.path.join(config['analysis_save_dir'],'jump_analysis_log.csv'),name=['jump'])
    # find all the text files that contain recording metadata
    text_file_list = find('*.txt', config['analysis_save_dir'])
    # remove vidclip files from the metadata list
    vidclip_file_list = find('*vidclip*.txt', config['analysis_save_dir'])
    for x in vidclip_file_list:
        text_file_list.remove(x)
    # iterate through the text files
    trial_count = 0
    for trial_path in sorted(text_file_list):
        trial_count = trial_count + 1
        # read the trial metadata data in
        with open(trial_path) as f:
            trial_contents = f.read()
        trial_metadata = json.loads(trial_contents)
        # get the name of the file
        trial_path_noext = os.path.splitext(trial_path)[0]
        head, trial_name_long = os.path.split(trial_path_noext)
        trial_name = '_'.join(trial_name_long.split('_')[:-1])
        config['recording_name'] = trial_name; config['trial_head'] = head
        print('analyzing '+config['recording_name'])
        # get the metadata out of vidclip text file
        for time_text_path in vidclip_file_list:
            if trial_name in time_text_path:
                with open(time_text_path) as f:
                    time_txt = f.read()
        time_dict = json.loads(time_txt)
        bin_group_keys = ['complete', 'early', 'jumpprep', 'late']
        # find the matching sets of .nc files produced during preprocessing
        for bin_group in bin_group_keys:
            if bin_group == 'complete':
                try:
                    leye = xr.open_dataset([i for i in find((trial_name + '*_Leye.nc'), head) if 'early' not in i and 'jumpprep' not in i and 'late' not in i][0])
                    reye = xr.open_dataset([i for i in find((trial_name + '*_Reye.nc'), head) if 'early' not in i and 'jumpprep' not in i and 'late' not in i][0])
                    side = xr.open_dataset([i for i in find((trial_name + '*_side.nc'), head) if 'early' not in i and 'jumpprep' not in i and 'late' not in i][0])
                    top = xr.open_dataset([i for i in find((trial_name + '*_Top.nc'), head) if 'early' not in i and 'jumpprep' not in i and 'late' not in i][0])
                except Exception as e:
                    logf.log([trial_path, traceback.format_exc()],PRINT=False)
                    continue
            else:
                try:
                    leye = xr.open_dataset([i for i in find((trial_name + '*_Leye.nc'), head) if bin_group in i][0])
                    reye = xr.open_dataset([i for i in find((trial_name + '*_Reye.nc'), head) if bin_group in i][0])
                    side = xr.open_dataset([i for i in find((trial_name + '*_side.nc'), head) if bin_group in i][0])
                    top = xr.open_dataset([i for i in find((trial_name + '*_Top.nc'), head) if bin_group in i][0])
                except Exception as e:
                    logf.log([trial_path, traceback.format_exc()],PRINT=False)
                    continue
            try:
                side_vid = find((trial_name + '*Side*.avi'), head)
                top_vid = find((trial_name + '*Top*.avi'), head)
                leye_vid = find((trial_name + '*LEYE*.avi'), head)
                reye_vid = find((trial_name + '*REYE*.avi'), head)
                for x in side_vid:
                    if 'plot' in x:
                        side_vid.remove(x)
                for x in top_vid:
                    if 'plot' in x:
                        top_vid.remove(x)
                for x in leye_vid:
                    if 'plot' in leye_vid or 'unflipped' in leye_vid:
                        leye_vid.remove(x)
                for x in reye_vid:
                    if 'plot' in reye_vid or 'unflipped' in reye_vid:
                        reye_vid.remove(x)
                side_vid = side_vid[0]
                top_vid = top_vid[0]
                leye_vid = leye_vid[0]
                reye_vid = reye_vid[0]
                # correlation figures
                trial_cc_data = jump_cc(reye, leye, top, side, time_dict, trial_metadata, config, bin_group)
                trial_cc_data.name = config['recording_name']
                # plot over video
                if config['plot_avi_vids'] is True and bin_group=='complete':
                    print('plotting jump gaze for side view of ' + config['recording_name'])
                    jump_gaze_trace(reye, leye, top, side, side_vid, config)
                    print('plotting videos with animated plots for ' + config['recording_name'])
                    animated_gaze_plot(reye, leye, top, side, side_vid, leye_vid, reye_vid, top_vid, config)
                if bin_group == 'complete':
                    try:
                        pooled_data = xr.merge([pooled_data, trial_cc_data])
                    except UnboundLocalError:
                        pooled_data = trial_cc_data.copy()
                elif bin_group == 'early':
                    try:
                        early_pooled_data = xr.merge([early_pooled_data, trial_cc_data])
                    except UnboundLocalError:
                        early_pooled_data = trial_cc_data.copy()
                elif bin_group == 'jumpprep':
                    try:
                        jumpprep_pooled_data = xr.merge([jumpprep_pooled_data, trial_cc_data])
                    except UnboundLocalError:
                        jumpprep_pooled_data = trial_cc_data.copy()
                elif bin_group == 'late':
                    try:
                        late_pooled_data = xr.merge([late_pooled_data, trial_cc_data])
                    except UnboundLocalError:
                        late_pooled_data = trial_cc_data.copy()
            except Exception as e:
                logf.log([trial_path, traceback.format_exc()],PRINT=False)
            print('done with trial '+str(trial_count)+' of '+str(len(text_file_list)))
    print('saving pooled data at ' + config['analysis_save_dir'])
    # save out an xarray of pooled data
    pooled_data.to_netcdf(os.path.join(config['analysis_save_dir'], 'pooled_jump_data.nc'))
    early_pooled_data.to_netcdf(os.path.join(config['analysis_save_dir'], 'early_pooled_jump_data.nc'))
    jumpprep_pooled_data.to_netcdf(os.path.join(config['analysis_save_dir'], 'jumpprep_pooled_jump_data.nc'))
    late_pooled_data.to_netcdf(os.path.join(config['analysis_save_dir'], 'late_pooled_jump_data.nc'))
    print('making plots of pooled data for all trials')
    # make a pdf of pooled data
    combined_pool_by_animal = pooled_jump_analysis(pooled_data, config, 'combined')
    early_pool_by_animal = pooled_jump_analysis(early_pooled_data, config, 'early')
    jumpprep_pool_by_animal = pooled_jump_analysis(jumpprep_pooled_data, config, 'jumpprep')
    late_pool_by_animal = pooled_jump_analysis(late_pooled_data, config, 'late')
    print('done analyzing ' + str(len(text_file_list)) + ' trials')

    combined_pool_by_animal.to_netcdf(os.path.join(config['analysis_save_dir'], 'pooled_jump_by_animal.nc'))
    early_pool_by_animal.to_netcdf(os.path.join(config['analysis_save_dir'], 'early_pooled_jump_by_animal.nc'))
    jumpprep_pool_by_animal.to_netcdf(os.path.join(config['analysis_save_dir'], 'jumpprep_pooled_jump_by_animal.nc'))
    late_pool_by_animal.to_netcdf(os.path.join(config['analysis_save_dir'], 'late_pooled_jump_by_animal.nc'))

    pdf = PdfPages(os.path.join(config['analysis_save_dir'], 'pooled_hists_by_animal.pdf'))

    for measurment_name in ['eye_th_div', 'head_pitch']:
        div_early = early_pool_by_animal.sel(jump_params=measurment_name)
        div_jumpprep = jumpprep_pool_by_animal.sel(jump_params=measurment_name)
        early = div_early.drop('variable').rename({'variable':'jump'}).to_array()
        jumpprep = div_jumpprep.drop('variable').rename({'variable':'jump'}).to_array()
        early_bins = jump_hist1(early)
        jumpprep_bins = jump_hist1(jumpprep)
        early = mean_within_animal1(early)
        jumpprep = mean_within_animal1(jumpprep)

        lags = np.arange(-25,25)
        plt.subplots(1,2, figsize=(9,4))
        plt.subplot(1,2,1)
        plt.title('early')
        plt.plot(lags, early_bins.T, alpha=0.3)
        plt.ylabel('proportion of time'); plt.xlabel(measurment_name)
        plt.subplot(1,2,2)
        plt.title('jumpprep')
        plt.ylabel('proportion of time')
        plt.xlabel(measurment_name)
        plt.plot(lags, jumpprep_bins.T, alpha=0.3)
        pdf.savefig()
        plt.close()

        axisplot = 0
        y1 = np.mean(early_bins,axisplot)
        err1 = np.std(np.array(early_bins,dtype=np.float64),axisplot)/np.sqrt(np.size(early_bins,axisplot))
        y2 = np.mean(jumpprep_bins,axisplot)
        err2 = np.std(np.array(jumpprep_bins,dtype=np.float64),axisplot)/np.sqrt(np.size(jumpprep_bins,axisplot))

        lags = np.arange(-25,25)
        plt.figure(figsize=(4,4))
        plt.plot(lags, y1)
        plt.fill_between(lags, y1-err1, y1+err1, alpha=0.3)
        plt.plot(lags, y2)
        plt.fill_between(lags, y2-err2, y2+err2, alpha=0.3)
        plt.ylabel('proportion of time'); plt.xlabel(measurment_name)
        plt.legend(['early', 'jump prep'])
        pdf.savefig()
        plt.close()

    pdf.close()

def split_nc_into_timebins(config):
    main_path = config['analysis_save_dir']
    prejump_window = 60 * 5 # five seconds before jump (60fps)
    nc_list = find('*.nc',main_path)
    nc_list = [i for i in nc_list if 'pooled_jump_data' not in i]
    nc_list1 = [i for i in nc_list if len(os.path.split(i)[1].split('_'))==4]
    for i in tqdm(range(len(nc_list1))):
        try:
            # open xarray
            nc_filepath = nc_list1[i]
            data = xr.open_dataset(nc_filepath)
            # open text file of time bins for each jump
            base_name = '_'.join(os.path.splitext(os.path.split(nc_filepath)[1])[0].split('_')[:-1])
            times_path = find('*'+base_name+'*eyecams_vidclip.txt', main_path)[0]
            times = dict(np.ndenumerate(np.loadtxt(times_path, dtype=str)))
            times = eval(times[list(times)[0]])
            jump_num = str(int(base_name.split('_')[-1].strip('0'))-1)
            base_cam = 'Side'
            start = slice(0, (times[base_cam+'_Jump'][jump_num]-times[base_cam+'_Start'][jump_num])-prejump_window)
            prejump = slice((times[base_cam+'_Jump'][jump_num]-times[base_cam+'_Start'][jump_num])-prejump_window,(times[base_cam+'_Jump'][jump_num]-times[base_cam+'_Start'][jump_num]))
            stop = slice(times[base_cam+'_Jump'][jump_num]-times[base_cam+'_Start'][jump_num], times[base_cam+'_End'][jump_num]-times[base_cam+'_Start'][jump_num])
            # index using these times
            early = data.sel(frame=start)
            jumpprep = data.sel(frame=prejump)
            late = data.sel(frame=stop)
            # save those as new nc files
            split_name = os.path.split(nc_filepath)[1].split('_')
            path_out = os.path.split(nc_filepath)[0]
            early.to_netcdf(os.path.join(path_out, '_'.join(split_name[:-1]+['early']+[split_name[-1]])))
            jumpprep.to_netcdf(os.path.join(path_out, '_'.join(split_name[:-1]+['jumpprep']+[split_name[-1]])))
            late.to_netcdf(os.path.join(path_out, '_'.join(split_name[:-1]+['late']+[split_name[-1]])))
        except Exception as e:
            print(e)