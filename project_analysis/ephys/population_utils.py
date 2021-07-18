"""
population_utils.py
"""
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils.ephys import load_ephys

def modulation_index(tuning, zerocent=True, lbound=1, ubound=-2):
    """
    get modulation index for a tuning curve
    selects lbound and ubound so that more noisy extremes can be ignored
    INPUTS
        tuning: tuning curve over bins
        zerocent: bool, is the data zero centered (i.e. should a modind be calcualted in two directions or one?)
        lbound: index of lower bound to use
        ubound: index of upper bound to use
    """
    # drop nans from tuning curve
    tuning = tuning[~np.isnan(tuning)]
    # if bins start at zero and increase in one direction only
    if zerocent is False:
        return np.round((tuning[ubound] - tuning[lbound]) / (tuning[ubound] + tuning[lbound]), 3)
    # if bins increase from zero in both the positive and negative direction
    elif zerocent is True:
        r0 = np.nanmean(tuning[4:6])
        modind_neg = np.round((tuning[lbound] - r0) / (tuning[lbound] + r0), 3)
        modind_pos = np.round((tuning[ubound] - r0) / (tuning[ubound] + r0), 3)
        return [modind_neg, modind_pos]

def saccade_modulation_index(trange, saccavg):
    """
    get the modulation index of eye movement at t=0 and t=100ms
    INPUTS
        trange: time range over which average saccades are aligned
        saccavg: average saccade in a given direction (left or right)
    """
    t0ind = (np.abs(trange-0)).argmin()
    t100ind = t0ind+4
    baseline = np.nanmean(saccavg[0:int(t100ind-((1/4)*t100ind))])
    r0 = np.round((saccavg[t0ind] - baseline) / (saccavg[t0ind] + baseline), 3)
    r100 = np.round((saccavg[t100ind] - baseline) / (saccavg[t100ind] + baseline), 3)
    return r0, r100

def make_unit_summary(df, savepath):
    """
    make a pdf summarizing each unit, where each unit is one page of panels
    also saves out an updated .json of data including the calculated modualation indexes, etc.
    INPUTS
        df: dataframe of all units, where each unit is an index
        savepath: where to save pdf and updated .json
    OUTPUTS
        newdf: updated dataframe of units
    """
    # set up pdf
    pdf = PdfPages(os.path.join(savepath, 'unit_summary_'+datetime.today().strftime('%m%d%y')+'.pdf'))
    samprate = 30000
    # set up new h5 file to save out including new metrics
    newdf = df.copy().reset_index()
    print('num units = ' + str(len(df)))
    # iterate through units
    for index, row in tqdm(df.iterrows()):
        lightfm = row['best_light_fm']
        darkfm = row['best_light_fm']
        unitfig = plt.figure(constrained_layout=True, figsize=(50,45))
        spec = gridspec.GridSpec(ncols=5, nrows=10, figure=unitfig)
        # blank title panel
        unitfig_title = unitfig.add_subplot(spec[0,0])
        unitfig_title.axis('off')
        unitfig_title.annotate(str(row['session'])+'_unit'+str(row['index']),size=15, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=20)
        try:
            # waveform
            unitfig_wv = unitfig.add_subplot(spec[0, 1])
            wv = row['waveform']
            unitfig_wv.plot(np.arange(len(wv))*1000/samprate,wv); unitfig_wv.set_ylabel('millivolts'); unitfig_wv.set_xlabel('msec')
            unitfig_wv.set_title(row['KSLabel']+' cont='+str(np.round(row['ContamPct'],3)), fontsize=20)
        except:
            pass
        try:
            # contrast response function
            unitfig_crf = unitfig.add_subplot(spec[0, 2])
            crange = row['hf1_wn_c_range']
            var_cent = row['hf1_wn_crf_cent']
            tuning = row['hf1_wn_crf_tuning']
            tuning_err = row['hf1_wn_crf_err']
            unitfig_crf.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            modind = modulation_index(tuning, zerocent=False)
            unitfig_crf.set_title('WN CRF\nmod.ind.='+str(modind), fontsize=20)
            unitfig_crf.set_xlabel('contrast a.u.'); unitfig_crf.set_ylabel('sp/sec')
            unitfig_crf.set_ylim(0,np.nanmax(tuning[:]*1.2))#; unitfig_crf.set_xlim([0,1])
            newdf.at[row['index'], 'hf1_wn_crf_modind'] = modind
        except:
            pass
        try:
            # psth gratings
            unitfig_grat_psth = unitfig.add_subplot(spec[0, 3])
            lower = -0.5; upper = 1.5; dt = 0.1
            bins = np.arange(lower,upper+dt,dt)
            psth = row['hf3_gratings_grating_psth']
            unitfig_grat_psth.plot(bins[0:-1]+ dt/2,psth)
            unitfig_grat_psth.set_title('gratings psth', fontsize=20)
            unitfig_grat_psth.set_xlabel('time'); unitfig_grat_psth.set_ylabel('sp/sec')
            unitfig_grat_psth.set_ylim([0,np.nanmax(psth)*1.2])
        except:
            pass
        try:
            # LFP trace relative to center of layer 4
            unitfig_lfp = unitfig.add_subplot(spec[0, 4])
            if np.size(row['hf4_revchecker_revchecker_mean_resp_per_ch'],0) == 64:
                shank_channels = [c for c in range(np.size(row['hf4_revchecker_revchecker_mean_resp_per_ch'], 0)) if int(np.floor(c/32)) == int(np.floor(int(row['ch'])/32))]
                whole_shank = row['hf4_revchecker_revchecker_mean_resp_per_ch'][shank_channels]
                shank_num = [0 if np.max(shank_channels) < 40 else 1][0]
                colors = plt.cm.jet(np.linspace(0,1,32))
                for ch_num in range(len(shank_channels)):
                    unitfig_lfp.plot(whole_shank[ch_num], color=colors[ch_num], alpha=0.1, linewidth=1) # all other channels
                unitfig_lfp.plot(whole_shank[row['hf4_revchecker_layer4center'][shank_num]], color=colors[row['hf4_revchecker_layer4center'][shank_num]], label='layer4', linewidth=4) # layer 4
            elif np.size(row['hf4_revchecker_revchecker_mean_resp_per_ch'],0) == 16:
                whole_shank = row['hf4_revchecker_revchecker_mean_resp_per_ch']
                colors = plt.cm.jet(np.linspace(0,1,16))
                shank_num = 0
                for ch_num in range(16):
                    unitfig_lfp.plot(row['hf4_revchecker_revchecker_mean_resp_per_ch'][ch_num], color=colors[ch_num], alpha=0.3, linewidth=1) # all other channels
                unitfig_lfp.plot(whole_shank[row['hf4_revchecker_layer4center']], color=colors[row['hf4_revchecker_layer4center']], label='layer4', linewidth=1) # layer 4
            elif np.size(row['hf4_revchecker_revchecker_mean_resp_per_ch'],0) == 128:
                shank_channels = [c for c in range(np.size(row['hf4_revchecker_revchecker_mean_resp_per_ch'], 0)) if int(np.floor(c/32)) == int(np.floor(int(row['ch'])/32))]
                whole_shank = row['hf4_revchecker_revchecker_mean_resp_per_ch'][shank_channels]
                shank_num = int(np.floor(int(row['ch'])/32))
                colors = plt.cm.jet(np.linspace(0,1,32))
                for ch_num in range(len(shank_channels)):
                    unitfig_lfp.plot(whole_shank[ch_num], color=colors[ch_num], alpha=0.1, linewidth=1) # all other channels
                unitfig_lfp.plot(whole_shank[row['hf4_revchecker_layer4center'][shank_num]], color=colors[row['hf4_revchecker_layer4center'][shank_num]], label='layer4', linewidth=4) # layer 4
            else:
                print('unrecognized probe count in LFP plots during unit summary! index='+str(index))
            row['ch'] = int(row['ch'])
            unitfig_lfp.plot(row['hf4_revchecker_revchecker_mean_resp_per_ch'][row['ch']%32], color=colors[row['ch']%32], label='this channel', linewidth=4) # current channel
            depth_to_layer4 = 0 # could be 350um, but currently, everything will stay relative to layer4 since we don't know angle of probe & other factors
            if row['probe_name'] == 'DB_P64-8':
                ch_spacing = 25/2
            else:
                ch_spacing = 25
            if shank_num == 0:
                position_of_ch = int(row['hf4_revchecker_lfp_rel_depth'][0][row['ch']])
                newdf.at[row['index'], 'hf4_revchecker_ch_lfp_relative_depth'] = position_of_ch
                depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
                newdf.at[row['index'], 'hf4_revchecker_depth_from_layer4'] = depth_from_surface
                unitfig_lfp.set_title('ch='+str(row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
            elif shank_num == 1:
                position_of_ch = int(row['hf4_revchecker_lfp_rel_depth'][1][row['ch']-32])
                newdf.at[row['index'], 'hf4_revchecker_ch_lfp_relative_depth'] = position_of_ch
                depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
                newdf.at[row['index'], 'hf4_revchecker_depth_from_layer4'] = depth_from_surface
                unitfig_lfp.set_title('ch='+str(row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
            elif shank_num == 2:
                position_of_ch = int(row['hf4_revchecker_lfp_rel_depth'][1][row['ch']-64])
                newdf.at[row['index'], 'hf4_revchecker_ch_lfp_relative_depth'] = position_of_ch
                depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
                newdf.at[row['index'], 'hf4_revchecker_depth_from_layer4'] = depth_from_surface
                unitfig_lfp.set_title('ch='+str(row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
            elif shank_num == 3:
                position_of_ch = int(row['hf4_revchecker_lfp_rel_depth'][1][row['ch']-96])
                newdf.at[row['index'], 'hf4_revchecker_ch_lfp_relative_depth'] = position_of_ch
                depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
                newdf.at[row['index'], 'hf4_revchecker_depth_from_layer4'] = depth_from_surface
                unitfig_lfp.set_title('ch='+str(row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
            unitfig_lfp.legend(); unitfig_lfp.axvline(x=(0.1*30000), color='k', linewidth=1)
            unitfig_lfp.set_xticks(np.arange(0,18000,18000/8))
            unitfig_lfp.set_xticklabels(np.arange(-100,500,75))
            unitfig_lfp.set_xlabel('msec'); unitfig_lfp.set_ylabel('uvolts')
        except:
            pass
        try:
            # wn sta
            unitfig_wnsta = unitfig.add_subplot(spec[1, 0])
            wnsta = np.reshape(row['hf1_wn_spike_triggered_average'],tuple(row['hf1_wn_sta_shape']))
            wnstaRange = np.max(np.abs(wnsta))*1.2
            if wnstaRange<0.25:
                wnstaRange=0.25
            unitfig_wnsta.set_title('WN STA', fontsize=20)
            unitfig_wnsta.imshow(wnsta,vmin=-wnstaRange,vmax=wnstaRange,cmap='jet')
            unitfig_wnsta.axis('off')
        except:
            pass
        try:
            # wn stv
            unitfig_wnstv = unitfig.add_subplot(spec[1, 1])
            wnstv = np.reshape(row['hf1_wn_spike_triggered_variance'],tuple(row['hf1_wn_sta_shape']))
            unitfig_wnstv.imshow(wnstv,vmin=-1,vmax=1)
            unitfig_wnstv.set_title('WN STV', fontsize=20)
            unitfig_wnstv.axis('off')
        except:
            pass
        try:
            # wn eye movements
            unitfig_wnsaccavg = unitfig.add_subplot(spec[1, 2])
            trange = row['hf1_wn_trange']
            upsacc_avg = row['hf1_wn_upsacc_avg']; downsacc_avg = row['hf1_wn_downsacc_avg']
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_wnsaccavg.set_title('WN left/right saccades', fontsize=20)
            unitfig_wnsaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_wnsaccavg.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
            unitfig_wnsaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_wnsaccavg.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
            unitfig_wnsaccavg.legend(['right','left'], loc=1)
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            unitfig_wnsaccavg.set_ylim([0,maxval*1.2])
            newdf.at[row['index'], 'hf1_wn_upsacc_modind_t0'] = modind_right[0]; newdf.at[row['index'], 'hf1_wn_downsacc_modind_t0'] = modind_left[0]
            newdf.at[row['index'], 'hf1_wn_upsacc_modind_t100'] = modind_right[1]; newdf.at[row['index'], 'hf1_wn_downsacc_modind_t100'] = modind_left[1]
        except:
            pass
        try:
            # wn spike rate vs pupil radius
            unitfig_wnsrpupilrad = unitfig.add_subplot(spec[1, 3])
            var_cent = row['hf1_wn_spike_rate_vs_pupil_radius_cent']
            tuning = row['hf1_wn_spike_rate_vs_pupil_radius_tuning']
            tuning_err = row['hf1_wn_spike_rate_vs_pupil_radius_err']
            modind = modulation_index(tuning, zerocent=False)
            unitfig_wnsrpupilrad.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_wnsrpupilrad.set_title('WN spike rate vs pupil radius\nmod.ind.='+str(modind), fontsize=20)
            unitfig_wnsrpupilrad.set_ylim(0,np.nanmax(tuning[:]*1.2))
            newdf.at[row['index'], 'hf1_wn_spike_rate_vs_pupil_radius_modind'] = modind
        except:
            pass
        try:
            # wn spike rate vs running speed
            unitfig_wnsrvspeed = unitfig.add_subplot(spec[1, 4])
            var_cent = row['hf1_wn_spike_rate_vs_spd_cent']
            tuning = row['hf1_wn_spike_rate_vs_spd_tuning']
            tuning_err = row['hf1_wn_spike_rate_vs_spd_err']
            modind = modulation_index(tuning, zerocent=False)
            unitfig_wnsrvspeed.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_wnsrvspeed.set_title('WN spike rate vs running speed\nmod.ind.='+str(modind), fontsize=20)
            unitfig_wnsrvspeed.set_ylim(0,np.nanmax(tuning[:]*1.2))
            newdf.at[row['index'], 'hf1_wn_spike_rate_vs_spd_modind'] = modind
        except:
            pass
        try:
            # fm1 sta
            unitfig_fm1sta = unitfig.add_subplot(spec[2, 0])
            fm1sta = np.reshape(row[lightfm+'_spike_triggered_average'],tuple(row[lightfm+'_sta_shape']))
            fm1staRange = np.max(np.abs(fm1sta))*1.2
            if fm1staRange<0.25:
                fm1staRange=0.25
            unitfig_fm1sta.set_title('FM1 STA')
            unitfig_fm1sta.imshow(fm1sta,vmin=-fm1staRange,vmax=fm1staRange,cmap='jet')
            unitfig_fm1sta.axis('off')
        except:
            pass
        try:
            # fm1 stv
            unitfig_fm1stv = unitfig.add_subplot(spec[2, 1])
            wnstv = np.reshape(row[lightfm+'_spike_triggered_variance'],tuple(row[lightfm+'_sta_shape']))
            unitfig_fm1stv.imshow(wnstv,vmin=-1,vmax=1)
            unitfig_fm1stv.set_title('FM1 STV', fontsize=20)
            unitfig_fm1stv.axis('off')
        except:
            pass
        try:
            # fm1 spike rate vs gz
            unitfig_fm1srvgz = unitfig.add_subplot(spec[2, 2])
            var_cent = row[lightfm+'_spike_rate_vs_gz_cent']
            tuning = row[lightfm+'_spike_rate_vs_gz_tuning']
            tuning_err = row[lightfm+'_spike_rate_vs_gz_err']
            modind = modulation_index(tuning)
            unitfig_fm1srvgz.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srvgz.set_title('FM1 spike rate vs gyro_z\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
            unitfig_fm1srvgz.set_ylim(0,np.nanmax(tuning[:]*1.2))
            newdf.at[row['index'], 'fm1_wn_spike_rate_vs_gz_modind_neg'] = modind[0]; newdf.at[row['index'], 'fm1_wn_spike_rate_vs_gz_modind_pos'] = modind[1]
        except:
            pass
        try:
            # fm1 spike rate vs gx
            unitfig_fm1srvgx = unitfig.add_subplot(spec[2, 3])
            var_cent = row[lightfm+'_spike_rate_vs_gx_cent']
            tuning = row[lightfm+'_spike_rate_vs_gx_tuning']
            tuning_err = row[lightfm+'_spike_rate_vs_gx_err']
            modind = modulation_index(tuning)
            unitfig_fm1srvgx.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srvgx.set_title('FM1 spike rate vs gyro_x\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
            unitfig_fm1srvgx.set_ylim(0,np.nanmax(tuning[:]*1.2))
            newdf.at[row['index'], 'fm1_wn_spike_rate_vs_gx_modind_neg'] = modind[0]; newdf.at[row['index'], 'fm1_wn_spike_rate_vs_gx_modind_pos'] = modind[1]
        except:
            pass
        try:
            # fm1 spike rate vs gy
            unitfig_fm1srvgy = unitfig.add_subplot(spec[2, 4])
            var_cent = row[lightfm+'_spike_rate_vs_gy_cent']
            tuning = row[lightfm+'_spike_rate_vs_gy_tuning']
            tuning_err = row[lightfm+'_spike_rate_vs_gy_err']
            modind = modulation_index(tuning)
            unitfig_fm1srvgy.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srvgy.set_title('FM1 spike rate vs gyro_y\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
            unitfig_fm1srvgy.set_ylim(0,np.nanmax(tuning[:]*1.2))
            newdf.at[row['index'], 'fm1_wn_spike_rate_vs_gy_modind_neg'] = modind[0]; newdf.at[row['index'], 'fm1_wn_spike_rate_vs_gy_modind_pos'] = modind[1]
        except:
            pass
        try:
            # fm1 glm receptive field at five lags
            glm = row[lightfm+'_glm_receptive_field']
            glm_cc = row[lightfm+'_glm_cc']
            lag_list = [-4,-2,0,2,4]
            crange = np.max(np.abs(glm))
            for glm_lag in range(5):
                unitfig_glm = unitfig.add_subplot(spec[3, glm_lag])
                unitfig_glm.imshow(glm[glm_lag],vmin=-crange,vmax=crange,cmap='jet')
                unitfig_glm.set_title('FM1 GLM RF\n(lag='+str(lag_list[glm_lag])+' cc='+str(np.round(glm_cc[glm_lag],2))+')', fontsize=20)
                unitfig_glm.axis('off')
        except:
            pass
        try:
            # gaze shift dEye
            unitfig_fm1upsacc_gazedEye = unitfig.add_subplot(spec[4, 1])
            upsacc_avg = row[lightfm+'_upsacc_avg_gaze_shift_dEye']
            downsacc_avg = row[lightfm+'_downsacc_avg_gaze_shift_dEye']
            trange = row[lightfm+'_trange']
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_fm1upsacc_gazedEye.set_title('FM1 gaze shift dEye', fontsize=20)
            unitfig_fm1upsacc_gazedEye.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1upsacc_gazedEye.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
            unitfig_fm1upsacc_gazedEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_gazedEye.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
            unitfig_fm1upsacc_gazedEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_gazedEye.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_gazedEye.set_ylabel('sp/sec')
            newdf.at[row['index'], 'fm1_upsacc_avg_gaze_shift_dEye_modind_t0'] = modind_right[0]; newdf.at[row['index'], 'fm1_downsacc_avg_gaze_shift_dEye_modind_t0'] = modind_left[0]
            newdf.at[row['index'], 'fm1_upsacc_avg_gaze_shift_dEye_modind_t100'] = modind_right[1]; newdf.at[row['index'], 'fm1_downsacc_avg_gaze_shift_dEye_modind_t100'] = modind_left[1]
        except:
            pass
        try:
            # comp dEye
            unitfig_fm1upsacc_compdEye = unitfig.add_subplot(spec[4, 2])
            upsacc_avg = row[lightfm+'_upsacc_avg_comp_dEye']
            downsacc_avg = row[lightfm+'_downsacc_avg_comp_dEye']
            trange = row[lightfm+'_trange']
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_fm1upsacc_compdEye.set_title('FM1 comp dEye', fontsize=20)
            unitfig_fm1upsacc_compdEye.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1upsacc_compdEye.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
            unitfig_fm1upsacc_compdEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_compdEye.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
            unitfig_fm1upsacc_compdEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_compdEye.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_compdEye.set_ylabel('sp/sec')
            newdf.at[row['index'], 'fm1_upsacc_avg_comp_dEye_modind_t0'] = modind_right[0]; newdf.at[row['index'], 'fm1_downsacc_avg_comp_dEye_modind_t0'] = modind_left[0]
            newdf.at[row['index'], 'fm1_upsacc_avg_comp_dEye_modind_t100'] = modind_right[1]; newdf.at[row['index'], 'fm1_downsacc_avg_comp_dEye_modind_t100'] = modind_left[1]
        except:
            pass
        try:
            # gaze shift dHead
            unitfig_fm1upsacc_gazedHead = unitfig.add_subplot(spec[4, 3])
            upsacc_avg = row[lightfm+'_upsacc_avg_gaze_shift_dHead']
            downsacc_avg = row[lightfm+'_downsacc_avg_gaze_shift_dHead']
            trange = row[lightfm+'_trange']
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_fm1upsacc_gazedHead.set_title('FM1 gaze shift dHead', fontsize=20)
            unitfig_fm1upsacc_gazedHead.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1upsacc_gazedHead.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
            unitfig_fm1upsacc_gazedHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_gazedHead.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
            unitfig_fm1upsacc_gazedHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_gazedHead.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_gazedHead.set_ylabel('sp/sec')
            newdf.at[row['index'], 'fm1_wn_upsacc_avg_gaze_shift_dHead_modind_t0'] = modind_right[0]; newdf.at[row['index'], 'fm1_downsacc_avg_gaze_shift_dHead_modind_t0'] = modind_left[0]
            newdf.at[row['index'], 'fm1_upsacc_avg_gaze_shift_dHead_modind_t100'] = modind_right[1]; newdf.at[row['index'], 'fm1_downsacc_avg_gaze_shift_dHead_modind_t100'] = modind_left[1]
        except:
            pass
        try:
            # gaze shift comp dHead
            unitfig_fm1upsacc_compdHead = unitfig.add_subplot(spec[4, 4])
            upsacc_avg = row[lightfm+'_upsacc_avg_comp_dHead']
            downsacc_avg = row[lightfm+'_downsacc_avg_comp_dHead']
            trange = row[lightfm+'_trange']
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_fm1upsacc_compdHead.set_title('FM1 comp dHead', fontsize=20)
            unitfig_fm1upsacc_compdHead.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1upsacc_compdHead.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
            unitfig_fm1upsacc_compdHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_compdHead.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
            unitfig_fm1upsacc_compdHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_compdHead.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_compdHead.set_ylabel('sp/sec')
            newdf.at[row['index'], 'fm1_upsacc_avg_comp_dHead_modind_t0'] = modind_right[0]; newdf.at[row['index'], 'fm1_downsacc_avg_comp_dHead_modind_t0'] = modind_left[0]
            newdf.at[row['index'], 'fm1_upsacc_avg_comp_dHead_modind_t100'] = modind_right[1]; newdf.at[row['index'], 'fm1_downsacc_avg_comp_dHead_modind_t100'] = modind_left[1]
        except:
            pass
        try:
            # orientation spatial frequency tuning curve
            unitfig_ori_tuning = unitfig.add_subplot(spec[6, 0])
            ori_tuning = np.mean(row['hf3_gratings_ori_tuning'],2) # [orientation, sf, tf]
            drift_spont = row['hf3_gratings_drift_spont']
            tuning = ori_tuning - drift_spont # subtract off spont rate
            tuning[tuning < 0] = 0 # set to 0 when tuning goes negative (i.e. when firing rate is below spontanious rate)
            th_pref = np.nanargmax(tuning,0) # get position of highest firing rate
            osi = np.zeros([3])
            for sf in range(3):
                R_pref = (tuning[th_pref[sf], sf] + (tuning[(th_pref[sf]+4)%8, sf])) * 0.5 # get that firing rate (avg between peaks)
                th_ortho = (th_pref[sf]+2)%8 # get ortho position
                R_ortho = (tuning[th_ortho, sf] + (tuning[(th_ortho+4)%8, sf])) * 0.5 # ortho firing rate (average between two peaks)
                osi[sf] = np.round((R_pref - R_ortho) / (R_pref + R_ortho),3)
            unitfig_ori_tuning.set_title('orientation tuning\nosi l='+str(osi[0])+'m='+str(osi[1])+'h='+str(osi[2]), fontsize=20)
            unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,0],label = 'low sf')
            unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,1],label = 'mid sf')
            unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,2],label = 'hi sf')
            unitfig_ori_tuning.plot([0,315],[drift_spont,drift_spont],'r:',label='spont')
            unitfig_ori_tuning.legend()
            unitfig_ori_tuning.set_ylim([0,np.nanmax(row['hf3_gratings_ori_tuning'][:,:,:])*1.2])
            newdf.at[row['index'], 'hf3_gratings_osi_low'] = osi[0]; newdf.at[row['index'], 'hf3_gratings_osi_mid'] = osi[1]; newdf.at[row['index'], 'hf3_gratings_osi_high'] = osi[2]
        except:
            pass
        try:
            # orientation temporal frequency tuning curve
            # low temporal freq
            unitfig_ori_tuning_low_tf = unitfig.add_subplot(spec[6, 1])
            tuning = row['hf3_gratings_ori_tuning'][:,:,0] - drift_spont # [:,low/mid/high sf,low/high tf]
            tuning[tuning < 0] = 0 # set to 0 when tuning goes negative (i.e. when firing rate is below spontanious rate)
            th_pref = np.nanargmax(tuning,0) # get position of highest firing rate
            osi = np.zeros([3])
            for sf in range(3):
                R_pref = (tuning[th_pref[sf], sf] + (tuning[(th_pref[sf]+4)%8, sf])) * 0.5 # get that firing rate (avg between peaks)
                th_ortho = (th_pref[sf]+2)%8 # get ortho position
                R_ortho = (tuning[th_ortho, sf] + (tuning[(th_ortho+4)%8, sf])) * 0.5 # ortho firing rate (average between two peaks)
                osi[sf] = np.round((R_pref - R_ortho) / (R_pref + R_ortho),3)
            unitfig_ori_tuning_low_tf.set_title('low tf\nosi l='+str(osi[0])+' m='+str(osi[1])+' h='+str(osi[2]), fontsize=20)
            unitfig_ori_tuning_low_tf.plot(np.arange(8)*45, row['hf3_gratings_ori_tuning'][:,:,0][:,0],label = 'low sf')
            unitfig_ori_tuning_low_tf.plot(np.arange(8)*45, row['hf3_gratings_ori_tuning'][:,:,0][:,1],label = 'mid sf')
            unitfig_ori_tuning_low_tf.plot(np.arange(8)*45, row['hf3_gratings_ori_tuning'][:,:,0][:,2],label = 'hi sf')
            unitfig_ori_tuning_low_tf.plot([0,315],[drift_spont,drift_spont],'r:',label='spont')
            unitfig_ori_tuning_low_tf.set_ylim([0,np.nanmax(row['hf3_gratings_ori_tuning'][:,:,:])*1.2])
            # high temporal freq
            unitfig_ori_tuning_high_tf = unitfig.add_subplot(spec[6, 2])
            tuning = row['hf3_gratings_ori_tuning'][:,:,1] - drift_spont # subtract off spont rate
            tuning[tuning < 0] = 0 # set to 0 when tuning goes negative (i.e. when firing rate is below spontanious rate)
            th_pref = np.nanargmax(tuning,0) # get position of highest firing rate
            osi = np.zeros([3])
            for sf in range(3):
                R_pref = (tuning[th_pref[sf], sf] + (tuning[(th_pref[sf]+4)%8, sf])) * 0.5 # get that firing rate (avg between peaks)
                th_ortho = (th_pref[sf]+2)%8 # get ortho position
                R_ortho = (tuning[th_ortho, sf] + (tuning[(th_ortho+4)%8, sf])) * 0.5 # ortho firing rate (average between two peaks)
                osi[sf] = np.round((R_pref - R_ortho) / (R_pref + R_ortho),3)
            unitfig_ori_tuning_high_tf.set_title('high tf\nosi l='+str(osi[0])+'m='+str(osi[1])+'h='+str(osi[2]), fontsize=20)
            unitfig_ori_tuning_high_tf.plot(np.arange(8)*45, row['hf3_gratings_ori_tuning'][:,:,1][:,0],label = 'low sf')
            unitfig_ori_tuning_high_tf.plot(np.arange(8)*45, row['hf3_gratings_ori_tuning'][:,:,1][:,1],label = 'mid sf')
            unitfig_ori_tuning_high_tf.plot(np.arange(8)*45, row['hf3_gratings_ori_tuning'][:,:,1][:,2],label = 'hi sf')
            unitfig_ori_tuning_high_tf.plot([0,315],[drift_spont,drift_spont],'r:',label='spont')
            unitfig_ori_tuning_high_tf.set_ylim([0,np.nanmax(row['hf3_gratings_ori_tuning'][:,:,:])*1.2])
        except:
            pass
        try:
            # fm1 eye movements
            unitfig_fm1saccavg = unitfig.add_subplot(spec[4, 0])
            trange = row[lightfm+'_trange']
            upsacc_avg = row[lightfm+'_upsacc_avg']; downsacc_avg = row[lightfm+'_downsacc_avg']
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_fm1saccavg.set_title('FM1 left/right saccades', fontsize=20)
            unitfig_fm1saccavg.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1saccavg.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
            unitfig_fm1saccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1saccavg.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            unitfig_fm1saccavg.set_ylim([0,maxval*1.2])
            newdf.at[row['index'], 'fm1_upsacc_modind_t0'] = modind_right[0]; newdf.at[row['index'], 'fm1_downsacc_modind_t0'] = modind_left[0]
            newdf.at[row['index'], 'fm1_upsacc_modind_t100'] = modind_right[1]; newdf.at[row['index'], 'fm1_downsacc_modind_t100'] = modind_left[1]
        except:
            pass
        try:
            # fm1 spike rate vs pupil radius
            unitfig_fm1srpupilrad = unitfig.add_subplot(spec[5, 0])
            var_cent = row[lightfm+'_spike_rate_vs_pupil_radius_cent']
            tuning = row[lightfm+'_spike_rate_vs_pupil_radius_tuning']
            tuning_err = row[lightfm+'_spike_rate_vs_pupil_radius_err']
            modind = modulation_index(tuning, zerocent=False)
            unitfig_fm1srpupilrad.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srpupilrad.set_title('FM1 spike rate vs pupil radius\nmod.ind.='+str(modind), fontsize=20)
            unitfig_fm1srpupilrad.set_ylim(0,np.nanmax(tuning[:]*1.2))
            newdf.at[row['index'], 'hf1_wn_spike_rate_vs_pupil_radius_modind'] = modind
        except:
            pass
        try:
            # fm1 spike rate vs theta
            unitfig_fm1srth = unitfig.add_subplot(spec[5, 1])
            var_cent = row[lightfm+'_spike_rate_vs_theta_cent']
            tuning = row[lightfm+'_spike_rate_vs_theta_tuning']
            tuning_err = row[lightfm+'_spike_rate_vs_theta_err']
            modind = modulation_index(tuning)
            unitfig_fm1srth.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srth.set_title('FM1 spike rate vs theta\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
            unitfig_fm1srth.set_ylim(0,np.nanmax(tuning[:]*1.2))
            newdf.at[row['index'], 'fm1_wn_spike_rate_vs_theta_modind_neg'] = modind[0]; newdf.at[row['index'], 'fm1_wn_spike_rate_vs_theta_modind_pos'] = modind[1]
        except:
            pass
        try:
            # fm1 spike rate vs phi
            unitfig_fm1srphi = unitfig.add_subplot(spec[5, 2])
            var_cent = row[lightfm+'_spike_rate_vs_phi_cent']
            tuning = row[lightfm+'_spike_rate_vs_phi_tuning']
            tuning_err = row[lightfm+'_spike_rate_vs_phi_err']
            modind = modulation_index(tuning)
            unitfig_fm1srphi.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srphi.set_title('FM1 spike rate vs phi\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
            unitfig_fm1srphi.set_ylim(0,np.nanmax(tuning[:]*1.2))
            newdf.at[row['index'], 'fm1_wn_spike_rate_vs_phi_modind_neg'] = modind[0]; newdf.at[row['index'], 'fm1_wn_spike_rate_vs_phi_modind_pos'] = modind[1]
        except:
            pass
        try:
            # fm1 spike rate vs roll
            unitfig_fm1srroll = unitfig.add_subplot(spec[5, 3])
            var_cent = row[lightfm+'_spike_rate_vs_roll_cent']
            tuning = row[lightfm+'_spike_rate_vs_roll_tuning']
            tuning_err = row[lightfm+'_spike_rate_vs_roll_err']
            modind = modulation_index(tuning, lbound=5, ubound=-6)
            unitfig_fm1srroll.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srroll.set_title('FM1 spike rate vs roll\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
            unitfig_fm1srroll.set_ylim(0,np.nanmax(tuning[:]*1.2)); unitfig_fm1srroll.set_xlim(-30,30)
            newdf.at[row['index'], 'fm1_wn_spike_rate_vs_roll_modind_neg'] = modind[0]; newdf.at[row['index'], 'fm1_wn_spike_rate_vs_roll_modind_pos'] = modind[1]
        except:
            pass
        try:
            # fm1 spike rate vs pitch
            unitfig_fm1srpitch = unitfig.add_subplot(spec[5, 4])
            var_cent = row[lightfm+'_spike_rate_vs_pitch_cent']
            tuning = row[lightfm+'_spike_rate_vs_pitch_tuning']
            tuning_err = row[lightfm+'_spike_rate_vs_pitch_err']
            modind = modulation_index(tuning, lbound=5, ubound=-6)
            unitfig_fm1srpitch.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srpitch.set_title('FM1 spike rate vs pitch\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
            unitfig_fm1srpitch.set_ylim(0,np.nanmax(tuning[:]*1.2)); unitfig_fm1srpitch.set_xlim(-30,30)
            newdf.at[row['index'], 'fm1_wn_spike_rate_vs_pitch_modind_neg'] = modind[0]; newdf.at[row['index'], 'fm1_wn_spike_rate_vs_pitch_modind_pos'] = modind[1]
        except:
            pass

        ### DARK FM PLOTS ###
        if darkfm is not None:
            try:
                # fm1 spike rate vs gz
                unitfig_darksrvgz = unitfig.add_subplot(spec[7, 0])
                var_cent = row[darkfm+'_spike_rate_vs_gz_cent']
                tuning = row[darkfm+'_spike_rate_vs_gz_tuning']
                tuning_err = row[lightfm+'_spike_rate_vs_gz_err']
                modind = modulation_index(tuning)
                unitfig_darksrvgz.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
                unitfig_darksrvgz.set_title('FM DARK spike rate vs gyro_z\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
                unitfig_darksrvgz.set_ylim(0,np.nanmax(tuning[:]*1.2))
                newdf.at[row['index'], darkfm+'_wn_spike_rate_vs_gz_modind_neg'] = modind[0]; newdf.at[row['index'], darkfm+'_spike_rate_vs_gz_modind_pos'] = modind[1]
            except:
                pass
            try:
                # fm1 spike rate vs gx
                unitfig_darksrvgx = unitfig.add_subplot(spec[7, 1])
                var_cent = row[darkfm+'_spike_rate_vs_gx_cent']
                tuning = row[darkfm+'_spike_rate_vs_gx_tuning']
                tuning_err = row[darkfm+'_spike_rate_vs_gx_err']
                modind = modulation_index(tuning)
                unitfig_darksrvgx.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
                unitfig_darksrvgx.set_title('FM DARK spike rate vs gyro_x\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
                unitfig_darksrvgx.set_ylim(0,np.nanmax(tuning[:]*1.2))
                newdf.at[row['index'], darkfm+'_spike_rate_vs_gx_modind_neg'] = modind[0]; newdf.at[row['index'], 'fm1_wn_spike_rate_vs_gx_modind_pos'] = modind[1]
            except:
                pass
            try:
                # fm1 spike rate vs gy
                unitfig_darksrvgy = unitfig.add_subplot(spec[7, 2])
                var_cent = row[darkfm+'_spike_rate_vs_gy_cent']
                tuning = row[darkfm+'_spike_rate_vs_gy_tuning']
                tuning_err = row[darkfm+'_spike_rate_vs_gy_err']
                modind = modulation_index(tuning)
                unitfig_darksrvgy.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
                unitfig_darksrvgy.set_title('FM DARK spike rate vs gyro_y\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
                unitfig_darksrvgy.set_ylim(0,np.nanmax(tuning[:]*1.2))
                newdf.at[row['index'], darkfm+'_spike_rate_vs_gy_modind_neg'] = modind[0]; newdf.at[row['index'], darkfm+'_spike_rate_vs_gy_modind_pos'] = modind[1]
            except:
                pass
            try:
                # gaze shift dEye
                unitfig_darkupsacc_gazedEye = unitfig.add_subplot(spec[8, 0])
                upsacc_avg = row[darkfm+'_upsacc_avg_gaze_shift_dEye']
                downsacc_avg = row[darkfm+'_downsacc_avg_gaze_shift_dEye']
                trange = row[darkfm+'_trange']
                maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
                modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
                unitfig_darkupsacc_gazedEye.set_title('FM DARK gaze shift dEye', fontsize=20)
                unitfig_darkupsacc_gazedEye.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
                unitfig_darkupsacc_gazedEye.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
                unitfig_darkupsacc_gazedEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
                unitfig_darkupsacc_gazedEye.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
                unitfig_darkupsacc_gazedEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
                unitfig_darkupsacc_gazedEye.set_ylim([0,maxval*1.2])
                unitfig_darkupsacc_gazedEye.set_ylabel('sp/sec')
                newdf.at[row['index'], darkfm+'_upsacc_avg_gaze_shift_dEye_modind_t0'] = modind_right[0]; newdf.at[row['index'], darkfm+'_downsacc_avg_gaze_shift_dEye_modind_t0'] = modind_left[0]
                newdf.at[row['index'], darkfm+'_upsacc_avg_gaze_shift_dEye_modind_t100'] = modind_right[1]; newdf.at[row['index'], darkfm+'_downsacc_avg_gaze_shift_dEye_modind_t100'] = modind_left[1]
            except:
                pass
            try:
                # comp dEye
                unitfig_darkupsacc_compdEye = unitfig.add_subplot(spec[8, 1])
                upsacc_avg = row[darkfm+'_upsacc_avg_comp_dEye']
                downsacc_avg = row[darkfm+'_downsacc_avg_comp_dEye']
                trange = row[darkfm+'_trange']
                maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
                modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
                unitfig_darkupsacc_compdEye.set_title('FM DARK comp dEye', fontsize=20)
                unitfig_darkupsacc_compdEye.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
                unitfig_darkupsacc_compdEye.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
                unitfig_darkupsacc_compdEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
                unitfig_darkupsacc_compdEye.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
                unitfig_darkupsacc_compdEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
                unitfig_darkupsacc_compdEye.set_ylim([0,maxval*1.2])
                unitfig_darkupsacc_compdEye.set_ylabel('sp/sec')
                newdf.at[row['index'], darkfm+'_upsacc_avg_comp_dEye_modind_t0'] = modind_right[0]; newdf.at[row['index'], darkfm+'_downsacc_avg_comp_dEye_modind_t0'] = modind_left[0]
                newdf.at[row['index'], darkfm+'_upsacc_avg_comp_dEye_modind_t100'] = modind_right[1]; newdf.at[row['index'], darkfm+'_downsacc_avg_comp_dEye_modind_t100'] = modind_left[1]
            except:
                pass
            try:
                # gaze shift dHead
                unitfig_darkupsacc_gazedHead = unitfig.add_subplot(spec[8, 2])
                upsacc_avg = row[darkfm+'_upsacc_avg_gaze_shift_dHead']
                downsacc_avg = row[darkfm+'_downsacc_avg_gaze_shift_dHead']
                trange = row[darkfm+'_trange']
                maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
                modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
                unitfig_darkupsacc_gazedHead.set_title('FM DARK gaze shift dHead', fontsize=20)
                unitfig_darkupsacc_gazedHead.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
                unitfig_darkupsacc_gazedHead.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
                unitfig_darkupsacc_gazedHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
                unitfig_darkupsacc_gazedHead.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
                unitfig_darkupsacc_gazedHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
                unitfig_darkupsacc_gazedHead.set_ylim([0,maxval*1.2])
                unitfig_darkupsacc_gazedHead.set_ylabel('sp/sec')
                newdf.at[row['index'], darkfm+'_wn_upsacc_avg_gaze_shift_dHead_modind_t0'] = modind_right[0]; newdf.at[row['index'], darkfm+'_downsacc_avg_gaze_shift_dHead_modind_t0'] = modind_left[0]
                newdf.at[row['index'], darkfm+'_upsacc_avg_gaze_shift_dHead_modind_t100'] = modind_right[1]; newdf.at[row['index'], darkfm+'_downsacc_avg_gaze_shift_dHead_modind_t100'] = modind_left[1]
            except:
                pass
            try:
                # gaze shift comp dHead
                unitfig_darkupsacc_compdHead = unitfig.add_subplot(spec[8, 3])
                upsacc_avg = row[darkfm+'_upsacc_avg_comp_dHead']
                downsacc_avg = row[darkfm+'_downsacc_avg_comp_dHead']
                trange = row[darkfm+'_trange']
                maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
                modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
                unitfig_darkupsacc_compdHead.set_title('FM DARK comp dHead', fontsize=20)
                unitfig_darkupsacc_compdHead.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
                unitfig_darkupsacc_compdHead.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
                unitfig_darkupsacc_compdHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
                unitfig_darkupsacc_compdHead.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
                unitfig_darkupsacc_compdHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
                unitfig_darkupsacc_compdHead.set_ylim([0,maxval*1.2])
                unitfig_darkupsacc_compdHead.set_ylabel('sp/sec')
                newdf.at[row['index'], darkfm+'_upsacc_avg_comp_dHead_modind_t0'] = modind_right[0]; newdf.at[row['index'], darkfm+'_downsacc_avg_comp_dHead_modind_t0'] = modind_left[0]
                newdf.at[row['index'], darkfm+'_upsacc_avg_comp_dHead_modind_t100'] = modind_right[1]; newdf.at[row['index'], darkfm+'_downsacc_avg_comp_dHead_modind_t100'] = modind_left[1]
            except:
                pass
            try:
                # fm1 eye movements
                unitfig_darksaccavg = unitfig.add_subplot(spec[8,4])
                trange = row[darkfm+'_trange']
                upsacc_avg = row[darkfm+'_upsacc_avg']
                downsacc_avg = row[darkfm+'_downsacc_avg']
                modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
                unitfig_darksaccavg.set_title('FM DARK left/right saccades', fontsize=20)
                unitfig_darksaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
                unitfig_darksaccavg.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction')
                unitfig_darksaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
                unitfig_darksaccavg.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction')
                maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
                unitfig_darksaccavg.set_ylim([0,maxval*1.2])
                newdf.at[row['index'], darkfm+'_upsacc_modind_t0'] = modind_right[0]; newdf.at[row['index'], darkfm+'_downsacc_modind_t0'] = modind_left[0]
                newdf.at[row['index'], darkfm+'_upsacc_modind_t100'] = modind_right[1]; newdf.at[row['index'], darkfm+'_downsacc_modind_t100'] = modind_left[1]
            except:
                pass
            try:
                # fm1 spike rate vs pupil radius
                unitfig_darksrpupilrad = unitfig.add_subplot(spec[9, 0])
                var_cent = row[darkfm+'_spike_rate_vs_pupil_radius_cent']
                tuning = row[darkfm+'_spike_rate_vs_pupil_radius_tuning']
                tuning_err = row[darkfm+'_spike_rate_vs_pupil_radius_err']
                modind = modulation_index(tuning, zerocent=False)
                unitfig_darksrpupilrad.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
                unitfig_darksrpupilrad.set_title('FM DARK spike rate vs pupil radius\nmod.ind.='+str(modind), fontsize=20)
                unitfig_darksrpupilrad.set_ylim(0,np.nanmax(tuning[:]*1.2))
                newdf.at[row['index'], darkfm+'_spike_rate_vs_pupil_radius_modind'] = modind
            except:
                pass
            try:
                # fm1 spike rate vs theta
                unitfig_darksrth = unitfig.add_subplot(spec[9, 1])
                var_cent = row[darkfm+'_spike_rate_vs_theta_cent']
                tuning = row[darkfm+'_spike_rate_vs_theta_tuning']
                tuning_err = row[darkfm+'_spike_rate_vs_theta_err']
                modind = modulation_index(tuning)
                unitfig_darksrth.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
                unitfig_darksrth.set_title('FM DARK spike rate vs theta\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
                unitfig_darksrth.set_ylim(0,np.nanmax(tuning[:]*1.2))
                newdf.at[row['index'], darkfm+'_spike_rate_vs_theta_modind_neg'] = modind[0]; newdf.at[row['index'], darkfm+'_spike_rate_vs_theta_modind_pos'] = modind[1]
            except:
                pass
            try:
                # fm1 spike rate vs phi
                unitfig_fm1srphi = unitfig.add_subplot(spec[9, 2])
                var_cent = row[darkfm+'_spike_rate_vs_phi_cent']
                tuning = row[darkfm+'_spike_rate_vs_phi_tuning']
                tuning_err = row[darkfm+'_spike_rate_vs_phi_err']
                modind = modulation_index(tuning)
                unitfig_fm1srphi.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
                unitfig_fm1srphi.set_title('FM DARK spike rate vs phi\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
                unitfig_fm1srphi.set_ylim(0,np.nanmax(tuning[:]*1.2))
                newdf.at[row['index'], 'fm1_wn_spike_rate_vs_phi_modind_neg'] = modind[0]; newdf.at[row['index'], 'fm1_wn_spike_rate_vs_phi_modind_pos'] = modind[1]
            except:
                pass
            try:
                # fm1 spike rate vs roll
                unitfig_fm1srroll = unitfig.add_subplot(spec[9, 3])
                var_cent = row[darkfm+'_spike_rate_vs_roll_cent']
                tuning = row[darkfm+'_spike_rate_vs_roll_tuning']
                tuning_err = row[darkfm+'_spike_rate_vs_roll_err']
                modind = modulation_index(tuning, lbound=5, ubound=-6)
                unitfig_fm1srroll.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
                unitfig_fm1srroll.set_title('FM DARK spike rate vs roll\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
                unitfig_fm1srroll.set_ylim(0,np.nanmax(tuning[:]*1.2)); unitfig_fm1srroll.set_xlim(-30,30)
                newdf.at[row['index'], darkfm+'_spike_rate_vs_roll_modind_neg'] = modind[0]; newdf.at[row['index'], darkfm+'_spike_rate_vs_roll_modind_pos'] = modind[1]
            except:
                pass
            try:
                # fm1 spike rate vs pitch
                unitfig_fm1srpitch = unitfig.add_subplot(spec[9, 4])
                var_cent = row[darkfm+'_spike_rate_vs_pitch_cent']
                tuning = row[darkfm+'_spike_rate_vs_pitch_tuning']
                tuning_err = row[darkfm+'_spike_rate_vs_pitch_err']
                modind = modulation_index(tuning, lbound=5, ubound=-6)
                unitfig_fm1srpitch.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
                unitfig_fm1srpitch.set_title('FM1 spike rate vs pitch\nmod.ind.='+str(modind[0])+'/'+str(modind[1]), fontsize=20)
                unitfig_fm1srpitch.set_ylim(0,np.nanmax(tuning[:]*1.2)); unitfig_fm1srpitch.set_xlim(-30,30)
                newdf.at[row['index'], darkfm+'_spike_rate_vs_pitch_modind_neg'] = modind[0]; newdf.at[row['index'], darkfm+'_spike_rate_vs_pitch_modind_pos'] = modind[1]
            except:
                pass

        plt.tight_layout()
        pdf.savefig(unitfig)
        plt.close()

    print('saving unit summary pdf')
    pdf.close()

    print('saving an updated json')
    path_out = os.path.join(savepath, 'pooled_ephys_unit_update_'+datetime.today().strftime('%m%d%y')+'.json')
    if os.path.isfile(path_out):
        os.remove(path_out)
    newdf.to_json(path_out, default_handler=str)

    return newdf

def make_session_summary(df, savepath):
    """
    make a pdf summarizing all sessions (i.e. each animal on each date)
    INPUTS
        df: ephys dataframe
        savepath: where to save pdf
    OUTPUTS
        None
    """
    pdf = PdfPages(os.path.join(savepath, 'session_summary_'+datetime.today().strftime('%m%d%y')+'.pdf'))
    df['unit'] = df.index.values
    print(df, df.index)
    df = df.set_index('session')
    print(df, df.index)

    unique_inds = sorted(list(set(df.index.values)))

    for unique_ind in tqdm(unique_inds):
        uniquedf = df.loc[unique_ind]
        # set up subplots
        plt.subplots(4,4,figsize=(25,25))
        # eye position vs head position
        try:
            plt.subplot(4,4,1)
            plt.title('dEye vs dHead (LIGHT)', fontsize=20)
            dEye = uniquedf['fm1_dEye'].iloc[0]
            dhead = uniquedf['fm1_dHead'].iloc[0]
            eyeT = uniquedf['fm1_eyeT'].iloc[0]
            plt.plot(dEye[0:-1:10],dhead(eyeT[0:-1:10]),'.')
            plt.xlabel('dEye'); plt.ylabel('dHead'); plt.xlim((-10,10)); plt.ylim((-10,10))
            plt.plot([-10,10],[10,-10], 'r')
        except:
            pass
        try:
            plt.subplot(4,4,2)
            plt.title('dEye vs dHead (DARK)', fontsize=20)
            dEye = uniquedf['fm1_dark_dEye'].iloc[0]
            dhead = uniquedf['fm1_dark_dHead'].iloc[0]
            eyeT = uniquedf['fm1_dark_eyeT'].iloc[0]
            plt.plot(dEye[0:-1:10],dhead(eyeT[0:-1:10]),'.')
            plt.xlabel('dEye'); plt.ylabel('dHead'); plt.xlim((-10,10)); plt.ylim((-10,10))
            plt.plot([-10,10],[10,-10], 'r')
        except:
            pass
        try:
            accT = uniquedf['fm1_accT'].iloc[0]
            roll_interp = uniquedf['fm1_roll_interp'].iloc[0]
            pitch_interp = uniquedf['fm1_pitch_interp'].iloc[0]
            th = uniquedf['fm1_theta'].iloc[0]
            phi = uniquedf['fm1_phi'].iloc[0]
            plt.subplot(4,4,3)
            plt.plot(pitch_interp[::100], th[::100], '.'); plt.xlabel('pitch'); plt.ylabel('theta')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
            plt.title('LIGHT'+unique_ind+' eye fit: m='+str(uniquedf['best_ellipse_fit_m'].iloc[0])+' r='+str(uniquedf['best_ellipse_fit_r'].iloc[0]), fontsize=20)
            plt.subplot(4,4,3)
            plt.plot(roll_interp[::100], phi[::100], '.'); plt.xlabel('roll'); plt.ylabel('phi')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')
        except:
            pass
        try:
            dark_roll_interp = uniquedf['fm1_dark_roll_interp'].iloc[0]
            dark_pitch_interp = uniquedf['fm1_dark_pitch_interp'].iloc[0]
            th_dark = uniquedf['fm1_dark_theta'].iloc[0]
            phi_dark = uniquedf['fm1_dark_phi'].iloc[0]
            plt.subplot(4,4,4)
            plt.plot(dark_pitch_interp[::100], th_dark[::100], '.'); plt.xlabel('pitch'); plt.ylabel('theta')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
            plt.title('DARK', fontsize=20)
            plt.subplot(4,4,3)
            plt.plot(dark_roll_interp[::100], phi_dark[::100], '.'); plt.xlabel('roll'); plt.ylabel('phi')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')
        except:
            pass
        try:
            # histogram of theta from -45 to 45deg (are eye movements in resonable range?)
            plt.subplot(4,4,5)
            plt.title('hist of FM theta', fontsize=20)
            plt.hist(uniquedf['fm1_theta'].iloc[0], range=[-45,45])
            # histogram of phi from -45 to 45deg (are eye movements in resonable range?)
            plt.subplot(4,4,6)
            plt.title('hist of FM phi', fontsize=20)
            plt.hist(uniquedf['fm1_phi'].iloc[0], range=[-45,45])
            # histogram of gyro z (resonable range?)
            plt.subplot(4,4,7)
            plt.title('hist of FM gyro z', fontsize=20)
            plt.hist(uniquedf['fm1_gz'].iloc[0], range=[2,4])
            # plot of contrast response functions on same panel scaled to max 30sp/sec
            # plot of average contrast reponse function across units
            plt.subplot(4,4,8)
            plt.title('CRFs', fontsize=20)
            for ind, row in uniquedf.iterrows():
                plt.errorbar(row['hf1_wn_crf_cent'],row['hf1_wn_crf_tuning'],yerr=row['hf1_wn_crf_err'])
            plt.ylim(0,30)
            plt.errorbar(uniquedf['hf1_wn_crf_cent'].iloc[0],np.mean(uniquedf['hf1_wn_crf_tuning'],axis=0),yerr=np.mean(uniquedf['hf1_wn_crf_err'],axis=0), color='k', linewidth=6)
            # lfp traces as separate shanks
            colors = plt.cm.jet(np.linspace(0,1,32))
            num_channels = np.size(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0],0)
            if num_channels == 64:
                for ch_num in np.arange(0,64):
                    if ch_num<=31:
                        plt.subplot(4,4,9)
                        plt.plot(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0][ch_num], color=colors[ch_num], linewidth=1)
                        plt.title('lfp trace, shank1', fontsize=20); plt.axvline(x=(0.1*30000))
                        plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                        plt.ylim([-1200,400])
                    if ch_num>31:
                        plt.subplot(4,4,10)
                        plt.plot(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0][ch_num], color=colors[ch_num-32], linewidth=1)
                        plt.title('lfp trace, shank2', fontsize=20); plt.axvline(x=(0.1*30000))
                        plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                        plt.ylim([-1200,400])
            # fm spike raster
            plt.subplot(4,4,11)
            plt.title('FM raster', fontsize=20)
            i = 0
            for ind, row in uniquedf.iterrows():
                plt.vlines(row['fm1_spikeT'],i-0.25,i+0.25)
                plt.xlim(0, 10); plt.xlabel('secs'); plt.ylabel('unit #')
                i = i+1
        except:
            pass
        # all psth plots in a single panel, with avg plotted over the top
        plt.subplot(4,4,12)
        lower = -0.5; upper = 1.5; dt = 0.1
        bins = np.arange(lower,upper+dt,dt)
        psth_list = []
        for ind, row in uniquedf.iterrows():
            try:
                plt.plot(bins[0:-1]+dt/2,row['hf3_gratings_grating_psth'])
                psth_list.append(row['hf3_gratings_grating_psth'])
            except ValueError:
                pass
        try:
            avg_psth = np.mean(np.array(psth_list), axis=0)
            plt.plot(bins[0:-1]+dt/2,avg_psth,color='k',linewidth=6)
            plt.title('gratings psth', fontsize=20); plt.xlabel('time'); plt.ylabel('sp/sec')
            plt.ylim([0,np.nanmax(avg_psth)*1.5])
        except:
            pass
        plt.subplot(4,4,13)
        plt.axis('off')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    print('saving session summary pdf')
    pdf.close()

def make_population_summary(df1, savepath):
    """
    summarize an entire population of ephys units
    INPUTS
        df1: updated dataframe from unit summary
        savepath: where to save the pdf and the updated dataframe
    OUTPUTS
        None
    """
    newdf = df1.copy()
    pdf = PdfPages(os.path.join(savepath, 'population_summary_'+datetime.today().strftime('%m%d%y')+'.pdf'))
    plt.figure()
    plt.title('normalized waveform')
    df1['norm_waveform'] = df1['waveform']
    for ind, row in df1.iterrows():
        if type(row['waveform']) == list:
            starting_val = np.mean(row['waveform'][:6])
            center_waveform = [i-starting_val for i in row['waveform']]
            norm_waveform = center_waveform / -np.min(center_waveform)
            plt.plot(norm_waveform)
            df1.at[ind, 'waveform_trough_width'] = len(norm_waveform[norm_waveform < -0.2])
            df1.at[ind, 'AHP'] = norm_waveform[27]
            df1.at[ind, 'waveform_peak'] = norm_waveform[18]
            df1.at[ind, 'norm_waveform'] = norm_waveform
    plt.ylim([-1,1]); plt.ylabel('millivolts'); plt.xlabel('msec')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.hist(df1['waveform_trough_width'],bins=range(3,35))
    plt.xlabel('trough width')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.xlabel('AHP')
    plt.hist(df1['AHP'],bins=60); plt.xlim([-1,1])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.title('thresh: AHP < 0, wvfm peak < 0')
    for ind, row in df1.iterrows():
        if row['AHP'] <=0 and row['waveform_peak'] < 0:
            plt.plot(row['norm_waveform'], 'g')
            df1.at[ind, 'waveform_type'] = 'narrow'
        elif row['AHP'] > 0 and row['waveform_peak'] < 0:
            plt.plot(row['norm_waveform'], 'b')
            df1.at[ind, 'waveform_type'] = 'broad'
        else:
            df1.at[ind, 'waveform_type'] = 'bad'
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.title('seperation by properties')
    plt.plot(df1['waveform_trough_width'][df1['waveform_peak'] < 0][df1['waveform_type']=='broad'], df1['AHP'][df1['waveform_peak'] < 0][df1['waveform_type']=='broad'], 'b.')
    plt.plot(df1['waveform_trough_width'][df1['waveform_peak'] < 0][df1['waveform_type']=='narrow'], df1['AHP'][df1['waveform_peak'] < 0][df1['waveform_type']=='narrow'], 'g.')
    plt.ylabel('AHP'); plt.xlabel('waveform trough width')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    km_labels = KMeans(n_clusters=2).fit(list(df1['norm_waveform'][df1['waveform_peak'] < 0].to_numpy())).labels_
    count = 0
    for ind, row in df1.iterrows():
        if row['waveform_peak'] < 0:
            df1.at[ind, 'waveform_km_label'] = km_labels[count]
            count = count+1
    plt.plot(df1['waveform_trough_width'][df1['waveform_peak'] < 0][df1['waveform_km_label']==0], df1['AHP'][df1['waveform_peak'] < 0][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['waveform_trough_width'][df1['waveform_peak'] < 0][df1['waveform_km_label']==1], df1['AHP'][df1['waveform_peak'] < 0][df1['waveform_km_label']==1], 'k.')
    plt.legend(['kmeans=0', 'kmeans=1'])
    plt.ylabel('AHP'); plt.xlabel('waveform trough width')
    plt.title('seperation by kmeans clustering')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if row['waveform_km_label']==0:
            plt.plot(row['norm_waveform'], 'r')
        elif row['waveform_km_label']==1:
            plt.plot(row['norm_waveform'], 'k')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    pca_in = np.zeros([379,61])
    for i in range(len(df1['norm_waveform'][df1['waveform_peak'] < 0])):
        try:
            pca_in[i,:] = df1['norm_waveform'][df1['waveform_peak'] < 0][i]
        except:
            pass
    pca = PCA(n_components=2)
    fit = pca.fit_transform(pca_in.T)
    components = pca.components_
    pca_ref = df1[df1['waveform_peak'] < 0].copy()
    i = 0
    for ind, row in pca_ref.iterrows():
        if i < 61:
            if row['waveform_km_label']==0:
                plt.plot(components[0,i].T,components[1,i],'r.')
            elif row['waveform_km_label']==1:
                plt.plot(components[0,i],components[1,i],'k.')
            i += 1
    plt.ylabel('PCA1'); plt.xlabel('PCA0')
    plt.title('seperation by PCA, with colors from kmeans')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.hist(df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0],color='r',bins=np.arange(-600,600,25),alpha=.5,orientation='horizontal')
    plt.xlabel('channels above or below center of layer 4')
    plt.hist(df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1],color='k',bins=np.arange(-600,600,25),alpha=.5,orientation='horizontal')
    plt.ylim([600,-600])
     
    plt.plot([0,14],[0,0],'k')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        tuning = row['hf1_wn_crf_tuning']
        if type(tuning) == np.ndarray or type(tuning) == list:
            tuning = [x for x in tuning if x != None]
            # thresh out units which have a small response to contrast, even if the modulation index is large
            df1.at[ind, 'responsive_to_contrast'] = np.abs(tuning[-2] - tuning[1]) > 1
        else:
            df1.at[ind, 'responsive_to_contrast'] = False
    plt.plot(df1['hf1_wn_crf_modind'][df1['responsive_to_contrast']==True][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['responsive_to_contrast']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['hf1_wn_crf_modind'][df1['responsive_to_contrast']==True][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['responsive_to_contrast']==True][df1['waveform_km_label']==1], 'k.')
    plt.xlabel('contrast response modulation index'); plt.ylabel('depth relative to layer 4'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for sf in ['low','mid','high']:
        df1['norm_ori_tuning_'+sf] = df1['hf3_gratings_ori_tuning'].copy().astype(object)
    for ind, row in df1.iterrows():
        if type(row['hf3_gratings_ori_tuning']) == list and ~np.isnan(row['hf3_gratings_drift_spont']):
            orientations = np.nanmean(np.array(row['hf3_gratings_ori_tuning'], dtype=np.float),2)
            for sfnum in range(3):
                sf = ['low','mid','high'][sfnum]
                df1.at[ind,'norm_ori_tuning_'+sf] = orientations[:,sfnum] - row['hf3_gratings_drift_spont']
            mean_for_sf = np.array([np.mean(df1.at[ind,'norm_ori_tuning_low']), np.mean(df1.at[ind,'norm_ori_tuning_mid']), np.mean(df1.at[ind,'norm_ori_tuning_high'])])
            mean_for_sf[mean_for_sf<0] = 0
            df1.at[ind,'sf_pref'] = ((mean_for_sf[0]*1)+(mean_for_sf[1]*2)+(mean_for_sf[2]*3))/np.sum(mean_for_sf)
            df1.at[ind,'responsive_to_gratings'] = [True if np.max(mean_for_sf)>2 else False][0]
        else:
            for sfnum in range(3):
                sf = ['low','mid','high'][sfnum]
                df1.at[ind,'norm_ori_tuning_'+sf] = None
            df1.at[ind,'responsive_to_gratings'] = False
            df1.at[ind,'sf_pref'] = np.nan

    use = df1#[df1['responsive_to_gratings']==True]
    for ind, row in use.iterrows():
        if row['waveform_km_label'] == 0:
            plt.plot(row['hf3_gratings_drift_spont'], row['hf4_revchecker_depth_from_layer4'],'r.')
        elif row['waveform_km_label'] == 1:
            plt.plot(row['hf3_gratings_drift_spont'], row['hf4_revchecker_depth_from_layer4'],'k.')
    plt.ylabel('depth relative to layer 4')
    plt.xlabel('gratings spont rate')
    plt.title('spont rate vs depth'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    # plot prefered spatial frequency for all gratings-responsive units
    plt.hist(df1['sf_pref'][df1['responsive_to_gratings']==True], bins=np.arange(1,3.25,0.25))
    plt.xlabel('prefered spatial frequency'); plt.ylabel('unit count')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['fm1_wn_spike_rate_vs_gz_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['fm1_wn_spike_rate_vs_gz_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('positive head gz tuning modulation index'); plt.xlabel('prefered orientation selectivity index')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['fm1_wn_spike_rate_vs_gz_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['fm1_wn_spike_rate_vs_gz_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('positive head gz tuning modulation index'); plt.xlabel('prefered orientation selectivity index')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['fm1_wn_spike_rate_vs_gx_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['fm1_wn_spike_rate_vs_gx_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('positive head gx tuning modulation index'); plt.xlabel('prefered orientation selectivity index')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['fm1_wn_spike_rate_vs_gx_modind_neg'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['fm1_wn_spike_rate_vs_gx_modind_neg'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('negative head gx tuning modulation index'); plt.xlabel('prefered orientation selectivity index')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['fm1_wn_spike_rate_vs_gy_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['fm1_wn_spike_rate_vs_gy_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('positive head gy tuning modulation index'); plt.xlabel('prefered orientation selectivity index')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['fm1_wn_spike_rate_vs_gy_modind_neg'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['fm1_wn_spike_rate_vs_gy_modind_neg'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('negative head gy tuning modulation index'); plt.xlabel('prefered orientation selectivity index')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['fm1_wn_spike_rate_vs_roll_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['fm1_wn_spike_rate_vs_roll_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('positive head roll tuning modulation index'); plt.xlabel('prefered orientation selectivity index')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['fm1_wn_spike_rate_vs_roll_modind_neg'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['fm1_wn_spike_rate_vs_roll_modind_neg'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('negative head roll tuning modulation index'); plt.xlabel('prefered orientation selectivity index')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['fm1_wn_spike_rate_vs_pitch_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['fm1_wn_spike_rate_vs_pitch_modind_pos'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('positive head pitch tuning modulation index'); plt.xlabel('prefered orientation selectivity index')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['fm1_wn_spike_rate_vs_pitch_modind_neg'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['fm1_wn_spike_rate_vs_pitch_modind_neg'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('negative head pitch tuning modulation index'); plt.xlabel('prefered orientation selectivity index')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['osi_for_sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('depth relative to layer 4'); plt.xlabel('prefered orientation selectivity index'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.plot(df1['sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('depth relative to layer 4'); plt.xlabel('pref sf'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.plot(df1['fm1_wn_spike_rate_vs_roll_modind_pos'][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['fm1_wn_spike_rate_vs_roll_modind_pos'][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('depth relative to layer 4'); plt.xlabel('positive head roll tuning modulation index'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.plot(df1['fm1_wn_spike_rate_vs_roll_modind_neg'][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['fm1_wn_spike_rate_vs_roll_modind_neg'][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('depth relative to layer 4'); plt.xlabel('negative head roll tuning modulation index'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.plot(df1['fm1_wn_spike_rate_vs_pitch_modind_pos'][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['fm1_wn_spike_rate_vs_pitch_modind_pos'][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('depth relative to layer 4'); plt.xlabel('positive head pitch tuning modulation index'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.plot(df1['fm1_wn_spike_rate_vs_pitch_modind_neg'][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['fm1_wn_spike_rate_vs_pitch_modind_neg'][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('depth relative to layer 4'); plt.xlabel('negative head pitch tuning modulation index'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()
    
    plt.figure()
    plt.plot(df1['fm1_wn_spike_rate_vs_theta_modind_pos'][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['fm1_wn_spike_rate_vs_theta_modind_pos'][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('depth relative to layer 4'); plt.xlabel('positive eye theta tuning modulation index'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.plot(df1['fm1_wn_spike_rate_vs_theta_modind_neg'][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['fm1_wn_spike_rate_vs_theta_modind_neg'][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('depth relative to layer 4'); plt.xlabel('negative eye theta tuning modulation index'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.plot(df1['fm1_wn_spike_rate_vs_phi_modind_pos'][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['fm1_wn_spike_rate_vs_phi_modind_pos'][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('depth relative to layer 4'); plt.xlabel('positive eye phi tuning modulation index'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.figure()
    plt.plot(df1['fm1_wn_spike_rate_vs_phi_modind_neg'][df1['waveform_km_label']==0], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1['fm1_wn_spike_rate_vs_phi_modind_neg'][df1['waveform_km_label']==1], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1], 'k.')
    plt.ylabel('depth relative to layer 4'); plt.xlabel('negative eye phi tuning modulation index'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    crfs0 = np.zeros([len(df1['hf1_wn_crf_tuning'][df1['waveform_km_label']==0]),11])
    crfs1 = np.zeros([len(df1['hf1_wn_crf_tuning'][df1['waveform_km_label']==1]),11])
    for i in range(len(df1['hf1_wn_crf_tuning'][df1['waveform_km_label']==0])):
        try:
            crfs0[i,:] = df1['hf1_wn_crf_tuning'][df1['waveform_km_label']==0][i]
        except:
            pass
    for i in range(len(df1['hf1_wn_crf_tuning'][df1['waveform_km_label']==1])):
        try:
            crfs1[i,:] = df1['hf1_wn_crf_tuning'][df1['waveform_km_label']==1][i]
        except:
            pass
    
    plt.figure()
    plt.title('contrast spont rate'); plt.ylabel('depth relative to layer 4'); plt.xlabel('contrast spont rate (sp/sec)')
    plt.plot(crfs0[:,4], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0], 'r.')
    plt.plot(crfs1[:,4], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1], 'k.'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()
    
    plt.figure()
    plt.title('contrast evoked rate'); plt.ylabel('depth relative to layer 4'); plt.xlabel('contrast evoked rate (sp/sec)')
    plt.plot(crfs0[:,-2], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==0], 'r.')
    plt.plot(crfs1[:,-2], df1['hf4_revchecker_depth_from_layer4'][df1['waveform_km_label']==1], 'k.'); plt.ylim([600,-600])
    plt.tight_layout(); pdf.savefig(); plt.close()

    print('saving population summary pdf')
    pdf.close()

    print('saving updated version of df')
    newdf.to_hdf(os.path.join(savepath, 'pooled_ephys_population_update_'+datetime.today().strftime('%m%d%y')+'.h5'), 'w')

    print('done')

def population_analysis(config):
    """
    load ephys data from all 'good' sessions
    summarize the data on the level of sessions, individual units, and then as a population of units
    INPUTS
        config: options dict
    OUTPUTS
        None
    """
    print('pooling ephys data')
    df = load_ephys(config['population']['metadata_csv_path'])
    # clean up h5 file
    cols = df.columns.values
    shcols = [c for c in cols if 'gratingssh' in c]
    for c in shcols:
        new_col = str(c.replace('gratingssh', 'gratings'))
        df = df.rename(columns={str(c): new_col})
    badcols = []
    for c in cols:
        if any(s in c for s in ['fm2','hf5','hf6','hf7','hf8']):
            badcols.append(c)
    df = df.drop(labels=badcols, axis=1)
    df = df.groupby(lambda x:x, axis=1); df = df.agg(np.nansum) # combine identical column names
    print('saving pooled ephys data to '+config['population']['save_path'])
    json_path = os.path.join(config['population']['save_path'],'pooled_ephys_'+datetime.today().strftime('%m%d%y')+'.json')
    if os.path.isfile(json_path):
        os.remove(json_path)
    df = df.reset_index()
    df.to_json(json_path, default_handler=str)
    print('writing session summary')
    make_session_summary(df, config['population']['save_path'])
    print('writing unit summary')
    unit_df = make_unit_summary(df, config['population']['save_path'])
    print('starting unit population analysis')
    make_population_summary(unit_df, config['population']['save_path'])