"""
population_utils.py
"""
import numpy as np
import ast
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import matplotlib.patches as mpatches
import itertools

from utils.ephys import load_ephys
from utils.format_data import flatten_series

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
        r0 = tuning[4]
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
    # set up new file to save out including new metrics
    newdf = df.copy().reset_index()
    print('num units = ' + str(len(df)))
    # iterate through units
    for index, row in tqdm(df.iterrows()):
        lightfm = 'fm1'
        darkfm = 'fm_dark'
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
            newdf.at[index, 'hf1_wn_crf_modind'] = modind
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
                newdf.at[index, 'hf4_revchecker_ch_lfp_relative_depth'] = position_of_ch
                depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
                newdf.at[index, 'hf4_revchecker_depth_from_layer4'] = depth_from_surface
                unitfig_lfp.set_title('ch='+str(row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
            elif shank_num == 1:
                position_of_ch = int(row['hf4_revchecker_lfp_rel_depth'][1][row['ch']-32])
                newdf.at[index, 'hf4_revchecker_ch_lfp_relative_depth'] = position_of_ch
                depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
                newdf.at[index, 'hf4_revchecker_depth_from_layer4'] = depth_from_surface
                unitfig_lfp.set_title('ch='+str(row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
            elif shank_num == 2:
                position_of_ch = int(row['hf4_revchecker_lfp_rel_depth'][1][row['ch']-64])
                newdf.at[index, 'hf4_revchecker_ch_lfp_relative_depth'] = position_of_ch
                depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
                newdf.at[index, 'hf4_revchecker_depth_from_layer4'] = depth_from_surface
                unitfig_lfp.set_title('ch='+str(row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
            elif shank_num == 3:
                position_of_ch = int(row['hf4_revchecker_lfp_rel_depth'][1][row['ch']-96])
                newdf.at[index, 'hf4_revchecker_ch_lfp_relative_depth'] = position_of_ch
                depth_from_surface = int(depth_to_layer4 + (ch_spacing * position_of_ch))
                newdf.at[index, 'hf4_revchecker_depth_from_layer4'] = depth_from_surface
                unitfig_lfp.set_title('ch='+str(row['ch'])+'\npos='+str(position_of_ch)+'\ndist2layer4='+str(depth_from_surface), fontsize=20)
            unitfig_lfp.legend(); unitfig_lfp.axvline(x=(0.1*30000), color='k', linewidth=1)
            unitfig_lfp.set_xticks(np.arange(0,18000,18000/8))
            unitfig_lfp.set_xticklabels(np.arange(-100,500,75))
            unitfig_lfp.set_xlabel('msec'); unitfig_lfp.set_ylabel('uvolts')
        except:
            pass
        try:
            # depth from LFP power profile using wn stim
            power_profiles = row['hf1_wn_lfp_power_profiles']
            ch_shank = int(np.floor(row['ch']/32))
            ch_shank_profile = power_profiles[ch_shank]
            ch_power = ch_shank_profile[int(row['ch']%32)]
            layer5cent = row['hf1_wn_lfp_layer5_centers'][ch_shank]
            if row['probe_name'] == 'DB_P64-8':
                ch_spacing = 25/2
            else:
                ch_spacing = 25
            ch_depth = ch_spacing*(row['ch']%32)-(layer5cent*ch_spacing)
            num_sites = 32
            unitfig_power_depth = unitfig.add_subplot(spec[6:8, 4])
            unitfig_power_depth.plot(ch_shank_profile,range(0,num_sites))
            unitfig_power_depth.plot(ch_shank_profile[layer5cent]+0.01,layer5cent,'r*',markersize=12)
            unitfig_power_depth.hlines(y=row['ch']%32, xmin=0, xmax=ch_power, colors='g', linewidth=5)
            unitfig_power_depth.set_ylim([33,-1]); unitfig_power_depth.set_yticks(list(range(-1,num_sites+1))); unitfig_power_depth.set_yticklabels(ch_spacing*np.arange(num_sites+2)-(layer5cent*ch_spacing))
            unitfig_power_depth.set_title('shank='+str(ch_shank)+' site='+str(row['ch']%32)+'\n depth='+str(ch_depth), fontsize=20)
            newdf.at[index, 'hf1_wn_depth_from_layer5'] = ch_depth
        except Exception as e:
            unitfig_power_depth = unitfig.add_subplot(spec[6:8, 4])
            unitfig_power_depth.set_title(e)
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
            unitfig_wnsaccavg.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
            unitfig_wnsaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_wnsaccavg.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
            unitfig_wnsaccavg.legend(['right','left'], loc=1)
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            unitfig_wnsaccavg.set_ylim([0,maxval*1.2])
            unitfig_wnsaccavg.set_xlim([-0.5,0.6])
            newdf.at[index, 'hf1_wn_upsacc_modind_t0'] = modind_right[0]; newdf.at[index, 'hf1_wn_downsacc_modind_t0'] = modind_left[0]
            newdf.at[index, 'hf1_wn_upsacc_modind_t100'] = modind_right[1]; newdf.at[index, 'hf1_wn_downsacc_modind_t100'] = modind_left[1]
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
            newdf.at[index, 'hf1_wn_spike_rate_vs_pupil_radius_modind'] = modind
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
            newdf.at[index, 'hf1_wn_spike_rate_vs_spd_modind'] = modind
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
            newdf.at[index, 'fm1_spike_rate_vs_gz_modind_neg'] = modind[0]; newdf.at[index, 'fm1_spike_rate_vs_gz_modind_pos'] = modind[1]
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
            newdf.at[index, 'fm1_spike_rate_vs_gx_modind_neg'] = modind[0]; newdf.at[index, 'fm1_spike_rate_vs_gx_modind_pos'] = modind[1]
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
            newdf.at[index, 'fm1_spike_rate_vs_gy_modind_neg'] = modind[0]; newdf.at[index, 'fm1_spike_rate_vs_gy_modind_pos'] = modind[1]
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
            unitfig_fm1upsacc_gazedEye.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
            unitfig_fm1upsacc_gazedEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_gazedEye.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
            unitfig_fm1upsacc_gazedEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_gazedEye.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_gazedEye.set_ylabel('sp/sec')
            unitfig_fm1upsacc_gazedEye.set_xlim([-0.5,0.6])
            newdf.at[index, 'fm1_upsacc_avg_gaze_shift_dEye_modind_t0'] = modind_right[0]; newdf.at[index, 'fm1_downsacc_avg_gaze_shift_dEye_modind_t0'] = modind_left[0]
            newdf.at[index, 'fm1_upsacc_avg_gaze_shift_dEye_modind_t100'] = modind_right[1]; newdf.at[index, 'fm1_downsacc_avg_gaze_shift_dEye_modind_t100'] = modind_left[1]
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
            unitfig_fm1upsacc_compdEye.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
            unitfig_fm1upsacc_compdEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_compdEye.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
            unitfig_fm1upsacc_compdEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_compdEye.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_compdEye.set_ylabel('sp/sec')
            unitfig_fm1upsacc_compdEye.set_xlim([-0.5,0.6])
            newdf.at[index, 'fm1_upsacc_avg_comp_dEye_modind_t0'] = modind_right[0]; newdf.at[index, 'fm1_downsacc_avg_comp_dEye_modind_t0'] = modind_left[0]
            newdf.at[index, 'fm1_upsacc_avg_comp_dEye_modind_t100'] = modind_right[1]; newdf.at[index, 'fm1_downsacc_avg_comp_dEye_modind_t100'] = modind_left[1]
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
            unitfig_fm1upsacc_gazedHead.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
            unitfig_fm1upsacc_gazedHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_gazedHead.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
            unitfig_fm1upsacc_gazedHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_gazedHead.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_gazedHead.set_ylabel('sp/sec')
            unitfig_fm1upsacc_gazedHead.set_xlim([-0.5,0.6])
            newdf.at[index, 'fm1_upsacc_avg_gaze_shift_dHead_modind_t0'] = modind_right[0]; newdf.at[index, 'fm1_downsacc_avg_gaze_shift_dHead_modind_t0'] = modind_left[0]
            newdf.at[index, 'fm1_upsacc_avg_gaze_shift_dHead_modind_t100'] = modind_right[1]; newdf.at[index, 'fm1_downsacc_avg_gaze_shift_dHead_modind_t100'] = modind_left[1]
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
            unitfig_fm1upsacc_compdHead.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
            unitfig_fm1upsacc_compdHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_compdHead.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
            unitfig_fm1upsacc_compdHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_compdHead.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_compdHead.set_ylabel('sp/sec')
            unitfig_fm1upsacc_compdHead.set_xlim([-0.5,0.6])
            newdf.at[index, 'fm1_upsacc_avg_comp_dHead_modind_t0'] = modind_right[0]; newdf.at[index, 'fm1_downsacc_avg_comp_dHead_modind_t0'] = modind_left[0]
            newdf.at[index, 'fm1_upsacc_avg_comp_dHead_modind_t100'] = modind_right[1]; newdf.at[index, 'fm1_downsacc_avg_comp_dHead_modind_t100'] = modind_left[1]
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
            dsi = np.zeros([3])
            for sf in range(3):
                R_pref = (tuning[th_pref[sf], sf] + (tuning[(th_pref[sf]+4)%8, sf])) * 0.5 # get that firing rate (avg between peaks)
                th_ortho = (th_pref[sf]+2)%8 # get ortho position
                R_ortho = (tuning[th_ortho, sf] + (tuning[(th_ortho+4)%8, sf])) * 0.5 # ortho firing rate (average between two peaks)
                # orientaiton selectivity index
                osi[sf] = (R_pref - R_ortho) / (R_pref + R_ortho)
                # direction selectivity index
                th_null = (th_pref[sf]+4)%8 # get other direction of same orientation
                R_null = tuning[th_null, sf] # tuning value at that peak
                dsi[sf] = (R_pref - R_null) / (R_pref + R_null)
            unitfig_ori_tuning.set_title('orientation tuning\n OSI l='+str(np.round(osi[0],3))+'m='+str(np.round(osi[1],3))+'h='+str(np.round(osi[2],3))+'\n DSI l='+str(np.round(dsi[0],3))+'m='+str(np.round(dsi[1],3))+'h='+str(np.round(dsi[2],3)), fontsize=20)
            unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,0],label = 'low sf')
            unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,1],label = 'mid sf')
            unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,2],label = 'hi sf')
            unitfig_ori_tuning.plot([0,315],[drift_spont,drift_spont],'r:',label='spont')
            unitfig_ori_tuning.legend()
            unitfig_ori_tuning.set_ylim([0,np.nanmax(row['hf3_gratings_ori_tuning'][:,:,:])*1.2])
            newdf.at[index, 'hf3_gratings_osi_low'] = osi[0]; newdf.at[index, 'hf3_gratings_osi_mid'] = osi[1]; newdf.at[index, 'hf3_gratings_osi_high'] = osi[2]
            newdf.at[index, 'hf3_gratings_dsi_low'] = dsi[0]; newdf.at[index, 'hf3_gratings_dsi_mid'] = dsi[1]; newdf.at[index, 'hf3_gratings_dsi_high'] = dsi[2]
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
                osi[sf] = (R_pref - R_ortho) / (R_pref + R_ortho)
                th_null = (th_pref[sf]+4)%8 # get other direction of same orientation
                R_null = tuning[th_null, sf] # tuning value at that peak
                dsi[sf] = (R_pref - R_null) / (R_pref + R_null)
            unitfig_ori_tuning.set_title('low tf\n OSI l='+str(np.round(osi[0],3))+'m='+str(np.round(osi[1],3))+'h='+str(np.round(osi[2],3))+'\n DSI l='+str(np.round(dsi[0],3))+'m='+str(np.round(dsi[1],3))+'h='+str(np.round(dsi[2],3)), fontsize=20)
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
                osi[sf] = (R_pref - R_ortho) / (R_pref + R_ortho)
                th_null = (th_pref[sf]+4)%8 # get other direction of same orientation
                R_null = tuning[th_null, sf] # tuning value at that peak
                dsi[sf] = (R_pref - R_null) / (R_pref + R_null)
            unitfig_ori_tuning.set_title('high tf\n OSI l='+str(np.round(osi[0],3))+'m='+str(np.round(osi[1],3))+'h='+str(np.round(osi[2],3))+'\n DSI l='+str(np.round(dsi[0],3))+'m='+str(np.round(dsi[1],3))+'h='+str(np.round(dsi[2],3)), fontsize=20)
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
            unitfig_fm1saccavg.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
            unitfig_fm1saccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1saccavg.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            unitfig_fm1saccavg.set_ylim([0,maxval*1.2])
            unitfig_fm1saccavg.set_xlim([-0.5,0.6])
            newdf.at[index, 'fm1_upsacc_modind_t0'] = modind_right[0]; newdf.at[index, 'fm1_downsacc_modind_t0'] = modind_left[0]
            newdf.at[index, 'fm1_upsacc_modind_t100'] = modind_right[1]; newdf.at[index, 'fm1_downsacc_modind_t100'] = modind_left[1]
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
            newdf.at[index, 'hf1_wn_spike_rate_vs_pupil_radius_modind'] = modind
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
            newdf.at[index, 'fm1_spike_rate_vs_theta_modind_neg'] = modind[0]; newdf.at[index, 'fm1_spike_rate_vs_theta_modind_pos'] = modind[1]
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
            newdf.at[index, 'fm1_spike_rate_vs_phi_modind_neg'] = modind[0]; newdf.at[index, 'fm1_spike_rate_vs_phi_modind_pos'] = modind[1]
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
            newdf.at[index, 'fm1_spike_rate_vs_roll_modind_neg'] = modind[0]; newdf.at[index, 'fm1_spike_rate_vs_roll_modind_pos'] = modind[1]
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
            newdf.at[index, 'fm1_spike_rate_vs_pitch_modind_neg'] = modind[0]; newdf.at[index, 'fm1_spike_rate_vs_pitch_modind_pos'] = modind[1]
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
                newdf.at[index, darkfm+'_wn_spike_rate_vs_gz_modind_neg'] = modind[0]; newdf.at[index, darkfm+'_spike_rate_vs_gz_modind_pos'] = modind[1]
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
                newdf.at[index, darkfm+'_spike_rate_vs_gx_modind_neg'] = modind[0]; newdf.at[index, 'fm1_spike_rate_vs_gx_modind_pos'] = modind[1]
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
                newdf.at[index, darkfm+'_spike_rate_vs_gy_modind_neg'] = modind[0]; newdf.at[index, darkfm+'_spike_rate_vs_gy_modind_pos'] = modind[1]
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
                unitfig_darkupsacc_gazedEye.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
                unitfig_darkupsacc_gazedEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
                unitfig_darkupsacc_gazedEye.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
                unitfig_darkupsacc_gazedEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
                unitfig_darkupsacc_gazedEye.set_ylim([0,maxval*1.2])
                unitfig_darkupsacc_gazedEye.set_ylabel('sp/sec')
                unitfig_darkupsacc_gazedEye.set_xlim([-0.5,0.6])
                newdf.at[index, darkfm+'_upsacc_avg_gaze_shift_dEye_modind_t0'] = modind_right[0]; newdf.at[index, darkfm+'_downsacc_avg_gaze_shift_dEye_modind_t0'] = modind_left[0]
                newdf.at[index, darkfm+'_upsacc_avg_gaze_shift_dEye_modind_t100'] = modind_right[1]; newdf.at[index, darkfm+'_downsacc_avg_gaze_shift_dEye_modind_t100'] = modind_left[1]
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
                unitfig_darkupsacc_compdEye.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
                unitfig_darkupsacc_compdEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
                unitfig_darkupsacc_compdEye.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
                unitfig_darkupsacc_compdEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
                unitfig_darkupsacc_compdEye.set_ylim([0,maxval*1.2])
                unitfig_darkupsacc_compdEye.set_ylabel('sp/sec')
                unitfig_darkupsacc_compdEye.set_xlim([-0.5,0.6])
                newdf.at[index, darkfm+'_upsacc_avg_comp_dEye_modind_t0'] = modind_right[0]; newdf.at[index, darkfm+'_downsacc_avg_comp_dEye_modind_t0'] = modind_left[0]
                newdf.at[index, darkfm+'_upsacc_avg_comp_dEye_modind_t100'] = modind_right[1]; newdf.at[index, darkfm+'_downsacc_avg_comp_dEye_modind_t100'] = modind_left[1]
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
                unitfig_darkupsacc_gazedHead.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
                unitfig_darkupsacc_gazedHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
                unitfig_darkupsacc_gazedHead.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
                unitfig_darkupsacc_gazedHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
                unitfig_darkupsacc_gazedHead.set_ylim([0,maxval*1.2])
                unitfig_darkupsacc_gazedHead.set_ylabel('sp/sec')
                unitfig_darkupsacc_gazedHead.set_xlim([-0.5,0.6])
                newdf.at[index, darkfm+'_wn_upsacc_avg_gaze_shift_dHead_modind_t0'] = modind_right[0]; newdf.at[index, darkfm+'_downsacc_avg_gaze_shift_dHead_modind_t0'] = modind_left[0]
                newdf.at[index, darkfm+'_upsacc_avg_gaze_shift_dHead_modind_t100'] = modind_right[1]; newdf.at[index, darkfm+'_downsacc_avg_gaze_shift_dHead_modind_t100'] = modind_left[1]
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
                unitfig_darkupsacc_compdHead.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
                unitfig_darkupsacc_compdHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
                unitfig_darkupsacc_compdHead.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
                unitfig_darkupsacc_compdHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
                unitfig_darkupsacc_compdHead.set_ylim([0,maxval*1.2])
                unitfig_darkupsacc_compdHead.set_ylabel('sp/sec')
                unitfig_darkupsacc_compdHead.set_xlim([-0.5,0.6])
                newdf.at[index, darkfm+'_upsacc_avg_comp_dHead_modind_t0'] = modind_right[0]; newdf.at[index, darkfm+'_downsacc_avg_comp_dHead_modind_t0'] = modind_left[0]
                newdf.at[index, darkfm+'_upsacc_avg_comp_dHead_modind_t100'] = modind_right[1]; newdf.at[index, darkfm+'_downsacc_avg_comp_dHead_modind_t100'] = modind_left[1]
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
                unitfig_darksaccavg.annotate('0ms='+str(modind_right[0])+' 100ms='+str(modind_right[1]),color='#1f77b4', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=15)
                unitfig_darksaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
                unitfig_darksaccavg.annotate('0ms='+str(modind_left[0])+' 100ms='+str(modind_left[1]),color='r', xy=(0.05, 0.87), xycoords='axes fraction', fontsize=15)
                maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
                unitfig_darksaccavg.set_ylim([0,maxval*1.2])
                unitfig_darksaccavg.set_xlim([-0.5,0.6])
                newdf.at[index, darkfm+'_upsacc_modind_t0'] = modind_right[0]; newdf.at[index, darkfm+'_downsacc_modind_t0'] = modind_left[0]
                newdf.at[index, darkfm+'_upsacc_modind_t100'] = modind_right[1]; newdf.at[index, darkfm+'_downsacc_modind_t100'] = modind_left[1]
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
                newdf.at[index, darkfm+'_spike_rate_vs_pupil_radius_modind'] = modind
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
                newdf.at[index, darkfm+'_spike_rate_vs_theta_modind_neg'] = modind[0]; newdf.at[index, darkfm+'_spike_rate_vs_theta_modind_pos'] = modind[1]
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
                newdf.at[index, 'fm1_spike_rate_vs_phi_modind_neg'] = modind[0]; newdf.at[index, 'fm1_spike_rate_vs_phi_modind_pos'] = modind[1]
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
                newdf.at[index, darkfm+'_spike_rate_vs_roll_modind_neg'] = modind[0]; newdf.at[index, darkfm+'_spike_rate_vs_roll_modind_pos'] = modind[1]
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
                newdf.at[index, darkfm+'_spike_rate_vs_pitch_modind_neg'] = modind[0]; newdf.at[index, darkfm+'_spike_rate_vs_pitch_modind_pos'] = modind[1]
            except:
                pass

        plt.tight_layout()
        pdf.savefig(unitfig)
        plt.close()

    print('saving unit summary pdf')
    pdf.close()

    print('saving an updated json')
    newdf.to_pickle(os.path.join(savepath, 'pooled_ephys_unit_update_'+datetime.today().strftime('%m%d%y')+'.pickle'))

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
    df = df.set_index('session')

    unique_inds = sorted(list(set(df.index.values)))

    for unique_ind in tqdm(unique_inds):
        uniquedf = df.loc[unique_ind]
        # set up subplots
        plt.subplots(4,5,figsize=(25,25))
        plt.suptitle(unique_ind+' eye fit: m='+str(uniquedf['best_ellipse_fit_m'].iloc[0])+' r='+str(uniquedf['best_ellipse_fit_r'].iloc[0]), fontsize=20)
        # eye position vs head position
        try:
            plt.subplot(4,5,1)
            plt.title('dEye vs dHead (LIGHT)', fontsize=20)
            dEye = uniquedf['fm1_dEye'].iloc[0]
            dhead = uniquedf['fm1_dHead'].iloc[0]
            eyeT = uniquedf['fm1_eyeT'].iloc[0]
            if len(dEye[0:-1:10]) == len(dhead(eyeT[0:-1:10])):
                plt.plot(dEye[0:-1:10],dhead(eyeT[0:-1:10]),'.')
            elif len(dEye[0:-1:10]) > len(dhead(eyeT[0:-1:10])):
                plt.plot(dEye[0:-1:10][:len(dhead(eyeT[0:-1:10]))],dhead(eyeT[0:-1:10]),'.')
            elif len(dEye[0:-1:10]) < len(dhead(eyeT[0:-1:10])):
                plt.plot(dEye[0:-1:10],dhead(eyeT[0:-1:10])[:len(dEye[0:-1:10])],'.')
            plt.xlabel('dEye'); plt.ylabel('dHead'); plt.xlim((-15,15)); plt.ylim((-15,15))
            plt.plot([-15,15],[15,-15], 'r')
        except Exception as e:
            print('deye/dhead light', e)
        try:
            plt.subplot(4,5,2)
            plt.title('dEye vs dHead (DARK)', fontsize=20)
            dEye = uniquedf['fm_dark_dEye'].iloc[0]
            dhead = uniquedf['fm_dark_dHead'].iloc[0]
            eyeT = uniquedf['fm_dark_eyeT'].iloc[0]
            plt.plot(dEye[0:-1:10],dhead(eyeT[0:-1:10]),'.')
            plt.xlabel('dEye'); plt.ylabel('dHead'); plt.xlim((-15,15)); plt.ylim((-15,15))
            plt.plot([-15,15],[15,-15], 'r')
        except Exception as e:
            print('deye/dhead dark', e)
        try:
            accT = uniquedf['fm1_accT'].iloc[0]
            roll_interp = uniquedf['fm1_roll_interp'].iloc[0]
            pitch_interp = uniquedf['fm1_pitch_interp'].iloc[0]
            th = uniquedf['fm1_theta'].iloc[0]
            phi = uniquedf['fm1_phi'].iloc[0]
            plt.subplot(4,5,3)
            plt.title('LIGHT', fontsize=20)
            plt.plot(pitch_interp[::100], th[::100], '.'); plt.xlabel('pitch'); plt.ylabel('theta')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
            plt.subplot(4,5,4)
            plt.plot(roll_interp[::100], phi[::100], '.'); plt.xlabel('roll'); plt.ylabel('phi')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')
        except Exception as e:
            print('theta/phi vs roll/pitch light', e)
        try:
            dark_roll_interp = uniquedf['fm_dark_roll_interp'].iloc[0]
            dark_pitch_interp = uniquedf['fm_dark_pitch_interp'].iloc[0]
            th_dark = uniquedf['fm_dark_theta'].iloc[0]
            phi_dark = uniquedf['fm_dark_phi'].iloc[0]
            plt.subplot(4,5,5)
            plt.plot(dark_pitch_interp[::100], th_dark[::100], '.'); plt.xlabel('pitch'); plt.ylabel('theta')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
            plt.title('DARK', fontsize=20)
            plt.subplot(4,5,6)
            plt.plot(dark_roll_interp[::100], phi_dark[::100], '.'); plt.xlabel('roll'); plt.ylabel('phi')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')
        except Exception as e:
            print('theta/phi vs roll/pitch dark', e)
        try:
            # histogram of theta from -45 to 45deg (are eye movements in resonable range?)
            plt.subplot(4,5,7)
            plt.title('hist of FM theta', fontsize=20)
            plt.hist(uniquedf['fm1_theta'].iloc[0], range=[-45,45])
            # histogram of phi from -45 to 45deg (are eye movements in resonable range?)
            plt.subplot(4,5,8)
            plt.title('hist of FM phi', fontsize=20)
            plt.hist(uniquedf['fm1_phi'].iloc[0], range=[-45,45])
            # histogram of gyro z (resonable range?)
            plt.subplot(4,5,9)
            plt.title('hist of FM gyro z', fontsize=20)
            plt.hist(uniquedf['fm1_gz'].iloc[0], range=[2,4])
            # plot of contrast response functions on same panel scaled to max 30sp/sec
            # plot of average contrast reponse function across units
            plt.subplot(4,5,10)
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
                        plt.subplot(4,5,11)
                        plt.plot(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0][ch_num], color=colors[ch_num], linewidth=1)
                        plt.title('ch1:32', fontsize=20); plt.axvline(x=(0.1*30000))
                        plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                        plt.ylim([-1200,400])
                    if ch_num>31:
                        plt.subplot(4,5,12)
                        plt.plot(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0][ch_num], color=colors[ch_num-32], linewidth=1)
                        plt.title('ch33:64', fontsize=20); plt.axvline(x=(0.1*30000))
                        plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                        plt.ylim([-1200,400])
            elif num_channels == 128:
                for ch_num in np.arange(0,128):
                    if ch_num < 32:
                        plt.subplot(4,5,11)
                        plt.plot(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0][ch_num], color=colors[ch_num], linewidth=1)
                        plt.title('ch1:32'); plt.axvline(x=(0.1*30000))
                        plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                    elif 32 <= ch_num < 64:
                        plt.subplot(4,5,12)
                        plt.plot(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0][ch_num], color=colors[ch_num-32], linewidth=1)
                        plt.title('ch33:64'); plt.axvline(x=(0.1*30000))
                        plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                    elif 64 <= ch_num < 96:
                        plt.subplot(4,5,13)
                        plt.plot(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0][ch_num], color=colors[ch_num-64], linewidth=1)
                        plt.title('ch33:64'); plt.axvline(x=(0.1*30000))
                        plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                    elif 96 <= ch_num < 128:
                        plt.subplot(4,5,14)
                        plt.plot(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0][ch_num], color=colors[ch_num-96], linewidth=1)
                        plt.title('ch33:64'); plt.axvline(x=(0.1*30000))
                        plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
            # fm spike raster
            plt.subplot(4,5,15)
            plt.title('FM raster', fontsize=20)
            i = 0
            for ind, row in uniquedf.iterrows():
                plt.vlines(row['fm1_spikeT'],i-0.25,i+0.25)
                plt.xlim(0, 10); plt.xlabel('secs'); plt.ylabel('unit #')
                i = i+1
        except Exception as e:
            print(e)
        # all psth plots in a single panel, with avg plotted over the top
        plt.subplot(4,5,16)
        lower = -0.5; upper = 1.5; dt = 0.1
        bins = np.arange(lower,upper+dt,dt)
        psth_list = []
        for ind, row in uniquedf.iterrows():
            try:
                plt.plot(bins[0:-1]+dt/2,row['hf3_gratings_grating_psth'])
                psth_list.append(row['hf3_gratings_grating_psth'])
            except Exception as e:
                print(e)
        try:
            avg_psth = np.mean(np.array(psth_list), axis=0)
            plt.plot(bins[0:-1]+dt/2,avg_psth,color='k',linewidth=6)
            plt.title('gratings psth', fontsize=20); plt.xlabel('time'); plt.ylabel('sp/sec')
            plt.ylim([0,np.nanmax(avg_psth)*1.5])
        except Exception as e:
            print(e)
        try:
            lfp_power_profile = uniquedf['hf1_wn_lfp_power_profiles'].iloc[0]
            layer5_cent = uniquedf['hf1_wn_lfp_layer5_centers'].iloc[0]
            if type(lfp_power_profile) == list:
                if uniquedf['probe_name'].iloc[0] == 'DB_P64-8':
                    ch_spacing = 25/2
                else:
                    ch_spacing = 25
                if '64' in uniquedf['probe_name'].iloc[0]:
                    norm_profile_sh0 = lfp_power_profile[0]
                    layer5_cent_sh0 = layer5_cent[0]
                    norm_profile_sh1 = lfp_power_profile[1]
                    layer5_cent_sh1 = layer5_cent[1]
                    plt.subplot(4,5,17)
                    plt.plot(norm_profile_sh0,range(0,32))
                    plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
                    plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
                    plt.title('shank0')
                    plt.subplot(4,5,18)
                    plt.plot(norm_profile_sh1,range(0,32))
                    plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
                    plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
                    plt.title('shank1')
                if '16' in uniquedf['probe_name'].iloc[0]:
                    norm_profile_sh0 = lfp_power_profile[0]
                    layer5_cent_sh0 = layer5_cent[0]
                    plt.subplot(4,5,17)
                    plt.tight_layout()
                    plt.plot(norm_profile_sh0,range(0,16))
                    plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
                    plt.ylim([17,-1]); plt.yticks(ticks=list(range(-1,17)),labels=(ch_spacing*np.arange(18)-(layer5_cent_sh0*ch_spacing)))
                    plt.title('shank0')
                if '128' in uniquedf['probe_name'].iloc[0]:
                    norm_profile_sh0 = lfp_power_profile[0]
                    layer5_cent_sh0 = layer5_cent[0]
                    norm_profile_sh1 = lfp_power_profile[1]
                    layer5_cent_sh1 = layer5_cent[1]
                    norm_profile_sh2 = lfp_power_profile[2]
                    layer5_cent_sh2 = layer5_cent[2]
                    norm_profile_sh3 = lfp_power_profile[3]
                    layer5_cent_sh3 = layer5_cent[3]
                    plt.subplot(4,5,17)
                    plt.plot(norm_profile_sh0,range(0,32))
                    plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
                    plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
                    plt.title('shank0')
                    plt.subplot(4,5,18)
                    plt.plot(norm_profile_sh1,range(0,32))
                    plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
                    plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
                    plt.title('shank1')
                    plt.subplot(4,5,19)
                    plt.plot(norm_profile_sh2,range(0,32))
                    plt.plot(norm_profile_sh2[layer5_cent_sh2]+0.01,layer5_cent_sh2,'r*',markersize=12)
                    plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh2*ch_spacing)))
                    plt.title('shank2')
                    plt.subplot(4,5,20)
                    plt.plot(norm_profile_sh3,range(0,32))
                    plt.plot(norm_profile_sh3[layer5_cent_sh3]+0.01,layer5_cent_sh3,'r*',markersize=12)
                    plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh3*ch_spacing)))
                    plt.title('shank3')
        except Exception as e:
            print(e)
        pdf.savefig()
    print('saving session summary pdf')
    pdf.close()

def plot_pop_vars(df1, varX, varY):
    fig = plt.figure()
    plt.plot(df1[varX][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], df1[varY][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], 'r.')
    plt.plot(df1[varX][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], df1[varY][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], 'k.')
    plt.ylabel(varY); plt.xlabel(varX)
    return fig

def plot_var_vs_var(df1, xvar, yvar, n, filter_for=None, force_range=None, along_y=False, use_median=False, abs=False, xpos_to_index=None):
    """
    filter_for: dict of value to require for a column of the dataframe
    """
    # plt.rcParams.update({'font.size': 22})
    fig = plt.subplot(n[0],n[1],n[2])
    if force_range is None:
        force_range = np.arange(0,0.310,0.05)
    for km in range(2):
        if km == 0:
            c = 'g'
        elif km == 1:
            c = 'b'
        x = df1[xvar][df1['waveform_km_label']==km]
        if xpos_to_index is not None:
            for ind, row in x.iteritems():
                x.at[ind] = row[xpos_to_index]
        if abs==True:
            x = np.abs(x)
        y = df1[yvar][df1['waveform_km_label']==km]
        if filter_for is not None:
            for key, val in filter_for.items():
                x = x[df1[key]==val]
                y = y[df1[key]==val]
        x = x.to_numpy().astype(float)
        y = y.to_numpy().astype(float)
        if use_median == False:
            stat2use = np.nanmean
        elif use_median == True:
            stat2use = 'median'
        if along_y == False:
            bin_means, bin_edges, bin_number = stats.binned_statistic(x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)], statistic=stat2use, bins=force_range)
            bin_std, _, _ = stats.binned_statistic(x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)], statistic=np.nanstd, bins=force_range)
            hist, _ = np.histogram(x[~np.isnan(x) & ~np.isnan(y)], bins=force_range)
        elif along_y == True:
            bin_means, bin_edges, bin_number = stats.binned_statistic(y[~np.isnan(x) & ~np.isnan(y)], x[~np.isnan(x) & ~np.isnan(y)], statistic=stat2use, bins=force_range)
            bin_std, _, _ = stats.binned_statistic(y[~np.isnan(x) & ~np.isnan(y)], x[~np.isnan(x) & ~np.isnan(y)], statistic=np.nanstd, bins=force_range)
            hist, _ = np.histogram(y[~np.isnan(x) & ~np.isnan(y)], bins=force_range)
        tuning_err = bin_std / np.sqrt(hist)
        if along_y == False:
            plt.plot(x, y, c+'.')
            plt.plot(bin_edges[:-1], bin_means, c+'-')
            plt.fill_between(bin_edges[:-1], bin_means-tuning_err, bin_means+tuning_err, color=c, alpha=0.3)
            num_outliers = len([i for i in x if i>np.max(force_range) or i<np.min(force_range)])
            plt.xlim([np.min(force_range), np.max(force_range)])
        elif along_y == True:
            plt.plot(x, y, c+'.', markersize=2)
            plt.plot(bin_means, bin_edges[:-1], c+'-')
            plt.fill_betweenx(bin_edges[:-1], bin_means-tuning_err, bin_means+tuning_err, color=c, alpha=0.3)
            num_outliers = len([i for i in y if i>np.max(force_range) or i<np.min(force_range)])
            plt.gca().invert_yaxis()
    plt.title('excluded='+str(num_outliers)+' pts in data='+str(np.sum(~pd.isnull(df1[xvar]) & ~pd.isnull(df1[yvar])))+' abs='+str(abs))
    return fig

def get_peak_trough(wv, baseline):
    wv = [i-baseline for i in wv]
    wv_flip = [-i for i in wv]
    peaks, peak_props = find_peaks(wv, height=0.1)
    troughs, trough_props = find_peaks(wv_flip, height=0.1)
    if len(peaks) > 1:
        peaks = peaks[np.argmax(peak_props['peak_heights'])]
    if len(troughs) > 1:
        troughs = troughs[np.argmax(trough_props['peak_heights'])]
    if peaks.size == 0:
        peaks = np.nan
    if troughs.size == 0:
        troughs = np.nan
    if ~np.isnan(peaks):
        peaks = int(peaks)
    if ~np.isnan(troughs):
        troughs = int(troughs)
    return peaks, troughs

def get_cluster_props(p, t):
    if ~np.isnan(p):
        has_peak = True
        peak_cent = p
    else:
        has_peak = False
        peak_cent = None
    if ~np.isnan(t):
        has_trough = True
        trough_cent = t
    else:
        has_trough = False
        trough_cent = None
    if has_peak and has_trough:
        return 'biphasic'
    elif has_trough and ~has_peak:
        return 'negative'
    elif peak_cent is not None and peak_cent <= (42-38):
        return 'early'
    elif peak_cent is not None and peak_cent > (42-38):
        return 'late'
    else:
        return 'unresponsive'

def plot_cluster_prop(df1, cluster_prop, waveform_keys, filter_for=None):
    # plt.rcParams.update({'font.size': 22})
    y = df1[cluster_prop].dropna()
    if filter_for is not None:
        for key, val in filter_for.items():
            y = y[df1[key]==val]
    fig = plt.subplots(1,4,figsize=(12,6))
    count = 1
    for key in waveform_keys:
        plt.subplot(1,4,count)
        plt.title(key)
        for labelnum in range(5):
            label = ['biphasic','negative','early','late','unresponsive'][labelnum]
            data = list(y[df1[key+'_cluster_type']==label].dropna())
            box = plt.boxplot(data, positions=[labelnum], widths=0.7)
            plt.plot(np.ones(len(data))*labelnum, data, '.')
        plt.xticks(range(5), ['biphasic','negative','early','late','unresponsive'])
        plt.ylabel(cluster_prop)
        count += 1
    return fig

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return array[idx]

def var_around_saccade(df1, movement):
    sessions = [i for i in df1['session'].unique() if type(i) != float]
    n_sessions = len(df1['session'].unique())
    trange = np.arange(-1,1.1,0.025)
    fig = plt.subplots(n_sessions,4, figsize=(20,30))
    count = 1
    for session_num in tqdm(range(len(sessions))):
        session = sessions[session_num]
        # get 0th index of units in this session (all units have identical info for these columns)
        row = df1[df1['session']==session].iloc[0]
        
        if type(row['fm1_eyeT']) != float and type(row['fm1_dEye']) != float and type(row['fm1_dHead']) != float:
            
            eyeT = row['fm1_eyeT'].values
            dEye = row['fm1_dEye']
            dhead = row['fm1_dHead']
            dgz = dEye + dhead(eyeT[0:-1])
            
            if movement=='eye_gaze_shifting':
                sthresh = 5
                rightsacc = eyeT[(np.append(dEye,0)>sthresh) & (np.append(dgz,0)>sthresh)]
                leftsacc = eyeT[(np.append(dEye,0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
            elif movement=='eye_comp':
                sthresh = 3
                rightsacc = eyeT[(np.append(dEye,0)>sthresh) & (np.append(dgz,0)<1)]
                leftsacc = eyeT[(np.append(dEye,0)<-sthresh) & (np.append(dgz,0)>-1)]
            elif movement=='head_gaze_shifting':
                sthresh = 3
                rightsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)>sthresh) & (np.append(dgz,0)>sthresh)]
                leftsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)<-sthresh) & (np.append(dgz,0)<-sthresh)]
            elif movement=='head_comp':
                sthresh = 3
                rightsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)>sthresh) & (np.append(dgz,0)<1)]
                leftsacc = eyeT[(np.append(dhead(eyeT[0:-1]),0)<-sthresh) & (np.append(dgz,0)>-1)]

            deye_mov_right = np.zeros([len(rightsacc), len(trange)]); deye_mov_left = np.zeros([len(leftsacc), len(trange)])
            dgz_mov_right = np.zeros([len(rightsacc), len(trange)]); dgz_mov_left = np.zeros([len(leftsacc), len(trange)])
            dhead_mov_right = np.zeros([len(rightsacc), len(trange)]); dhead_mov_left = np.zeros([len(leftsacc), len(trange)])
            
            dhead = dhead(eyeT[0:-1])

            for sind in range(len(rightsacc)):
                s = rightsacc[sind]
                mov_ind = np.where([eyeT==find_nearest(eyeT, s)])[1]
                trange_inds = list(mov_ind + np.arange(-42,42))
                if np.max(trange_inds) < len(dEye):
                    deye_mov_right[sind,:] = dEye[np.array(trange_inds)]
                if np.max(trange_inds) < len(dgz):
                    dgz_mov_right[sind,:] = dgz[np.array(trange_inds)]
                if np.max(trange_inds) < len(dhead):
                    dhead_mov_right[sind,:] = dhead[np.array(trange_inds)]
            for sind in range(len(leftsacc)):
                s = leftsacc[sind]
                mov_ind = np.where([eyeT==find_nearest(eyeT, s)])[1]
                trange_inds = list(mov_ind + np.arange(-42,42))
                if np.max(trange_inds) < len(dEye):
                    deye_mov_left[sind,:] = dEye[np.array(trange_inds)]
                if np.max(trange_inds) < len(dgz):
                    dgz_mov_left[sind,:] = dgz[np.array(trange_inds)]
                if np.max(trange_inds) < len(dhead):
                    dhead_mov_left[sind,:] = dhead[np.array(trange_inds)]
            
            plt.subplot(n_sessions,4, count)
            count += 1
            plt.plot(trange*2, np.nanmean(deye_mov_right,0), color='#1f77b4')
            plt.plot(trange*2, np.nanmean(deye_mov_left,0), color='r')
            plt.title(session + movement)
            plt.ylabel('deye')
            plt.subplot(n_sessions,4, count)
            count += 1
            plt.plot(trange*2, np.nancumsum(np.nanmean(deye_mov_right,0)), color='#1f77b4')
            plt.plot(trange*2, np.nancumsum(np.nanmean(deye_mov_left,0)), color='r')
            plt.ylabel('cumulative deye')
            plt.subplot(n_sessions,4, count)
            count += 1
            plt.plot(trange*2, np.nanmean(dhead_mov_right,0), color='#1f77b4')
            plt.plot(trange*2, np.nanmean(dhead_mov_left,0), color='r')
            plt.ylabel('dhead')
            plt.subplot(n_sessions,4, count)
            count += 1
            plt.plot(trange*2, np.nancumsum(np.nanmean(dhead_mov_right,0)), color='#1f77b4')
            plt.plot(trange*2, np.nancumsum(np.nanmean(dhead_mov_left,0)), color='r')
            plt.ylabel('cumulative dhead')

    plt.tight_layout()
    return fig

def len_without_nan(x_list):
    count = []
    for x in x_list:
        count.append(len(x[~np.isnan(x)]))
    return np.sum(count)

def gratings_polar_plot(df, attr, labels, num_bins=8):
    fig = plt.subplots(3,int(np.ceil((len(labels)+1)/3)),figsize=(15,15))
    ax = plt.subplot(3,int(np.ceil((len(labels)+1)/3)),1,projection='polar')
    s = df[attr]
    ax.bar(np.linspace(0,(2*np.pi)-np.deg2rad(45),8), np.histogram(s, bins=8)[0], width=(2*np.pi)/8, bottom=0, alpha=0.5, color='tab:gray')
    plt.title('all')
    for count, label in enumerate(labels):
        s = df[attr][df['movement_psth_type']==label]
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        ax = plt.subplot(3,int(np.ceil((len(labels)+1)/3)),count+2,projection='polar')
        ax.bar(np.linspace(0,(2*np.pi)-np.deg2rad(45),8), np.histogram(s, bins=8)[0], width=(2*np.pi)/8, bottom=0, alpha=0.5, color=tempcolor)
        plt.title(label+' (cells='+str(len(s))+')')
    plt.tight_layout()
    return fig

def make_population_summary(df1, savepath):
    """
    summarize an entire population of ephys units
    INPUTS
        df1: updated dataframe from unit summary
        savepath: where to save the pdf and the updated dataframe
    OUTPUTS
        None
    """
    # plt.rcParams.update({'font.size': 22})
    print('opening pdf')
    pdf = PdfPages(os.path.join(savepath, 'population_summary_'+datetime.today().strftime('%m%d%y')+'.pdf'))
    
    ### waveform fig
    print('labeling by waveform')
    plt.subplots(2,4, figsize=(20,10))
    plt.subplot(2,4,1)
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

    print('kmeans')
    km_labels = KMeans(n_clusters=2).fit(list(df1['norm_waveform'][df1['waveform_peak'] < 0].to_numpy())).labels_
    # make excitatory (fast spiking) always group 0
    # excitatory should always have a smaller mean waveform trough
    # if it's larger, flip the kmeans labels
    if np.mean(df1['waveform_trough_width'][df1['waveform_peak']<0][km_labels==0]) > np.mean(df1['waveform_trough_width'][df1['waveform_peak']<0][km_labels==1]):
        km_labels = [0 if i==1 else 1 for i in km_labels]
    count = 0
    for ind, row in df1.iterrows():
        if row['waveform_peak'] < 0:
            df1.at[ind, 'waveform_km_label'] = km_labels[count]
            count = count+1
    plt.subplot(2,4,2)
    for ind, row in df1.iterrows():
        if row['waveform_km_label']==0:
            plt.plot(row['norm_waveform'], 'g')
        elif row['waveform_km_label']==1:
            plt.plot(row['norm_waveform'], 'b')

    plt.subplot(2,4,3)
    plt.plot(df1['waveform_trough_width'][df1['waveform_peak'] < 0][df1['waveform_km_label']==0], df1['AHP'][df1['waveform_peak'] < 0][df1['waveform_km_label']==0], 'g.')
    plt.plot(df1['waveform_trough_width'][df1['waveform_peak'] < 0][df1['waveform_km_label']==1], df1['AHP'][df1['waveform_peak'] < 0][df1['waveform_km_label']==1], 'b.')
    plt.legend(['kmeans=0', 'kmeans=1'])
    plt.ylabel('AHP'); plt.xlabel('waveform trough width')
    plt.title('seperation by kmeans clustering')
    bluepatch = mpatches.Patch(color='g', label='inhibitory')
    greenpatch = mpatches.Patch(color='b', label='excitatory')
    plt.legend(handles=[bluepatch, greenpatch])

    plt.subplot(2,4,4)
    plt.hist(df1['waveform_trough_width'],bins=range(3,35))
    plt.xlabel('trough width')

    plt.subplot(2,4,5)
    plt.xlabel('AHP')
    plt.hist(df1['AHP'],bins=60); plt.xlim([-1,1])

    plt.subplot(2,4,6)
    plt.title('seperation by properties')
    plt.plot(df1['waveform_trough_width'][df1['waveform_peak'] < 0][df1['waveform_km_label']==0], df1['AHP'][df1['waveform_peak'] < 0][df1['waveform_km_label']==0], 'g.')
    plt.plot(df1['waveform_trough_width'][df1['waveform_peak'] < 0][df1['waveform_km_label']==1], df1['AHP'][df1['waveform_peak'] < 0][df1['waveform_km_label']==1], 'b.')
    plt.ylabel('AHP'); plt.xlabel('waveform trough width')

    print('depth plot')
    plt.subplot(2,4,7)
    plt.hist(df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==1],color='b',bins=np.arange(-600,600,25),alpha=0.3,orientation='horizontal')
    plt.hist(df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==0],color='g',bins=np.arange(-600,600,25),alpha=0.3,orientation='horizontal')
    plt.xlabel('channels above or below center of layer 5'); plt.gca().invert_yaxis()
    plt.plot([0,14],[0,0],'k')

    plt.subplot(2,4,8)
    plt.axis('off')

    plt.tight_layout(); pdf.savefig(); plt.close()

    print('panels of osi vs variable')
    ### osi figure
    for ind, row in df1.iterrows():
        tuning = row['hf1_wn_crf_tuning']
        if type(tuning) == np.ndarray or type(tuning) == list:
            tuning = [x for x in tuning if x != None]
            # thresh out units which have a small response to contrast, even if the modulation index is large
            df1.at[ind, 'responsive_to_contrast'] = np.abs(tuning[-2] - tuning[1]) > 1
        else:
            df1.at[ind, 'responsive_to_contrast'] = False
    depth_range = [np.max(df1['hf1_wn_depth_from_layer5'][df1['responsive_to_contrast']==True])+50, np.min(df1['hf1_wn_depth_from_layer5'][df1['responsive_to_contrast']==True])+50]

    ### contrast
    crfs0 = np.zeros([len(df1['hf1_wn_crf_tuning'][df1['waveform_km_label']==0]),11])
    crfs1 = np.zeros([len(df1['hf1_wn_crf_tuning'][df1['waveform_km_label']==1]),11])
    for i, x in df1['hf1_wn_crf_tuning'].iteritems():
        if type(x) == str:
            x = np.array([np.nan if i=='nan' else i for i in list(x.split(' ')[1:-2])])
        if type(x) != float:
            df1.at[i, 'hf1_wn_spont_rate'] = x[0]
            df1.at[i, 'hf1_wn_max_contrast_rate'] = x[-1]
            df1.at[i, 'hf1_wn_evoked_rate'] = x[-1] - x[0]
        else:
            df1.at[i, 'hf1_wn_spont_rate'] = np.nan
            df1.at[i, 'hf1_wn_max_contrast_rate'] = np.nan
            df1.at[i, 'hf1_wn_evoked_rate'] = np.nan
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

    # plt.figure()
    # for i in range(np.size(crfs0, 0)):
    #     plt.plot(crfs0[i,:], alpha=0.5)
    # plt.tight_layout(); pdf.savefig(); plt.close()
    # plt.figure()
    # for i in range(np.size(crfs1, 0)):
    #     plt.plot(crfs1[i,:], alpha=0.5)
    # plt.tight_layout(); pdf.savefig(); plt.close()

    plt.subplots(2,3, figsize=(15,10))
    n = 1

    plt.subplot(2,3,n)
    plt.bar(['responsive', 'not responsive'], height=[len(df1[df1['responsive_to_contrast']==True])/len(df1), len(df1[df1['responsive_to_contrast']==False])/len(df1)])
    plt.title('fraction responsive to contrast'); plt.ylim([0,1])

    n += 1
    plt.subplot(2,3,n)
    plt.ylabel('depth relative to layer 5'); plt.xlabel('contrast spont rate (sp/sec)')
    plt.plot(df1['hf1_wn_spont_rate'][df1['waveform_km_label']==0], df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==0], 'g.', markersize=1)
    plt.plot(df1['hf1_wn_spont_rate'][df1['waveform_km_label']==1], df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==1], 'b.', markersize=1); plt.ylim(depth_range)
    stat2use = 'median'
    force_range = np.arange(-650,750,100)
    for count in range(2):
        x = df1['hf1_wn_spont_rate'][df1['waveform_km_label']==count]
        y = df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==count]
        c = ['g','b'][count]
        bin_means, bin_edges, bin_number = stats.binned_statistic(y[~np.isnan(x) & ~np.isnan(y)], x[~np.isnan(x) & ~np.isnan(y)], statistic=stat2use, bins=force_range)
        bin_std, _, _ = stats.binned_statistic(y[~np.isnan(x) & ~np.isnan(y)], x[~np.isnan(x) & ~np.isnan(y)], statistic='std', bins=force_range)
        hist, _ = np.histogram(y[~np.isnan(x) & ~np.isnan(y)], bins=force_range)
        tuning_err = bin_std / np.sqrt(hist)
        plt.plot(bin_means, bin_edges[:-1], c+'-')
        plt.fill_betweenx(bin_edges[:-1], bin_means-tuning_err, bin_means+tuning_err, color=c, alpha=0.3)
    # plt.xlim([-5,30])
    plt.title('pts in data='+str(len_without_nan([crfs0[:,0], crfs1[:,0]])))
    n += 1
    plt.subplot(2,3,n)
    plt.ylabel('depth relative to layer 5'); plt.xlabel('max contrast rate (sp/sec)')
    plt.plot(df1['hf1_wn_max_contrast_rate'][df1['waveform_km_label']==0], df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==0], 'g.', markersize=1)
    plt.plot(df1['hf1_wn_max_contrast_rate'][df1['waveform_km_label']==1], df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==1], 'b.', markersize=1); plt.ylim(depth_range)
    stat2use = 'median'
    for count in range(2):
        x = df1['hf1_wn_max_contrast_rate'][df1['waveform_km_label']==count]
        y = df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==count]
        c = ['g','b'][count]
        bin_means, bin_edges, bin_number = stats.binned_statistic(y[~np.isnan(x) & ~np.isnan(y)], x[~np.isnan(x) & ~np.isnan(y)], statistic=stat2use, bins=force_range)
        bin_std, _, _ = stats.binned_statistic(y[~np.isnan(x) & ~np.isnan(y)], x[~np.isnan(x) & ~np.isnan(y)], statistic='std', bins=force_range)
        hist, _ = np.histogram(y[~np.isnan(x) & ~np.isnan(y)], bins=force_range)
        tuning_err = bin_std / np.sqrt(hist)
        plt.plot(bin_means, bin_edges[:-1], c+'-')
        plt.fill_betweenx(bin_edges[:-1], bin_means-tuning_err, bin_means+tuning_err, color=c, alpha=0.3)
    # plt.xlim([-5,50])
    plt.title('pts in data='+str(len_without_nan([crfs0[:,-1], crfs1[:,-1]])))
    n += 1
    plt.subplot(2,3,n)
    plt.ylabel('depth relative to layer 5'); plt.xlabel('contrast evoked rate (sp/sec)')
    plt.plot(df1['hf1_wn_evoked_rate'][df1['waveform_km_label']==0], df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==0], 'g.', markersize=1)
    plt.plot(df1['hf1_wn_evoked_rate'][df1['waveform_km_label']==1], df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==1], 'b.', markersize=1); plt.ylim(depth_range)
    stat2use = 'median'
    for count in range(2):
        x = df1['hf1_wn_evoked_rate'][df1['waveform_km_label']==count]
        y = df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==count]
        c = ['g','b'][count]
        bin_means, bin_edges, bin_number = stats.binned_statistic(y[~np.isnan(x) & ~np.isnan(y)], x[~np.isnan(x) & ~np.isnan(y)], statistic=stat2use, bins=force_range)
        bin_std, _, _ = stats.binned_statistic(y[~np.isnan(x) & ~np.isnan(y)], x[~np.isnan(x) & ~np.isnan(y)], statistic='std', bins=force_range)
        hist, _ = np.histogram(y[~np.isnan(x) & ~np.isnan(y)], bins=force_range)
        tuning_err = bin_std / np.sqrt(hist)
        plt.plot(bin_means, bin_edges[:-1], c+'-')
        plt.fill_betweenx(bin_edges[:-1], bin_means-tuning_err, bin_means+tuning_err, color=c, alpha=0.3)
    # plt.xlim([-10,30])
    plt.title('pts in data='+str(len_without_nan([crfs0[:,-1] - crfs0[:,0], crfs1[:,-1] - crfs1[:,0]])))
    
    n += 1
    fig = plot_var_vs_var(df1, 'hf1_wn_crf_modind', 'hf1_wn_depth_from_layer5', (2,3,n), filter_for={'responsive_to_contrast':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.ylabel('depth relative to layer 5'); plt.xlabel('wn contrast modulation index'); plt.legend(handles=[bluepatch, greenpatch])

    plt.tight_layout(); pdf.savefig(); plt.close()

    ### gratings
    for sf in ['low','mid','high']:
        df1['norm_ori_tuning_'+sf] = df1['hf3_gratings_ori_tuning'].copy().astype(object)
    for ind, row in df1.iterrows():
        try:
            orientations = np.nanmean(np.array(row['hf3_gratings_ori_tuning'], dtype=np.float),2)
            for sfnum in range(3):
                sf = ['low','mid','high'][sfnum]
                df1.at[ind,'norm_ori_tuning_'+sf] = orientations[:,sfnum] - row['hf3_gratings_drift_spont']
            mean_for_sf = np.array([np.mean(df1.at[ind,'norm_ori_tuning_low']), np.mean(df1.at[ind,'norm_ori_tuning_mid']), np.mean(df1.at[ind,'norm_ori_tuning_high'])])
            mean_for_sf[mean_for_sf<0] = 0
            df1.at[ind,'sf_pref'] = ((mean_for_sf[0]*1)+(mean_for_sf[1]*2)+(mean_for_sf[2]*3))/np.sum(mean_for_sf)
            df1.at[ind,'responsive_to_gratings'] = [True if np.max(mean_for_sf)>2 else False][0]
        except:
            for sfnum in range(3):
                sf = ['low','mid','high'][sfnum]
                df1.at[ind,'norm_ori_tuning_'+sf] = None
            df1.at[ind,'responsive_to_gratings'] = False
            df1.at[ind,'sf_pref'] = np.nan

    df1['osi_for_sf_pref'] = np.nan
    df1['dsi_for_sf_pref'] = np.nan
    for ind, row in df1.iterrows():
        if ~np.isnan(row['sf_pref']):
            best_sf_pref = int(np.round(row['sf_pref']))
            df1.at[ind, 'osi_for_sf_pref'] = row[(['hf3_gratings_osi_low','hf3_gratings_osi_mid','hf3_gratings_osi_high'][best_sf_pref-1])]
            df1.at[ind, 'dsi_for_sf_pref'] = row[(['hf3_gratings_dsi_low','hf3_gratings_dsi_mid','hf3_gratings_dsi_high'][best_sf_pref-1])]

    for ind, row in df1.iterrows():
        try:
            mean_for_sf = np.array([np.mean(df1.at[ind,'norm_ori_tuning_low']), np.mean(df1.at[ind,'norm_ori_tuning_mid']), np.mean(df1.at[ind,'norm_ori_tuning_high'])])
            mean_for_sf[mean_for_sf<0] = 0
            df1.at[ind, 'hf3_gratings_evoked_rate'] = np.max(mean_for_sf) - row['hf3_gratings_drift_spont']
        except:
            pass

    for ind, row in df1.iterrows():
        if type(row['hf3_gratings_ori_tuning_tf']) == str:
            df1.at[ind, 'hf3_gratings_ori_tuning_tf'] = np.array(ast.literal_eval(row['hf3_gratings_ori_tuning_tf'].replace('   ',' ').replace('  ',' ').replace('[ ','[').replace(' ]',']').replace(' ', ',').replace(',,',',')))
        if type(row['hf3_gratings_ori_tuning']) == str:
            df1.at[ind, 'hf3_gratings_ori_tuning'] = np.array(ast.literal_eval(row['hf3_gratings_ori_tuning'].replace('   ',' ').replace('  ',' ').replace('[ ','[').replace(' ]',']').replace(' ', ',').replace(',,',',')))
    for ind, row in df1.iterrows():
        if type(row['hf3_gratings_ori_tuning_tf']) != float:
            tuning = np.nanmean(row['hf3_gratings_ori_tuning'],1)
            tuning = tuning - row['hf3_gratings_drift_spont']
            tuning[tuning < 0] = 0
            mean_for_tf = np.array([np.mean(tuning[:,0]), np.mean(tuning[:,1])])
            tf_pref = ((mean_for_tf[0]*1)+(mean_for_tf[1]*2))/np.sum(mean_for_tf)
            df1.at[ind, 'tf_pref'] = tf_pref

    plt.subplots(3,11, figsize=(55,15))
    n = 1
    plt.subplot(3,11,n)
    plt.bar(['responsive', 'not responsive'], height=[len(df1[df1['responsive_to_gratings']==True])/len(df1), len(df1[df1['responsive_to_gratings']==False])/len(df1)])
    plt.title('fraction responsive to gratings'); plt.ylim([0,1])

    n += 1
    fig = plot_var_vs_var(df1, 'hf3_gratings_drift_spont', 'hf1_wn_depth_from_layer5', (3,11,n), force_range=np.arange(-650,750,100), along_y=True, use_median=True)
    plt.ylabel('depth relative to layer 5'); plt.xlabel('grat spont rate'); plt.gca().invert_yaxis()

    n += 1
    fig = plot_var_vs_var(df1, 'hf3_gratings_evoked_rate', 'hf1_wn_depth_from_layer5', (3,11,n), force_range=np.arange(-650,750,100), along_y=True, use_median=True)
    plt.ylabel('depth relative to layer 5'); plt.xlabel('grat evoked rate'); plt.gca().invert_yaxis()

    n += 1
    plt.subplot(3,11,n)
    plt.hist(df1['sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], color='g', alpha=0.3, bins=np.arange(1,3.25,0.25))
    plt.hist(df1['sf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], color='b', alpha=0.3, bins=np.arange(1,3.25,0.25))
    plt.xlabel('prefered spatial frequency'); plt.ylabel('unit count')

    n += 1
    plt.subplot(3,11,n)
    plt.hist(df1['tf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==0], color='g', alpha=0.3, bins=np.arange(1,2.25,0.25))
    plt.hist(df1['tf_pref'][df1['responsive_to_gratings']==True][df1['waveform_km_label']==1], color='b', alpha=0.3, bins=np.arange(1,2.25,0.25))
    plt.xlabel('prefered temporal frequency'); plt.ylabel('unit count')

    n += 1
    fig = plot_var_vs_var(df1, 'osi_for_sf_pref', 'hf1_wn_depth_from_layer5', (3,11,n), filter_for={'responsive_to_gratings':True}, force_range=np.arange(-650,750,100), along_y=True, use_median=True, abs=True)
    plt.xlabel('orientation selectivity index for prefered sf'); plt.ylabel('depth relative to layer 5')
    plt.gca().invert_yaxis()

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gz_modind_pos', 'osi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('orientation selectivity index for prefered sf'); plt.xlabel('gyro z positive-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gz_modind_neg', 'osi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('orientation selectivity index for prefered sf'); plt.xlabel('gyro z negative-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gx_modind_pos', 'osi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('orientation selectivity index for prefered sf'); plt.xlabel('gyro x positive-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gx_modind_neg', 'osi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('orientation selectivity index for prefered sf'); plt.xlabel('gyro x negative-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gy_modind_pos', 'osi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('orientation selectivity index for prefered sf'); plt.xlabel('gyro y positive-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gy_modind_neg', 'osi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('orientation selectivity index for prefered sf'); plt.xlabel('gyro y negative-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_roll_modind_pos', 'osi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('orientation selectivity index for prefered sf'); plt.xlabel('head roll positive-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_roll_modind_neg', 'osi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('orientation selectivity index for prefered sf'); plt.xlabel('head roll negative-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_pitch_modind_pos', 'osi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('orientation selectivity index for prefered sf'); plt.xlabel('head pitch positive-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_pitch_modind_neg', 'osi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('orientation selectivity index for prefered sf'); plt.xlabel('head pitch negative-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gz_modind_pos', 'dsi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('direction selectivity index for prefered sf'); plt.xlabel('gyro z positive-direction modulation index'); plt.legend(handles=[bluepatch, greenpatch])

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gz_modind_neg', 'dsi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('direction selectivity index for prefered sf'); plt.xlabel('gyro z negative-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gx_modind_pos', 'dsi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('direction selectivity index for prefered sf'); plt.xlabel('gyro x positive-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gx_modind_neg', 'dsi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('direction selectivity index for prefered sf'); plt.xlabel('gyro x negative-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gy_modind_pos', 'dsi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('direction selectivity index for prefered sf'); plt.xlabel('gyro y positive-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_gy_modind_neg', 'dsi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('direction selectivity index for prefered sf'); plt.xlabel('gyro y negative-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_roll_modind_pos', 'dsi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('direction selectivity index for prefered sf'); plt.xlabel('head roll positive-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_roll_modind_neg', 'dsi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('direction selectivity index for prefered sf'); plt.xlabel('head roll negative-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_pitch_modind_pos', 'dsi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('direction selectivity index for prefered sf'); plt.xlabel('head pitch positive-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_pitch_modind_neg', 'dsi_for_sf_pref', (3,11,n), filter_for={'responsive_to_gratings':True}, abs=True)
    plt.ylabel('direction selectivity index for prefered sf'); plt.xlabel('head pitch negative-direction modulation index')

    n += 1
    fig = plot_var_vs_var(df1, 'dsi_for_sf_pref', 'hf1_wn_depth_from_layer5', (3,11,n), filter_for={'responsive_to_gratings':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.xlabel('direction selectivity index for prefered sf'); plt.ylabel('depth relative to layer 5')
    plt.gca().invert_yaxis()

    n += 1
    fig = plot_var_vs_var(df1, 'sf_pref', 'hf1_wn_depth_from_layer5', (3,11,n), filter_for={'responsive_to_gratings':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.xlabel('prefered spatial frequency'); plt.ylabel('depth relative to layer 5')
    plt.gca().invert_yaxis()

    n += 1
    nrange = range(n,33)
    for i in nrange:
        plt.subplot(3,11,i)
        plt.axis('off')

    plt.tight_layout(); pdf.savefig(); plt.close()
    
    print('firing rate by stim')
    plt.subplots(2,2,figsize=(24,10))
    ax = plt.subplot(2,2,1)
    for ind, row in df1.iterrows():
        if type(row['fm1_spikeT']) == str:
            df1.at[ind,'fm1_spikeT'] = ast.literal_eval(row['fm1_spikeT'].replace('   ',' ').replace('  ',' ').replace('[ ','[').replace(' ]',']').replace(' ', ',').replace(',,',','))
        if type(row['hf3_gratings_spikeT']) == str:
            df1.at[ind,'hf3_gratings_spikeT'] = ast.literal_eval(row['hf3_gratings_spikeT'].replace('   ',' ').replace('  ',' ').replace('[ ','[').replace(' ]',']').replace(' ', ',').replace(',,',','))
        if type(row['fm_dark_spikeT']) == str:
            df1.at[ind,'fm_dark_spikeT'] = ast.literal_eval(row['fm_dark_spikeT'].replace('   ',' ').replace('  ',' ').replace('[ ','[').replace(' ]',']').replace(' ', ',').replace(',,',','))
        if type(row['hf1_wn_spikeT']) == str:
            df1.at[ind,'hf1_wn_spikeT'] = ast.literal_eval(row['hf1_wn_spikeT'].replace('   ',' ').replace('  ',' ').replace('[ ','[').replace(' ]',']').replace(' ', ',').replace(',,',','))

    for ind, row in df1.iterrows():
        if type(row['fm1_spikeT']) != float:
            df1.at[ind,'fm1_rec_rate'] = len(row['fm1_spikeT']) / (row['fm1_spikeT'][-1] - row['fm1_spikeT'][0])
        if type(row['fm_dark_spikeT']) != float:
            df1.at[ind,'fm_dark_rec_rate'] = len(row['fm_dark_spikeT']) / (row['fm_dark_spikeT'][-1] - row['fm_dark_spikeT'][0])
        if type(row['hf3_gratings_spikeT']) != float:
            df1.at[ind,'hf3_gratings_rec_rate'] = len(row['hf3_gratings_spikeT']) / (row['hf3_gratings_spikeT'][-1] - row['hf3_gratings_spikeT'][0])
        if type(row['hf1_wn_spikeT']) != float:
            df1.at[ind,'hf1_wn_rec_rate'] = len(row['hf1_wn_spikeT']) / (row['hf1_wn_spikeT'][-1] - row['hf1_wn_spikeT'][0])
    
    labels = ['grat', 'wn', 'fm light', 'fm dark']
    x = np.arange(len(labels))
    width = 0.35; a = 1
    exc_rates = np.array([df1['hf3_gratings_rec_rate'][df1['waveform_km_label']==1], df1['hf1_wn_rec_rate'][df1['waveform_km_label']==1], df1['fm1_rec_rate'][df1['waveform_km_label']==1], df1['fm_dark_rec_rate'][df1['waveform_km_label']==1]])
    inh_rates = np.array([df1['hf3_gratings_rec_rate'][df1['waveform_km_label']==0], df1['hf1_wn_rec_rate'][df1['waveform_km_label']==0], df1['fm1_rec_rate'][df1['waveform_km_label']==0], df1['fm_dark_rec_rate'][df1['waveform_km_label']==0]])
    plt.bar(x - width/2, np.nanmedian(exc_rates,a), yerr=np.nanstd(exc_rates,a)/np.sqrt(np.size(exc_rates,a)), color='b', width=width, label='exc')
    plt.bar(x + width/2, np.nanmedian(inh_rates,a), yerr=np.nanstd(inh_rates,a)/np.sqrt(np.size(inh_rates,a)), color='g', width=width, label='inh')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.legend()
    plt.ylabel('sp/sec')
    plt.title('median spike rate across full recording')

    active_time_by_session = dict()
    sessions = [x for x in df1['session'].unique() if str(x) != 'nan']
    for session in sessions:
        session_data = df1[df1['session']==session]
        # find active times
        model_dt = 0.025
        if type(session_data['fm1_eyeT'].iloc[0]) != float:
            # light setup
            fm_light_eyeT = session_data['fm1_eyeT'].iloc[0].values
            fm_light_gz = session_data['fm1_gz'].iloc[0]
            fm_light_accT = session_data['fm1_accT'].iloc[0]
            light_model_t = np.arange(0,np.nanmax(fm_light_eyeT),model_dt)
            light_model_gz = interp1d(fm_light_accT,(fm_light_gz-np.mean(fm_light_gz))*7.5,bounds_error=False)(light_model_t)
            light_model_active = np.convolve(np.abs(light_model_gz),np.ones(np.int(1/model_dt)),'same')
            light_active = light_model_active>40

            n_units = len(session_data)
            light_model_nsp = np.zeros((n_units, len(light_model_t)))
            bins = np.append(light_model_t, light_model_t[-1]+model_dt)
            i = 0
            for ind, row in session_data.iterrows():
                light_model_nsp[i,:], bins = np.histogram(row['fm1_spikeT'], bins)
                unit_active_spikes = light_model_nsp[i, light_active]
                unit_stationary_spikes = light_model_nsp[i, ~light_active]
                df1.at[ind,'fm1_active_rec_rate'] = np.sum(unit_active_spikes) / (len(unit_active_spikes)*model_dt)
                df1.at[ind,'fm1_stationary_rec_rate'] = np.sum(unit_stationary_spikes) / (len(unit_stationary_spikes)*model_dt)
                i += 1

            active_time_by_session.setdefault('light', {})[session] = np.sum(light_active) / len(light_active)

        if type(session_data['fm_dark_eyeT'].iloc[0]) != float:
            del unit_active_spikes, unit_stationary_spikes
            
            # dark setup
            fm_dark_eyeT = session_data['fm_dark_eyeT'].iloc[0].values
            fm_dark_gz = session_data['fm_dark_gz'].iloc[0]
            fm_dark_accT = session_data['fm_dark_accT'].iloc[0]
            dark_model_t = np.arange(0,np.nanmax(fm_dark_eyeT),model_dt)
            dark_model_gz = interp1d(fm_dark_accT,(fm_dark_gz-np.mean(fm_dark_gz))*7.5,bounds_error=False)(dark_model_t)
            dark_model_active = np.convolve(np.abs(dark_model_gz),np.ones(np.int(1/model_dt)),'same')
            dark_active = dark_model_active>40
            
            
            n_units = len(session_data)
            dark_model_nsp = np.zeros((n_units, len(dark_model_t)))
            bins = np.append(dark_model_t, dark_model_t[-1]+model_dt)
            i = 0
            for ind, row in session_data.iterrows():
                dark_model_nsp[i,:], bins = np.histogram(row['fm_dark_spikeT'], bins)
                unit_active_spikes = dark_model_nsp[i, dark_active]
                unit_stationary_spikes = dark_model_nsp[i, ~dark_active]
                df1.at[ind,'fm_dark_active_rec_rate'] = np.sum(unit_active_spikes) / (len(unit_active_spikes)*model_dt)
                df1.at[ind,'fm_dark_stationary_rec_rate'] = np.sum(unit_stationary_spikes) / (len(unit_stationary_spikes)*model_dt)
                i += 1

            active_time_by_session.setdefault('dark', {})[session] = np.sum(dark_active) / len(dark_active)


    ax = plt.subplot(2,2,2)
    labels = ['active light','stationary light','active dark','stationary dark']
    x = np.arange(len(labels))
    width = 0.35
    exc_rates = np.array([df1['fm1_active_rec_rate'][df1['waveform_km_label']==1], df1['fm1_stationary_rec_rate'][df1['waveform_km_label']==1], df1['fm_dark_active_rec_rate'][df1['waveform_km_label']==1], df1['fm_dark_stationary_rec_rate'][df1['waveform_km_label']==1]])
    inh_rates = np.array([df1['fm1_active_rec_rate'][df1['waveform_km_label']==0], df1['fm1_stationary_rec_rate'][df1['waveform_km_label']==0], df1['fm_dark_active_rec_rate'][df1['waveform_km_label']==0], df1['fm_dark_stationary_rec_rate'][df1['waveform_km_label']==0]])
    plt.bar(x - width/2, np.nanmedian(exc_rates,1), yerr=np.nanstd(exc_rates,1)/np.sqrt(np.size(exc_rates,1)), color='b', width=width, label='exc')
    plt.bar(x + width/2, np.nanmedian(inh_rates,1), yerr=np.nanstd(inh_rates,1)/np.sqrt(np.size(inh_rates,1)), color='g', width=width, label='inh')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.legend()
    plt.ylabel('sp/sec')
    plt.title('median spike rate during freely moving active and stationary periods')

    ax = plt.subplot(2,2,3)
    labels = ['grat spont', 'grat stim', 'wn spont', 'wn max contrast', 'fm light stationary', 'fm light active', 'fm dark stationary', 'fm dark active']
    x = np.arange(len(labels))
    width = 0.35; a = 0
    exc_rates = pd.concat([df1['hf3_gratings_drift_spont'][df1['waveform_km_label']==1].astype(float), df1['hf3_gratings_evoked_rate'][df1['waveform_km_label']==1]+df1['hf3_gratings_drift_spont'][df1['waveform_km_label']==1].astype(float),
                        df1['hf1_wn_spont_rate'][df1['waveform_km_label']==1], df1['hf1_wn_evoked_rate'][df1['waveform_km_label']==1]+df1['hf1_wn_spont_rate'][df1['waveform_km_label']==1],
                        df1['fm1_stationary_rec_rate'][df1['waveform_km_label']==1], df1['fm1_active_rec_rate'][df1['waveform_km_label']==1],
                        df1['fm_dark_stationary_rec_rate'][df1['waveform_km_label']==1], df1['fm_dark_active_rec_rate'][df1['waveform_km_label']==1]], axis=1)
    inh_rates = pd.concat([df1['hf3_gratings_drift_spont'][df1['waveform_km_label']==0].astype(float), df1['hf3_gratings_evoked_rate'][df1['waveform_km_label']==0]+df1['hf3_gratings_drift_spont'][df1['waveform_km_label']==0].astype(float),
                        df1['hf1_wn_spont_rate'][df1['waveform_km_label']==0], df1['hf1_wn_evoked_rate'][df1['waveform_km_label']==0]+df1['hf1_wn_spont_rate'][df1['waveform_km_label']==0],
                        df1['fm1_stationary_rec_rate'][df1['waveform_km_label']==0], df1['fm1_active_rec_rate'][df1['waveform_km_label']==0],
                        df1['fm_dark_stationary_rec_rate'][df1['waveform_km_label']==0], df1['fm_dark_active_rec_rate'][df1['waveform_km_label']==0]], axis=1)
    plt.bar(x - width/2, np.nanmedian(exc_rates,a), yerr=np.nanstd(exc_rates,a)/np.sqrt(np.size(exc_rates,a)), color='b', width=width, label='exc')
    plt.bar(x + width/2, np.nanmedian(inh_rates,a), yerr=np.nanstd(inh_rates,a)/np.sqrt(np.size(inh_rates,a)), color='g', width=width, label='inh')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.ylim([0,20])
    plt.title('median spike rate in active/stationary or spont/evoked')
    plt.legend()
    plt.ylabel('sp/sec')

    plt.tight_layout(); pdf.savefig(); plt.close()

    light = np.array([val for key,val in active_time_by_session['light'].items()])
    light_err = np.std(light) / np.sqrt(len(light))
    dark = np.array([val for key,val in active_time_by_session['dark'].items()])
    dark_err = np.std(dark) / np.sqrt(len(dark))
    fig, ax = plt.subplots(1,1,figsize=(3,5))
    plt.bar(0, np.mean(light), yerr=light_err, width=0.5, color='yellow')
    plt.plot(np.zeros(len(light)), light, 'o', color='tab:gray')
    plt.bar(1, np.mean(dark), yerr=dark_err, width=0.5, color='cadetblue')
    plt.plot(np.ones(len(dark)), dark, 'o', color='tab:gray')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['light','dark'])
    plt.ylim([0,1])
    plt.ylabel('fraction of time spent active')
    plt.tight_layout(); pdf.savefig(); plt.close()

    for ind, row in df1.iterrows():
        if type(row['fm1_spikeT']) != float:
            df1.at[ind, 'fires_2sp_sec'] = (True if (len(row['fm1_spikeT'])/np.nanmax(row['fm1_eyeT']))>2 else False)
        else:
            df1.at[ind, 'fires_2sp_sec'] = False

    ### movement signals along depth
    plt.subplots(2,4, figsize=(20,10))
    n = 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_roll_modind_pos', 'hf1_wn_depth_from_layer5', (2,4,n), filter_for={'fires_2sp_sec':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.xlabel('spike rate modulation with positive head roll'); plt.ylabel('depth relative to layer 5')
    plt.gca().invert_yaxis()
    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_roll_modind_neg', 'hf1_wn_depth_from_layer5', (2,4,n), filter_for={'fires_2sp_sec':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.xlabel('spike rate modulation with negative head roll'); plt.ylabel('depth relative to layer 5')
    plt.gca().invert_yaxis()
    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_pitch_modind_pos', 'hf1_wn_depth_from_layer5', (2,4,n), filter_for={'fires_2sp_sec':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.xlabel('spike rate modulation with positive head pitch'); plt.ylabel('depth relative to layer 5')
    plt.gca().invert_yaxis()
    n +=1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_pitch_modind_neg', 'hf1_wn_depth_from_layer5', (2,4,n), filter_for={'fires_2sp_sec':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.xlabel('spike rate modulation with negative head pitch'); plt.ylabel('depth relative to layer 5'); plt.legend(handles=[bluepatch, greenpatch])
    plt.gca().invert_yaxis()
    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_theta_modind_pos', 'hf1_wn_depth_from_layer5', (2,4,n), filter_for={'fires_2sp_sec':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.xlabel('spike rate modulation with positive eye theta'); plt.ylabel('depth relative to layer 5')
    plt.gca().invert_yaxis()
    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_theta_modind_neg', 'hf1_wn_depth_from_layer5', (2,4,n), filter_for={'fires_2sp_sec':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.xlabel('spike rate modulation with negative eye theta'); plt.ylabel('depth relative to layer 5')
    plt.gca().invert_yaxis()
    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_phi_modind_pos', 'hf1_wn_depth_from_layer5', (2,4,n), filter_for={'fires_2sp_sec':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.xlabel('spike rate modulation with positive eye phi'); plt.ylabel('depth relative to layer 5')
    plt.gca().invert_yaxis()
    n += 1
    fig = plot_var_vs_var(df1, 'fm1_spike_rate_vs_phi_modind_neg', 'hf1_wn_depth_from_layer5', (2,4,n), filter_for={'fires_2sp_sec':True}, force_range=np.arange(-650,750,100), along_y=True, abs=True)
    plt.xlabel('spike rate modulation with negative eye phi'); plt.ylabel('depth relative to layer 5')
    plt.gca().invert_yaxis()
    plt.tight_layout(); pdf.savefig(); plt.close()

    ### waveform clustering figures
    for ind, row in df1.iterrows():
        # direction preference
        left_deflection = row['fm1_downsacc_avg_gaze_shift_dEye']
        right_deflection = row['fm1_upsacc_avg_gaze_shift_dEye']
        left_right_zscores = [(np.max(np.abs(left_deflection))-np.mean(left_deflection)) / np.std(left_deflection), (np.max(np.abs(right_deflection))-np.mean(right_deflection)) / np.std(right_deflection)]
        left_right_index = np.argmax(np.abs(left_right_zscores))
        saccade_direction_pref = ['L','R'][left_right_index]
        df1.at[ind, 'gaze_shift_direction_pref'] = saccade_direction_pref; df1.at[ind, 'gaze_shift_direction_pref_ind'] = left_right_index
        # direction preference for compensatory movements
        left_deflection = row['fm1_downsacc_avg_comp_dEye']
        right_deflection = row['fm1_upsacc_avg_comp_dEye']
        left_right_zscores = [(np.max(np.abs(left_deflection))-np.mean(left_deflection)) / np.std(left_deflection), (np.max(np.abs(right_deflection))-np.mean(right_deflection)) / np.std(right_deflection)]
        left_right_index = np.argmax(np.abs(left_right_zscores))
        saccade_direction_pref = ['L','R'][left_right_index]
        df1.at[ind, 'comp_direction_pref'] = saccade_direction_pref; df1.at[ind, 'comp_direction_pref_ind'] = left_right_index
    for ind, row in df1.iterrows():
        # more compensatory or more gaze-shifting?
        comp_deflection = [row['fm1_downsacc_avg_comp_dEye'],row['fm1_upsacc_avg_comp_dEye']][int(row['comp_direction_pref_ind'])]
        gazeshift_deflection = [row['fm1_downsacc_avg_gaze_shift_dEye'],row['fm1_upsacc_avg_gaze_shift_dEye']][int(row['gaze_shift_direction_pref_ind'])]
        comp_gazeshift_zscores = [(np.max(np.abs(comp_deflection))-np.mean(comp_deflection)) / np.std(comp_deflection), (np.max(np.abs(gazeshift_deflection))-np.mean(gazeshift_deflection)) / np.std(gazeshift_deflection)]
        comp_gazeshift_index = np.argmax(np.abs(comp_gazeshift_zscores))
        comp_gazeshift_pref = ['comp','gaze_shift'][comp_gazeshift_index]
        df1.at[ind, 'comp_gazeshift_pref'] = comp_gazeshift_pref
    df1['dark_gaze_shift_using_light_direction_pref'] = np.array(np.nan).astype(object)
    df1['dark_comp_using_light_direction_pref'] = np.array(np.nan).astype(object)
    df1['dark_gaze_shift_using_light_direction_opp'] = np.array(np.nan).astype(object)
    df1['dark_comp_using_light_direction_opp'] = np.array(np.nan).astype(object)
    for ind, row in df1.iterrows():
        deflection_at_pref_direction = [row['fm1_downsacc_avg_gaze_shift_dEye'],row['fm1_upsacc_avg_gaze_shift_dEye']][int(row['gaze_shift_direction_pref_ind'])]
        norm_deflection = (deflection_at_pref_direction-np.nanmean(deflection_at_pref_direction)) / np.nanmax(np.abs(deflection_at_pref_direction))
        df1.at[ind, 'norm_deflection_at_pref_direction'] = norm_deflection.astype(object)
        
        deflection_at_pref_direction = [row['fm1_downsacc_avg_comp_dEye'],row['fm1_upsacc_avg_comp_dEye']][int(row['gaze_shift_direction_pref_ind'])]
        norm_comp_deflection = (deflection_at_pref_direction-np.nanmean(deflection_at_pref_direction)) / np.nanmax(np.abs(deflection_at_pref_direction))
        df1.at[ind, 'norm_deflection_at_pref_direction_comp'] = norm_comp_deflection.astype(object)

        deflection_at_pref_direction = [row['fm1_downsacc_avg_gaze_shift_dEye'],row['fm1_upsacc_avg_gaze_shift_dEye']][1-int(row['gaze_shift_direction_pref_ind'])]
        norm_deflection = (deflection_at_pref_direction-np.nanmean(deflection_at_pref_direction)) / np.nanmax(np.abs(deflection_at_pref_direction))
        df1.at[ind, 'norm_deflection_at_opp_direction'] = norm_deflection.astype(object)
        
        deflection_at_pref_direction = [row['fm1_downsacc_avg_comp_dEye'],row['fm1_upsacc_avg_comp_dEye']][1-int(row['gaze_shift_direction_pref_ind'])]
        norm_comp_deflection = (deflection_at_pref_direction-np.nanmean(deflection_at_pref_direction)) / np.nanmax(np.abs(deflection_at_pref_direction))
        df1.at[ind, 'norm_deflection_at_opp_direction_comp'] = norm_comp_deflection.astype(object)
        
        dark_gaze_shift = [row['fm_dark_downsacc_avg_gaze_shift_dEye'],row['fm_dark_upsacc_avg_gaze_shift_dEye']][int(row['gaze_shift_direction_pref_ind'])]
        dark_gaze_shift_norm = ((dark_gaze_shift-np.nanmean(dark_gaze_shift)) / np.nanmax(np.abs(dark_gaze_shift)))
        dark_comp = [row['fm_dark_downsacc_avg_comp_dEye'],row['fm_dark_upsacc_avg_comp_dEye']][int(row['gaze_shift_direction_pref_ind'])]
        dark_comp_norm = ((dark_comp-np.nanmean(dark_comp)) / np.nanmax(np.abs(dark_comp)))

        dark_gaze_shift_opp = [row['fm_dark_downsacc_avg_gaze_shift_dEye'],row['fm_dark_upsacc_avg_gaze_shift_dEye']][1-int(row['gaze_shift_direction_pref_ind'])]
        dark_gaze_shift_norm_opp = ((dark_gaze_shift_opp-np.nanmean(dark_gaze_shift_opp)) / np.nanmax(np.abs(dark_gaze_shift_opp)))
        dark_comp_opp = [row['fm_dark_downsacc_avg_comp_dEye'],row['fm_dark_upsacc_avg_comp_dEye']][1-int(row['gaze_shift_direction_pref_ind'])]
        dark_comp_norm_opp = ((dark_comp_opp-np.nanmean(dark_comp_opp)) / np.nanmax(np.abs(dark_comp_opp)))
        
        if type(dark_comp_norm) != float and type(dark_gaze_shift_norm) != float and type(dark_gaze_shift_norm_opp) != float and type(dark_comp_norm_opp) != float:
            df1.at[ind, 'dark_gaze_shift_using_light_direction_pref'] = dark_gaze_shift_norm.astype(object)
            df1.at[ind, 'dark_comp_using_light_direction_pref'] = dark_comp_norm.astype(object)
            df1.at[ind, 'dark_gaze_shift_using_light_direction_opp'] = dark_gaze_shift_norm_opp.astype(object)
            df1.at[ind, 'dark_comp_using_light_direction_opp'] = dark_comp_norm_opp.astype(object)

    norm_deflection = flatten_series(df1['norm_deflection_at_pref_direction'])
    agg = AgglomerativeClustering(n_clusters=5)
    agg.fit(norm_deflection)
    print('visualizing with pca')
    reduced_data = PCA(n_components=2).fit_transform(norm_deflection)
    h = .02
    plt.figure()
    Z = agg.labels_
    plt.scatter(reduced_data[:,0],reduced_data[:,1],c=Z,s=5)
    plt.tight_layout(); pdf.savefig(); plt.close()
    df1['gauss_clust'] = Z
    print('plotting clusters')
    trange = np.arange(-1,1.1,0.025)
    plt.subplots(2,3, figsize=(15,10))
    count = 1
    mean_cluster = dict()
    for label in range(5):
        plt.subplot(2,3,count)
        plt.title(str(label)+' count='+str(len(df1['norm_deflection_at_pref_direction'][df1['gauss_clust']==label].dropna())))
        inhibitory_nested = df1['norm_deflection_at_pref_direction'][df1['gauss_clust']==label][df1['waveform_km_label']==0].ravel()
        sz1 = (np.size(inhibitory_nested, 0) if type(inhibitory_nested) != np.float else 0)
        for i in range(sz1):
            temp_sz0 = (len(inhibitory_nested[i]) if type(inhibitory_nested[i]) != np.float else 0)
            if temp_sz0 > 0:
                sz0 = temp_sz0
        if sz0 > 0 and sz1 > 0:
            inhibitory = np.zeros([sz1,sz0])
            for i in range(sz1):
                inhibitory[i,:] = inhibitory_nested[i]
            plt.plot(0.5*(trange[0:-1] + trange[1:]), inhibitory.T, 'g', alpha=0.1)
        else:
            inhibitory = np.nan
        excitatory_nested = df1['norm_deflection_at_pref_direction'][df1['gauss_clust']==label][df1['waveform_km_label']==1].ravel()
        sz1 = (np.size(excitatory_nested, 0) if type(excitatory_nested) != np.float else 0)
        for i in range(sz1):
            temp_sz0 = (len(excitatory_nested[i]) if type(excitatory_nested[i]) != np.float else 0)
            if temp_sz0 > 0:
                sz0 = temp_sz0
        if sz0 > 0 and sz1 > 0:
            excitatory = np.zeros([sz1,sz0])
            for i in range(sz1):
                excitatory[i,:] = excitatory_nested[i]
            plt.plot(0.5*(trange[0:-1] + trange[1:]), excitatory.T, 'b', alpha=0.1)
        else:
            excitatory = np.nan
        if type(inhibitory) != float or type(excitatory) != float:
            if type(inhibitory) != float and type(excitatory) != float:
                all_units = np.nanmean(np.concatenate([inhibitory, excitatory], axis=0), axis=0)
            elif type(inhibitory) != float and type(excitatory) == float:
                all_units = np.nanmean(inhibitory, axis=0)
            elif type(inhibitory) == float and type(excitatory) != float:
                all_units = np.nanmean(excitatory, axis=0)
            mean_cluster[label] = all_units
            plt.plot(0.5*(trange[0:-1] + trange[1:]), all_units, 'k', linewidth=5)
        else:
            mean_cluster[label] = np.nan
        count += 1
        plt.xlim([-0.5,0.75])
    plt.legend(handles=[bluepatch, greenpatch])
    plt.tight_layout(); pdf.savefig(); plt.close()
    print('relabeling based on peak finding')
    cluster_to_cell_type = dict()
    for cluster_num, orig_cluster in mean_cluster.items():
        cluster = flatten_series(df1['norm_deflection_at_pref_direction'][df1['gauss_clust']==cluster_num])
        cluster_mean = np.nanmean(cluster, 0)
        baseline = np.nanmean(cluster_mean[:30])
        p, t = get_peak_trough(cluster_mean[38:50], baseline)
        cluster_to_cell_type[cluster_num] = get_cluster_props(p, t)
    for ind, row in df1.iterrows():
        df1.at[ind, 'movement_psth_type_simple'] = cluster_to_cell_type[row['gauss_clust']]
    
    early = flatten_series(df1['norm_deflection_at_opp_direction_comp'][df1['movement_psth_type_simple']=='early'])
    early_cluster_km_labels = KMeans(n_clusters=2).fit(early).labels_
    early0mean = np.nanmean(early[early_cluster_km_labels==0], 0)
    early1mean = np.nanmean(early[early_cluster_km_labels==1], 0)
    comp_responsive_early_cluster = np.argmax([(np.max(np.abs(early0mean))-np.mean(early0mean)) / np.std(early0mean), (np.max(np.abs(early1mean))-np.mean(early1mean)) / np.std(early1mean)])
    plt.subplots(2,1,figsize=(6,10))
    plt.subplot(2,1,1)
    plt.plot(0.5*(trange[0:-1] + trange[1:]), early[early_cluster_km_labels==0].T, color='tab:purple', alpha=0.4)
    plt.ylim([-0.8,0.8]); plt.xlim([-0.25,0.5])
    plt.plot(0.5*(trange[0:-1] + trange[1:]), early0mean, 'k')
    plt.subplot(2,1,2)
    plt.plot(0.5*(trange[0:-1] + trange[1:]), early[early_cluster_km_labels==1].T, color='tab:purple', alpha=0.4)
    plt.ylim([-0.8,0.8]); plt.xlim([-0.25,0.5])
    plt.plot(0.5*(trange[0:-1] + trange[1:]), early1mean, 'k')
    plt.tight_layout(); pdf.savefig(); plt.close()
    early_inds = df1['norm_deflection_at_opp_direction_comp'][df1['movement_psth_type_simple']=='early'].index.values
    for i in range(np.size(early, 0)):
        real_ind = early_inds[i]
        df1.at[real_ind, 'early_comp_responsive'] = (True if early_cluster_km_labels[i]==comp_responsive_early_cluster else False)

    late = flatten_series(df1['norm_deflection_at_opp_direction_comp'][df1['movement_psth_type_simple']=='late'])
    late_cluster_km_labels = KMeans(n_clusters=2).fit(late).labels_
    late0mean = np.nanmean(late[late_cluster_km_labels==0], 0)
    late1mean = np.nanmean(late[late_cluster_km_labels==1], 0)
    comp_responsive_late_cluster = np.argmax([(np.max(np.abs(late0mean))-np.mean(late0mean)) / np.std(late0mean), (np.max(np.abs(late1mean))-np.mean(late1mean)) / np.std(late1mean)])
    trange = np.arange(-1,1.1,0.025)
    plt.subplots(2,1,figsize=(6,10))
    plt.subplot(2,1,1)
    plt.plot(0.5*(trange[0:-1] + trange[1:]), late[late_cluster_km_labels==0].T, color='tab:blue', alpha=0.4)
    plt.ylim([-0.8,0.8]); plt.xlim([-0.25,0.5])
    plt.plot(0.5*(trange[0:-1] + trange[1:]), late0mean, 'k')
    plt.subplot(2,1,2)
    plt.plot(0.5*(trange[0:-1] + trange[1:]), late[late_cluster_km_labels==1].T, color='tab:cyan', alpha=0.4)
    plt.ylim([-0.8,0.8]); plt.xlim([-0.25,0.5])
    plt.plot(0.5*(trange[0:-1] + trange[1:]), late1mean, 'k')
    plt.tight_layout(); pdf.savefig(); plt.close()
    late_inds = df1['norm_deflection_at_opp_direction_comp'][df1['movement_psth_type_simple']=='late'].index.values
    for i in range(np.size(late, 0)):
        real_ind = late_inds[i]
        df1.at[real_ind, 'late_comp_responsive'] = (True if late_cluster_km_labels[i]==comp_responsive_late_cluster else False)

    for ind, row in df1.iterrows():
        if row['movement_psth_type_simple'] == 'late':
            if row['late_comp_responsive']:
                df1.at[ind, 'movement_psth_type'] = 'late positive and compensatory responsive'
            elif not row['late_comp_responsive']:
                df1.at[ind, 'movement_psth_type'] = 'late positive'
        elif row['movement_psth_type_simple'] != 'early':
            df1.at[ind, 'movement_psth_type'] = row['movement_psth_type_simple']
        elif row['movement_psth_type_simple'] == 'early':
            df1.at[ind, 'movement_psth_type'] = 'early positive'
            # if row['early_comp_responsive']:
            #     df1.at[ind, 'movement_psth_type'] = 'early positive and compensatory responsive'
            # elif not row['early_comp_responsive']:
            #     df1.at[ind, 'movement_psth_type'] = 'early positive'
    
    print('plotting clusters')
    plt.subplots(2,3, figsize=(14,9))
    trange = np.arange(-1,1.1,0.025)
    labels = sorted(df1['movement_psth_type'].unique())
    for count, label in enumerate(labels):
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        cluster = df1[df1['movement_psth_type']==label]
        plt.subplot(2,3,count+1)
        for ind, row in cluster.iterrows():
            plt.plot(0.5*(trange[0:-1] + trange[1:]), row['norm_deflection_at_pref_direction'], color=tempcolor, alpha=0.2)
        plt.plot(0.5*(trange[0:-1] + trange[1:]), np.mean(cluster['norm_deflection_at_pref_direction'], 0), 'k')
        plt.xlim([-0.25,0.5])
        plt.title(label+' -- gaze shift')
        plt.vlines(0, -1, 1, linestyles='dotted', colors='k')
        plt.ylim([-0.8,0.8])
    plt.tight_layout(); pdf.savefig(); plt.close()

    print('plotting clusters when compensatory')
    plt.subplots(2,3, figsize=(14,9))
    trange = np.arange(-1,1.1,0.025)
    labels = sorted(df1['movement_psth_type'].unique())
    for count, label in enumerate(labels):
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        cluster = df1[df1['movement_psth_type']==label]
        plt.subplot(2,3,count+1)
        for ind, row in cluster.iterrows():
            plt.plot(0.5*(trange[0:-1] + trange[1:]), row['norm_deflection_at_pref_direction_comp'], color=tempcolor, alpha=0.2)
        plt.plot(0.5*(trange[0:-1] + trange[1:]), np.mean(cluster['norm_deflection_at_pref_direction_comp'], 0), 'k')
        plt.xlim([-0.25,0.5])
        plt.title(label+' -- same comp')
        plt.vlines(0, -1, 1, linestyles='dotted', colors='k')
        plt.ylim([-0.8,0.8])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.subplots(2,3, figsize=(14,9))
    trange = np.arange(-1,1.1,0.025)
    labels = sorted(df1['movement_psth_type'].unique())
    for count, label in enumerate(labels):
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        cluster = df1[df1['movement_psth_type']==label]
        plt.subplot(2,3,count+1)
        for ind, row in cluster.iterrows():
            plt.plot(0.5*(trange[0:-1] + trange[1:]), row['norm_deflection_at_opp_direction'], color=tempcolor, alpha=0.2)
        plt.plot(0.5*(trange[0:-1] + trange[1:]), np.mean(cluster['norm_deflection_at_opp_direction'], 0), 'k')
        plt.xlim([-0.25,0.5])
        plt.title(label+' -- opposite gaze shift')
        plt.vlines(0, -1, 1, linestyles='dotted', colors='k')
        plt.ylim([-0.8,0.8])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.subplots(2,3, figsize=(14,9))
    trange = np.arange(-1,1.1,0.025)
    labels = sorted(df1['movement_psth_type'].unique())
    for count, label in enumerate(labels):
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        cluster = df1[df1['movement_psth_type']==label]
        plt.subplot(2,3,count+1)
        for ind, row in cluster.iterrows():
            plt.plot(0.5*(trange[0:-1] + trange[1:]), row['norm_deflection_at_opp_direction_comp'], color=tempcolor, alpha=0.2)
        plt.plot(0.5*(trange[0:-1] + trange[1:]), np.mean(cluster['norm_deflection_at_opp_direction_comp'], 0), 'k')
        plt.xlim([-0.25,0.5])
        plt.title(label+' -- opposite comp')
        plt.vlines(0, -1, 1, linestyles='dotted', colors='k')
        plt.ylim([-0.8,0.8])
    plt.tight_layout(); pdf.savefig(); plt.close()
    
    plt.figure(figsize=(8,6))
    for count, label in enumerate(labels):
        cluster = flatten_series(df1['norm_deflection_at_pref_direction'][df1['movement_psth_type']==label])
        cluster_mean = np.nanmean(cluster, 0)
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        plt.plot(0.5*(trange[0:-1] + trange[1:]), cluster_mean, color=tempcolor, label=label, linewidth=5)
        plt.legend()
        plt.xlim([-0.25,0.5])
        plt.vlines(0,-0.5,0.5,linestyles='dotted',colors='k')
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.subplots(2,3, figsize=(14,9))
    for count, label in enumerate(labels):
        plt.subplot(2,3,count+1)
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        plt.hist(df1['hf1_wn_depth_from_layer5'][df1['movement_psth_type']==label],bins=list(np.arange(-650,650+100,100)),orientation='horizontal',color=tempcolor)
        plt.title(label)
        plt.gca().invert_yaxis()
    plt.tight_layout(); pdf.savefig(); plt.close()

    props = ['dsi_for_sf_pref', 'osi_for_sf_pref', 'hf1_wn_crf_modind', 'sf_pref', 'tf_pref']
    prop_labels = ['direction selectivity','orientation selectivity','contrast response','prefered spatial frequency','prefered temporal frequency']
    plt.subplots(5,1,figsize=(16,25))
    for fig_count, prop in enumerate(props):
        ax = plt.subplot(5,1,fig_count+1)
        for label_count, label in enumerate(labels):
            s = df1[prop][df1['movement_psth_type']==label].dropna()
            s_mean = np.nanmean(s)
            stderr = np.nanstd(s) / np.sqrt(np.size(s,0))
            lbound = label_count-0.2; ubound = label_count+0.2
            x_jitter = np.random.uniform(lbound, ubound, np.size(s,0))
            tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][label_count]
            plt.plot(x_jitter, np.array(s), '.', color=tempcolor)
            plt.hlines(s_mean, lbound, ubound, color=tempcolor, linewidth=5)
            plt.vlines(label_count, s_mean-stderr, s_mean+stderr, color=tempcolor, linewidth=5)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        plt.ylabel(prop_labels[fig_count])
    plt.tight_layout(); pdf.savefig(); plt.close()

    for ind, row in df1.iterrows():
        ori_tuning = np.mean(row['hf3_gratings_ori_tuning'],2) # [orientation, sf, tf]
        drift_spont = row['hf3_gratings_drift_spont']
        tuning = ori_tuning - drift_spont # subtract off spont rate
        tuning[tuning < 0] = 0 # set to 0 when tuning goes negative (i.e. when firing rate is below spontanious rate)
        th_pref = np.nanargmax(tuning,0) # get position of highest firing rate
        prefered_direction = np.zeros(3)
        prefered_orientation = np.zeros(3)
        best_tuning_for_sf = np.zeros(3)
        for sf in range(3):
            R_pref = (tuning[th_pref[sf], sf] + (tuning[(th_pref[sf]+4)%8, sf])) * 0.5 # get that firing rate (avg between peaks)
            th_ortho = (th_pref[sf]+2)%8 # get ortho position
            R_ortho = (tuning[th_ortho, sf] + (tuning[(th_ortho+4)%8, sf])) * 0.5  # ortho firing rate (average between two peaks)
            th_null = (th_pref[sf]+4)%8 # get other direction of same orientation
            R_null = tuning[th_null, sf] # tuning value at that peak
            prefered_direction[sf] = (np.arange(8)*45)[th_null]
            prefered_orientation[sf] = (np.arange(8)*45)[th_pref[sf]]
            best_tuning_for_sf[sf] = R_pref
        best_sf_ind = np.argmax(best_tuning_for_sf)
        df1.at[ind, 'best_direction'] = prefered_direction[best_sf_ind]
        df1.at[ind, 'best_orientation'] = prefered_orientation[best_sf_ind]
    plt.subplots(4,2, figsize=(16,9))
    plt.subplot(4,2,1)
    plt.hist(df1['best_orientation'], bins=8, density=True, color='tab:gray')
    plt.title('all')
    for count, label in enumerate(labels):
        plt.subplot(4,2,count+2)
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        plt.hist(df1['best_orientation'][df1['movement_psth_type']==label], bins=8, density=True, color=tempcolor)
        plt.title(label +' count='+str(len(df1['best_orientation'][df1['movement_psth_type']==label])))
    plt.tight_layout(); pdf.savefig(); plt.close()

    gratings_polar_fig = gratings_polar_plot(df1,'best_orientation',labels)
    pdf.savefig(); plt.close()

    key_data = np.zeros([len(labels),2])
    label_count = 0
    for label in labels:
        num_inh = len(df1[df1['movement_psth_type']==label][df1['waveform_km_label']==0])
        num_exc = len(df1[df1['movement_psth_type']==label][df1['waveform_km_label']==1])
        if num_inh > 0:
            key_data[label_count, 0] = num_inh / len(df1[df1['waveform_km_label']==0])
        if num_exc > 0:
            key_data[label_count, 1] = num_exc / len(df1[df1['waveform_km_label']==1])
        label_count += 1
    fig, ax = plt.subplots(1,1, figsize=(17,6))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, key_data[:,0], width=width, label='inhibitory', color='g')
    plt.bar(x + width/2, key_data[:,1], width=width, label='excitatory', color='b')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout(); pdf.savefig(); plt.close()

    df1['has_dark'] = ~pd.isnull(df1['dark_gaze_shift_using_light_direction_pref'])

    plt.subplots(2,3, figsize=(14,9))
    trange = np.arange(-1,1.1,0.025)
    labels = sorted(df1['movement_psth_type'].unique())
    for count, label in enumerate(labels):
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        cluster = df1[df1['movement_psth_type']==label]
        plt.subplot(2,3,count+1)
        for ind, row in cluster.iterrows():
            if row['has_dark']:
                plt.plot(0.5*(trange[0:-1] + trange[1:]), row['dark_gaze_shift_using_light_direction_pref'], color=tempcolor, alpha=0.2)
        plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(cluster['dark_gaze_shift_using_light_direction_pref'][cluster['has_dark']]), 0), 'k')
        plt.xlim([-0.25,0.5])
        plt.title(label+' -- dark same gaze shift')
        plt.vlines(0, -1, 1, linestyles='dotted', colors='k')
        plt.ylim([-0.8,0.8])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.subplots(2,3, figsize=(14,9))
    trange = np.arange(-1,1.1,0.025)
    labels = sorted(df1['movement_psth_type'].unique())
    for count, label in enumerate(labels):
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        cluster = df1[df1['movement_psth_type']==label]
        plt.subplot(2,3,count+1)
        for ind, row in cluster.iterrows():
            if row['has_dark']:
                plt.plot(0.5*(trange[0:-1] + trange[1:]), row['dark_comp_using_light_direction_pref'], color=tempcolor, alpha=0.2)
        plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(cluster['dark_comp_using_light_direction_pref'][cluster['has_dark']]), 0), 'k')
        plt.xlim([-0.25,0.5])
        plt.title(label+' -- dark same comp')
        plt.vlines(0, -1, 1, linestyles='dotted', colors='k')
        plt.ylim([-0.8,0.8])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.subplots(2,3, figsize=(14,9))
    trange = np.arange(-1,1.1,0.025)
    labels = sorted(df1['movement_psth_type'].unique())
    for count, label in enumerate(labels):
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        cluster = df1[df1['movement_psth_type']==label]
        plt.subplot(2,3,count+1)
        for ind, row in cluster.iterrows():
            if row['has_dark']:
                plt.plot(0.5*(trange[0:-1] + trange[1:]), row['dark_gaze_shift_using_light_direction_opp'], color=tempcolor, alpha=0.2)
        plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(cluster['dark_gaze_shift_using_light_direction_opp'][cluster['has_dark']]), 0), 'k')
        plt.xlim([-0.25,0.5])
        plt.title(label+' -- dark opposite gaze shift')
        plt.vlines(0, -1, 1, linestyles='dotted', colors='k')
        plt.ylim([-0.8,0.8])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.subplots(2,3, figsize=(14,9))
    trange = np.arange(-1,1.1,0.025)
    labels = sorted(df1['movement_psth_type'].unique())
    for count, label in enumerate(labels):
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][count]
        cluster = df1[df1['movement_psth_type']==label]
        plt.subplot(2,3,count+1)
        for ind, row in cluster.iterrows():
            if row['has_dark']:
                plt.plot(0.5*(trange[0:-1] + trange[1:]), row['dark_comp_using_light_direction_opp'], color=tempcolor, alpha=0.2)
        plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(cluster['dark_comp_using_light_direction_opp'][cluster['has_dark']]), 0), 'k')
        plt.xlim([-0.25,0.5])
        plt.title(label+' -- dark opposite comp')
        plt.vlines(0, -1, 1, linestyles='dotted', colors='k')
        plt.ylim([-0.8,0.8])
    plt.tight_layout(); pdf.savefig(); plt.close()

    plt.subplots(2,2,figsize=(12,12))
    plt.subplot(2,2,1)
    plt.title('late positive, light'); plt.xlim([-0.25,0.5]); plt.ylim([-0.8,0.8])
    for ind, row in df1['norm_deflection_at_pref_direction_comp'][df1['movement_psth_type']=='late positive'].iteritems():
        plt.plot(0.5*(trange[0:-1] + trange[1:]), row, color='tab:pink', alpha=0.2)
    plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(df1['norm_deflection_at_pref_direction_comp'][df1['movement_psth_type']=='late positive']),0),'k')
    plt.subplot(2,2,2)
    plt.title('late positive, dark'); plt.xlim([-0.25,0.5]); plt.ylim([-0.8,0.8])
    for ind, row in df1['dark_comp_using_light_direction_pref'][df1['movement_psth_type']=='late positive'][df1['has_dark']].iteritems():
        plt.plot(0.5*(trange[0:-1] + trange[1:]), row, color='tab:pink', alpha=0.2)
    plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(df1['dark_comp_using_light_direction_pref'][df1['movement_psth_type']=='late positive'][df1['has_dark']]),0),'k')
    plt.subplot(2,2,3)
    plt.title('late positive with comp response, light'); plt.xlim([-0.25,0.5]); plt.ylim([-0.8,0.8])
    for ind, row in df1['norm_deflection_at_pref_direction_comp'][df1['movement_psth_type']=='late positive and compensatory responsive'].iteritems():
        plt.plot(0.5*(trange[0:-1] + trange[1:]), row, color='tab:purple', alpha=0.2)
    plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(df1['norm_deflection_at_pref_direction_comp'][df1['movement_psth_type']=='late positive and compensatory responsive']),0),'k')
    plt.subplot(2,2,4)
    plt.title('late positive with comp response, dark'); plt.xlim([-0.25,0.5]); plt.ylim([-0.8,0.8])
    for ind, row in df1['dark_comp_using_light_direction_pref'][df1['movement_psth_type']=='late positive and compensatory responsive'][df1['has_dark']].iteritems():
        plt.plot(0.5*(trange[0:-1] + trange[1:]), row, color='tab:purple', alpha=0.2)
    plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(df1['dark_comp_using_light_direction_pref'][df1['movement_psth_type']=='late positive and compensatory responsive'][df1['has_dark']]),0),'k')
    plt.tight_layout(); pdf.savefig(); plt.close()

    labels = sorted(df1['movement_psth_type'].unique())
    stims = ['norm_deflection_at_pref_direction','norm_deflection_at_pref_direction_comp','norm_deflection_at_opp_direction','norm_deflection_at_opp_direction_comp',
            'dark_gaze_shift_using_light_direction_pref','dark_comp_using_light_direction_pref','dark_gaze_shift_using_light_direction_opp','dark_comp_using_light_direction_opp']
    stim_titles = ['pref gaze shift','pref comp','opp gaze shift','opp comp','dark pref gaze shift','dark pref comp','dark opp gaze shift','dark opp comp']
    for label_count, label in enumerate(labels):
        tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][label_count]
        cluster = df1[df1['movement_psth_type']==label]
        plt.subplots(2,4,figsize=(14,9))
        for stim_count in range(8):
            stim = stims[stim_count]; stim_name = stim_titles[stim_count]
            if 'dark' in stim_name:
                usecluster = cluster[cluster['has_dark']]
            else:
                usecluster = cluster
            plt.subplot(2,4,stim_count+1)
            for ind, row in usecluster.iterrows():
                plt.plot(0.5*(trange[0:-1] + trange[1:]), usecluster.loc[ind, stim], color=tempcolor, alpha=0.2)
            plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(usecluster[stim]), 0), 'k')
            plt.xlim([-0.25,0.5])
            plt.title(stim_name)
            plt.vlines(0, -1, 1, linestyles='dotted', colors='k')
            plt.ylim([-0.8,0.8])
        plt.tight_layout(); pdf.savefig(); plt.close()

    props = [['fm_dark_spike_rate_vs_theta_modind_neg','fm_dark_spike_rate_vs_theta_modind_pos'],
        ['fm_dark_spike_rate_vs_roll_modind_neg','fm_dark_spike_rate_vs_roll_modind_pos'],
         ['fm_dark_spike_rate_vs_pitch_modind_neg','fm_dark_spike_rate_vs_pitch_modind_pos'],
         ['fm1_spike_rate_vs_theta_modind_neg','fm1_spike_rate_vs_theta_modind_pos'],
         ['fm1_spike_rate_vs_phi_modind_neg','fm1_spike_rate_vs_phi_modind_pos'],
         ['fm1_spike_rate_vs_roll_modind_neg','fm1_spike_rate_vs_roll_modind_pos'],
         ['fm1_spike_rate_vs_pitch_modind_neg','fm1_spike_rate_vs_pitch_modind_pos'],
         ['fm1_spike_rate_vs_gz_modind_neg','fm1_spike_rate_vs_gz_modind_pos'],
        ['fm1_spike_rate_vs_gx_modind_neg','fm1_spike_rate_vs_gx_modind_pos'],
        ['fm1_spike_rate_vs_gy_modind_neg','fm1_spike_rate_vs_gy_modind_pos'],
        ['fm_dark_wn_spike_rate_vs_gz_modind_neg','fm_dark_spike_rate_vs_gz_modind_pos'],
        ['fm_dark_spike_rate_vs_gx_modind_neg','fm_dark_spike_rate_vs_gx_modind_neg'],
        ['fm_dark_spike_rate_vs_gy_modind_pos','fm1_spike_rate_vs_gy_modind_pos']]
    prop_labels = ['dark theta','dark roll','dark pitch','light theta','light phi','light roll','light pitch',
               'light gyro z','light gyro x','light gyro y','dark gyro z','dark gyro x','dark gyro y']
    labels = sorted(df1['movement_psth_type'].unique())
    plt.subplots(13,1,figsize=(19,70))
    for fig_count, prop in enumerate(props):
        ax = plt.subplot(13,1,fig_count+1)
        for label_count, label in enumerate(labels):
            s0 = df1[prop[0]][df1['movement_psth_type']==label].dropna()
            s1 = df1[prop[1]][df1['movement_psth_type']==label].dropna()
            s = np.abs(pd.concat([s0,s1]))
            s_mean = np.nanmean(s)
            stderr = np.nanstd(s) / np.sqrt(np.size(s,0))
            lbound = label_count-0.2; ubound = label_count+0.2
            x_jitter = np.random.uniform(lbound, ubound, np.size(s,0))
            tempcolor = ['tab:orange','tab:purple','tab:blue', 'tab:cyan', 'tab:green', 'tab:red'][label_count]
            plt.plot(x_jitter, np.array(s), '.', color=tempcolor)
            plt.hlines(s_mean, lbound, ubound, color=tempcolor, linewidth=5)
            plt.vlines(label_count, s_mean-stderr, s_mean+stderr, color=tempcolor, linewidth=5)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        plt.ylabel(prop_labels[fig_count])
        plt.ylim([0,1])
    plt.tight_layout(); pdf.savefig(); plt.close()

    # plt.subplots(2,2,figsize=(12,12))
    # plt.subplot(2,2,1)
    # plt.title('early positive, light'); plt.xlim([-0.25,0.5]); plt.ylim([-0.8,0.8])
    # for ind, row in df1['norm_deflection_at_pref_direction_comp'][df1['movement_psth_type']=='early positive'].iteritems():
    #     plt.plot(0.5*(trange[0:-1] + trange[1:]), row, color='tab:blue', alpha=0.2)
    # plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(df1['norm_deflection_at_pref_direction_comp'][df1['movement_psth_type']=='early positive']),0),'k')
    # plt.subplot(2,2,2)
    # plt.title('early positive, dark'); plt.xlim([-0.25,0.5]); plt.ylim([-0.8,0.8])
    # for ind, row in df1['dark_comp_using_light_direction_pref'][df1['movement_psth_type']=='early positive'][df1['has_dark']].iteritems():
    #     plt.plot(0.5*(trange[0:-1] + trange[1:]), row, color='tab:blue', alpha=0.2)
    # plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(df1['dark_comp_using_light_direction_pref'][df1['movement_psth_type']=='early positive'][df1['has_dark']]),0),'k')
    # plt.subplot(2,2,3)
    # plt.title('early positive with comp response, light'); plt.xlim([-0.25,0.5]); plt.ylim([-0.8,0.8])
    # for ind, row in df1['norm_deflection_at_pref_direction_comp'][df1['movement_psth_type']=='early positive and compensatory responsive'].iteritems():
    #     plt.plot(0.5*(trange[0:-1] + trange[1:]), row, color='tab:cyan', alpha=0.2)
    # plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(df1['norm_deflection_at_pref_direction_comp'][df1['movement_psth_type']=='early positive and compensatory responsive']),0),'k')
    # plt.subplot(2,2,4)
    # plt.title('early positive with comp response, dark'); plt.xlim([-0.25,0.5]); plt.ylim([-0.8,0.8])
    # for ind, row in df1['dark_comp_using_light_direction_pref'][df1['movement_psth_type']=='early positive and compensatory responsive'][df1['has_dark']].iteritems():
    #     plt.plot(0.5*(trange[0:-1] + trange[1:]), row, color='tab:cyan', alpha=0.2)
    # plt.plot(0.5*(trange[0:-1] + trange[1:]), np.nanmean(flatten_series(df1['dark_comp_using_light_direction_pref'][df1['movement_psth_type']=='early positive and compensatory responsive'][df1['has_dark']]),0),'k')
    # plt.tight_layout(); pdf.savefig(); plt.close()

    # print('matrix of cluster changes between movement types')
    # fig, axes = plt.subplots(1,1,figsize=(12,12))
    # for this_key_pair in waveform_key_pairs:
    #     count_matrix = np.zeros([5,5])
    #     cluster_dict = {'biphasic':0, 'negative':1, 'early':2, 'late':3, 'unresponsive':4}
    #     key0 = this_key_pair[0]
    #     key1 = this_key_pair[1]
    #     for ind, row in df1.iterrows():
    #         if type(row[key0+'_cluster_type'])==str and type(row[key1+'_cluster_type'])==str:
    #             first_cluster = row[key0+'_cluster_type']
    #             second_cluster = row[key1+'_cluster_type']
    #             pos0 = cluster_dict[first_cluster]
    #             pos1 = cluster_dict[second_cluster]
    #             count_matrix[pos0, pos1] = count_matrix[pos0, pos1] + 1
    #     for i in range(4,5):
    #         count_matrix[i,i] = np.nan
    #     ax = plt.subplot(3,2,count)
    #     im = plt.imshow(count_matrix/np.nanmax(count_matrix), cmap='Blues', vmin=0, vmax=1)
    #     ax.set_xticks(np.arange(5))
    #     ax.set_xticklabels(['biphasic','negative','early','late','unresponsive'])
    #     ax.set_yticks(np.arange(5))
    #     ax.set_yticklabels(['biphasic','negative','early','late','unresponsive'])
    #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #     plt.ylabel(key0); plt.xlabel(key1)
    #     plt.colorbar(im)
    # plt.tight_layout(); pdf.savefig(); plt.close()

    # print('dhead and deye around time of gaze shifting eye movements')
    # var_around_saccade_fig = var_around_saccade(df1, 'eye_gaze_shifting')
    # plt.tight_layout(); pdf.savefig(); plt.close()
    # print('dhead and deye around time of compesatory eye movements')
    # var_around_saccade_fig = var_around_saccade(df1, 'eye_comp')
    # plt.tight_layout(); pdf.savefig(); plt.close()
    # print('dhead and deye around time of gaze shifting head movements')
    # var_around_saccade_fig = var_around_saccade(df1, 'head_gaze_shifting')
    # plt.tight_layout(); pdf.savefig(); plt.close()
    # print('dhead and deye around time of compensatory head movements')
    # var_around_saccade_fig = var_around_saccade(df1, 'head_comp')
    # plt.tight_layout(); pdf.savefig(); plt.close()

    print('saving population summary pdf')
    pdf.close()
    df1.to_pickle(os.path.join(savepath, 'pooled_ephys_population_update_'+datetime.today().strftime('%m%d%y')+'.pickle'))

    print('done')

    return df1

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
    # fix typos
    cols = df.columns.values
    shcols = [c for c in cols if 'gratingssh' in c]
    for c in shcols:
        new_col = str(c.replace('gratingssh', 'gratings'))
        df = df.rename(columns={str(c): new_col})
    # remove fm2, hf5-8 recordings
    cols = df.columns.values; badcols = []
    for c in cols:
        if any(s in c for s in ['hf5','hf6','hf7','hf8']):
            badcols.append(c)
    df = df.drop(labels=badcols, axis=1)
    # drop duplicate columns
    duplicates = df.columns.values[df.columns.duplicated()]
    for d in duplicates:
        temp = df[d].iloc[:,0].combine_first(df[d].iloc[:,1])
        df = df.drop(columns=d)
        df[d] = temp
    print('saving pooled ephys data to '+config['population']['save_path'])
    path_out = os.path.join(config['population']['save_path'],'pooled_ephys_'+datetime.today().strftime('%m%d%y')+'.pickle')
    if os.path.isfile(path_out):
        os.remove(path_out)
    df = df.reset_index()
    df.to_pickle(path_out)
    print('writing session summary')
    make_session_summary(df, config['population']['save_path'])
    print('writing unit summary')
    unit_df = make_unit_summary(df, config['population']['save_path'])
    del df
    print('starting unit population analysis')
    make_population_summary(unit_df, config['population']['save_path'])