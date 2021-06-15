"""
ephys_population.py
"""
import pandas as pd
import numpy as np
import xarray as xr
import os, sys
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from project_analysis.ephys.ephys_utils import *

def make_unit_summary(df, savepath):
    pdf = PdfPages(os.path.join(savepath, 'unit_summary.pdf'))
    samprate = 30000
    for index, row in tqdm(df.iterrows()):

        # set which fm recording to use
        try:
            fmA = row['best_fm_rec']
        except:
            fmA = 'fm1'

        unitfig = plt.figure(constrained_layout=True, figsize=(12,18))
        spec = gridspec.GridSpec(ncols=3, nrows=8, figure=unitfig)

        # waveform
        unitfig_wv = unitfig.add_subplot(spec[0, 0])
        wv = row['waveform']
        unitfig_wv.plot(np.arange(len(wv))*1000/samprate,wv)
        unitfig_wv.set_title(str(row['session'])+'_unit'+str(index)+' '+row['KSLabel']+' cont='+str(np.round(row['ContamPct'],4)))

        try:
            unitfig_crf = unitfig.add_subplot(spec[1, 0])
            var_cent = row['hf1_wn_crf_cent']
            tuning = row['hf1_wn_crf_tuning']
            tuning_err = row['hf1_wn_crf_err']
            unitfig_crf.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            new_crf = tuning[~np.isnan(tuning)]
            modind = np.round((new_crf[-1] - new_crf[0]) / (new_crf[-1] + new_crf[0]), 3)
            unitfig_crf.set_title('WN contrast response; modulation index='+str(modind))
            unitfig_crf.set_xlabel('contrast a.u.'); unitfig_crf.set_ylabel('sp/sec')
            unitfig_crf.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # wn sta
            unitfig_wnsta = unitfig.add_subplot(spec[0, 1])
            wnsta = np.reshape(row['hf1_wn_spike_triggered_average'],tuple(row['hf1_wn_sta_shape']))
            wnstaRange = np.max(np.abs(wnsta))*1.2
            if wnstaRange<0.25:
                wnstaRange=0.25
            unitfig_wnsta.set_title('WN spike triggered average')
            unitfig_wnsta.imshow(wnsta,vmin=-wnstaRange,vmax=wnstaRange,cmap='jet')
            unitfig_wnsta.axis('off')
        except:
            pass

        try:
            # wn stv
            unitfig_wnstv = unitfig.add_subplot(spec[1, 1])
            wnstv = np.reshape(row['hf1_wn_spike_triggered_variance'],tuple(row['hf1_wn_sta_shape']))
            unitfig_wnstv.imshow(wnstv,vmin=-1,vmax=1)
            unitfig_wnstv.set_title('WN spike triggered variance')
            unitfig_wnstv.axis('off')
        except:
            pass

        try:
            # fm1 sta
            unitfig_fm1sta = unitfig.add_subplot(spec[0, 2])
            fm1sta = np.reshape(row[fmA+'_spike_triggered_average'],tuple(row[fmA+'_sta_shape'])) #change to fm1!
            fm1staRange = np.max(np.abs(fm1sta))*1.2
            if fm1staRange<0.25:
                fm1staRange=0.25
            unitfig_fm1sta.set_title('FM1 spike triggered average')
            unitfig_fm1sta.imshow(fm1sta,vmin=-fm1staRange,vmax=fm1staRange,cmap='jet')
            unitfig_fm1sta.axis('off')
        except:
            pass

        try:
            # fm1 stv
            unitfig_fm1stv = unitfig.add_subplot(spec[1, 2])
            wnstv = np.reshape(row[fmA+'_spike_triggered_variance'],tuple(row[fmA+'_sta_shape']))
            unitfig_fm1stv.imshow(wnstv,vmin=-1,vmax=1)
            unitfig_fm1stv.set_title('FM1 spike triggered variance')
            unitfig_fm1stv.axis('off')
        except:
            pass

        try:
            # orientation tuning curve
            unitfig_ori_tuning = unitfig.add_subplot(spec[3, 0])
            ori_tuning = row['hf3_gratings_ori_tuning']
            drift_spont = row['hf3_gratings_drift_spont']
            R_pref = (np.arange(8)*45)[np.argmax(ori_tuning, 0)]
            R_ortho = R_pref + np.rad2deg(np.pi/2)
            osi = np.round((R_pref - R_ortho) / (R_pref + R_ortho),3)
            unitfig_ori_tuning.set_title('orientation tuning; OSI low='+str(osi[0])+'mid='+str(osi[1])+'high='+str(osi[2]))
            unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,0],label = 'low sf')
            unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,1],label = 'mid sf')
            unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,2],label = 'hi sf')
            unitfig_ori_tuning.plot([0,315],[drift_spont,drift_spont],'r:',label='spont')
            unitfig_ori_tuning.legend()
            unitfig_ori_tuning.set_ylim([0,np.max(ori_tuning)*1.2])
        except:
            pass

        try:
            # fm1 eye movements
            unitfig_fm1saccavg = unitfig.add_subplot(spec[2, 2])
            trange = row[fmA+'_trange']
            upsacc_avg = row[fmA+'_upsacc_avg']; downsacc_avg = row[fmA+'_downsacc_avg']
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_fm1saccavg.set_title('FM1 left/right saccades, mod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
            unitfig_fm1saccavg.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1saccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            unitfig_fm1saccavg.set_ylim([0,maxval*1.2])
        except:
            pass

        try:
            # wn eye movements
            unitfig_wnsaccavg = unitfig.add_subplot(spec[2, 1])
            trange = row['hf1_wn_trange']
            upsacc_avg = row['hf1_wn_upsacc_avg']; downsacc_avg = row['hf1_wn_downsacc_avg']
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_wnsaccavg.set_title('WN left/right saccades, mod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
            unitfig_wnsaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_wnsaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_wnsaccavg.legend(['right','left'])
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            unitfig_wnsaccavg.set_ylim([0,maxval*1.2])
        except:
            pass

        try:
            # wn spike rate vs pupil radius
            unitfig_wnsrpupilrad = unitfig.add_subplot(spec[3, 1])
            var_cent = row['hf1_wn_spike_rate_vs_pupil_radius_cent']
            tuning = row['hf1_wn_spike_rate_vs_pupil_radius_tuning']
            tuning_err = row['hf1_wn_spike_rate_vs_pupil_radius_err']
            modind = modulation_index(tuning)
            unitfig_wnsrpupilrad.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_wnsrpupilrad.set_title('WN spike rate vs pupil radius, mod.ind.='+str(modind[0])+'/'+str(modind[1]))
            unitfig_wnsrpupilrad.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # fm1 spike rate vs pupil radius
            unitfig_fm1srpupilrad = unitfig.add_subplot(spec[3, 2])
            var_cent = row[fmA+'_spike_rate_vs_pupil_radius_cent']
            tuning = row[fmA+'_spike_rate_vs_pupil_radius_tuning']
            tuning_err = row[fmA+'_spike_rate_vs_pupil_radius_err']
            modind = modulation_index(tuning)
            unitfig_fm1srpupilrad.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srpupilrad.set_title('FM1 spike rate vs pupil radius, mod.ind.='+str(modind[0])+'/'+str(modind[1]))
            unitfig_fm1srpupilrad.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # fm1 spike rate vs theta
            unitfig_fm1srth = unitfig.add_subplot(spec[4, 2])
            var_cent = row[fmA+'_spike_rate_vs_theta_cent']
            tuning = row[fmA+'_spike_rate_vs_theta_tuning']
            tuning_err = row[fmA+'_spike_rate_vs_theta_err']
            modind = modulation_index(tuning)
            unitfig_fm1srth.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srth.set_title('FM1 spike rate vs theta, mod.ind.='+str(modind[0])+'/'+str(modind[1]))
            unitfig_fm1srth.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # wn spike rate vs theta
            unitfig_wnsrth = unitfig.add_subplot(spec[4, 1])
            var_cent = row['hf1_wn_spike_rate_vs_theta_cent']
            tuning = row['hf1_wn_spike_rate_vs_theta_tuning']
            tuning_err = row['hf1_wn_spike_rate_vs_theta_err']
            modind = modulation_index(tuning)
            unitfig_wnsrth.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_wnsrth.set_title('WN spike rate vs theta, mod.ind.='+str(modind[0])+'/'+str(modind[1]))
            unitfig_wnsrth.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # wn spike rate vs gz
            unitfig_wnsrvgz = unitfig.add_subplot(spec[4, 0])
            var_cent = row['hf1_wn_spike_rate_vs_gz_cent']
            tuning = row['hf1_wn_spike_rate_vs_gz_tuning']
            tuning_err = row['hf1_wn_spike_rate_vs_gz_err']
            modind = modulation_index(tuning)
            unitfig_wnsrvgz.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_wnsrvgz.set_title('WN spike rate vs running speed, mod.ind.='+str(modind[0])+'/'+str(modind[1]))
            unitfig_wnsrvgz.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # fm1 spike rate vs gz
            unitfig_fm1srvgz = unitfig.add_subplot(spec[5, 0])
            var_cent = row[fmA+'_spike_rate_vs_gz_cent']
            tuning = row[fmA+'_spike_rate_vs_gz_tuning']
            tuning_err = row[fmA+'_spike_rate_vs_gz_err']
            modind = modulation_index(tuning)
            unitfig_fm1srvgz.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srvgz.set_title('FM1 spike rate vs gyro_z, mod.ind.='+str(modind[0])+'/'+str(modind[1]))
            unitfig_fm1srvgz.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # fm1 spike rate vs gx
            unitfig_fm1srvgx = unitfig.add_subplot(spec[5, 1])
            var_cent = row[fmA+'_spike_rate_vs_gx_cent']
            tuning = row[fmA+'_spike_rate_vs_gx_tuning']
            tuning_err = row[fmA+'_spike_rate_vs_gx_err']
            modind = modulation_index(tuning)
            unitfig_fm1srvgx.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srvgx.set_title('FM1 spike rate vs gyro_x, mod.ind.='+str(modind[0])+'/'+str(modind[1]))
            unitfig_fm1srvgx.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # fm1 spike rate vs gy
            unitfig_fm1srvgy = unitfig.add_subplot(spec[5, 2])
            var_cent = row[fmA+'_spike_rate_vs_gy_cent']
            tuning = row[fmA+'_spike_rate_vs_gy_tuning']
            tuning_err = row[fmA+'_spike_rate_vs_gy_err']
            modind = modulation_index(tuning)
            unitfig_fm1srvgy.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srvgy.set_title('FM1 spike rate vs gyro_y, mod.ind.='+str(modind[0])+'/'+str(modind[1]))
            unitfig_fm1srvgy.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

            ### fm1
            # gaze shift dEye
        try:
            unitfig_fm1upsacc_gazedEye = unitfig.add_subplot(spec[6, 0])
            upsacc_avg = row[fmA+'_upsacc_avg_gaze_shift_dEye']
            downsacc_avg = row[fmA+'_downsacc_avg_gaze_shift_dEye']
            trange = row[fmA+'_trange']
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_fm1upsacc_gazedEye.set_title('FM1 gaze shift dEye left/right saccades, mod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
            unitfig_fm1upsacc_gazedEye.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1upsacc_gazedEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_gazedEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_gazedEye.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_gazedEye.set_ylabel('sp/sec')
        except:
            pass

        try:
            # comp dEye
            unitfig_fm1upsacc_compdEye = unitfig.add_subplot(spec[6, 1])
            upsacc_avg = row[fmA+'_upsacc_avg_comp_dEye']
            downsacc_avg = row[fmA+'_downsacc_avg_comp_dEye']
            trange = row[fmA+'_trange']
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_fm1upsacc_compdEye.set_title('FM1 comp dEye left/right saccades, mod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
            unitfig_fm1upsacc_compdEye.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1upsacc_compdEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_compdEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_compdEye.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_compdEye.set_ylabel('sp/sec')
        except:
            pass
        
        try:
            # gaze shift dEye
            unitfig_fm1upsacc_gazedHead = unitfig.add_subplot(spec[6, 2])
            upsacc_avg = row[fmA+'_upsacc_avg_gaze_shift_dHead']
            downsacc_avg = row[fmA+'_downsacc_avg_gaze_shift_dHead']
            trange = row[fmA+'_trange']
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_fm1upsacc_gazedHead.set_title('FM1 gaze shift dHead left/right saccades, mod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
            unitfig_fm1upsacc_gazedHead.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1upsacc_gazedHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_gazedHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_gazedHead.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_gazedHead.set_ylabel('sp/sec')
        except:
            pass
        
        try:
            # gaze shift dHead
            unitfig_fm1upsacc_compdHead = unitfig.add_subplot(spec[7, 0])
            upsacc_avg = row[fmA+'_upsacc_avg_comp_dHead']
            downsacc_avg = row[fmA+'_downsacc_avg_comp_dHead']
            trange = row[fmA+'_trange']
            maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
            modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
            unitfig_fm1upsacc_compdHead.set_title('FM1 comp dHead left/right saccades, mod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
            unitfig_fm1upsacc_compdHead.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1upsacc_compdHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_fm1upsacc_compdHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
            unitfig_fm1upsacc_compdHead.set_ylim([0,maxval*1.2])
            unitfig_fm1upsacc_compdHead.set_ylabel('sp/sec')
        except:
            pass
        
        try:
            # psth gratings
            unitfig_grat_psth = unitfig.add_subplot(spec[2, 0])
            lower = -0.5; upper = 1.5; dt = 0.1
            bins = np.arange(lower,upper+dt,dt)
            psth = row['hf3_gratings_grating_psth']
            unitfig_grat_psth.plot(bins[0:-1]+ dt/2,psth)
            unitfig_grat_psth.set_title('gratings psth')
            unitfig_grat_psth.set_xlabel('time'); unitfig_grat_psth.set_ylabel('sp/sec')
            unitfig_grat_psth.set_ylim([0,np.nanmax(psth)*1.2])
        except:
            pass

        # LFP trace relative to center of layer 4
        try:
            unitfig_lfp = unitfig.add_subplot(spec[7, 1])
            if np.size(row['hf4_revchecker_revchecker_mean_resp_per_ch'],0) == 64:
                shank_channels = [c for c in range(np.size(row['hf4_revchecker_revchecker_mean_resp_per_ch'], 0)) if int(np.floor(c/32)) == int(np.floor(int(row['ch'])/32))]
                whole_shank = row['hf4_revchecker_revchecker_mean_resp_per_ch'][shank_channels]
                unitfig_lfp.plot(whole_shank.T, color='k', alpha=0.1, linewidth=1)
                unitfig_lfp.plot(whole_shank[np.argmin(np.min(whole_shank, axis=1)),:], label='layer4', color='r')
            else:
                unitfig_lfp.plot(row['hf4_revchecker_revchecker_mean_resp_per_ch'].T, color='k', alpha=0.1, linewidth=1)
            unitfig_lfp.plot(row['hf4_revchecker_revchecker_mean_resp_per_ch'][row['ch']], label='this channel', color='b')
            session_inds = [i for i,r in df.iterrows() if r['session'] == row['session']]
            indrange = list(range(len(session_inds))); label_dict = dict(zip(session_inds, indrange))
            plain_ind = label_dict[row['ch']]
            unitfig_lfp.set_title('ch='+str(row['ch'])+'pos='+str(row['hf4_revchecker_lfp_rel_depth'][plain_ind]))
            unitfig_lfp.legend(); unitfig_lfp.axvline(x=(0.1*30000), color='k', linewidth=1)
            unitfig_lfp.set_xticks(np.arange(0,18000,18000/8))
            unitfig_lfp.set_xticklabels(np.arange(-100,500,75))
            unitfig_lfp.set_xlabel('msec'); unitfig_lfp.set_ylabel('uvolts')
        except:
            pass

        pdf.savefig(unitfig)
        plt.close()
    print('saving pdf')
    pdf.close()

def make_session_summary(df, savepath):
    pdf = PdfPages(os.path.join(savepath, 'session_summary.pdf'))
    df['unit'] = df.index.values
    df = df.set_index('session')

    unique_inds = sorted(list(set(df.index.values)))

    for unique_ind in tqdm(unique_inds):
        uniquedf = df.loc[unique_ind]
        # set up subplots
        plt.subplots(2,4,figsize=(25,10))
        plt.suptitle(unique_ind+'eye fit: m='+str(uniquedf['best_ellipse_fit_m'].iloc[0])+' r='+str(uniquedf['best_ellipse_fit_r'].iloc[0]))
        # eye position vs head position
        try:
            plt.subplot(2,4,1)
            plt.title('dEye vs dHead')
            dEye = uniquedf['fm1_dEye'].iloc[0]
            dHead = uniquedf['fm1_dHead'].iloc[0]
            eyeT = uniquedf['fm1_eyeT'].iloc[0]
            if len(dEye[0:-1:10]) == len(dHead(eyeT[0:-1:10])):
                plt.plot(dEye[0:-1:10],dHead(eyeT[0:-1:10]),'.')
            elif len(dEye[0:-1:10]) > len(dHead(eyeT[0:-1:10])):
                len_diff = len(dEye[0:-1:10]) - len(dHead(eyeT[0:-1:10]))
                plt.plot(dEye[0:-1:10][:-len_diff],dHead(eyeT[0:-1:10]),'.')
            elif len(dEye[0:-1:10]) < len(dHead(eyeT[0:-1:10])):
                len_diff = len(dHead(eyeT[0:-1:10])) - len(dEye[0:-1:10])
                plt.plot(dEye[0:-1:10],dHead(eyeT[0:-1:10])[:-len_diff],'.')
            plt.xlabel('dEye'); plt.ylabel('dHead'); plt.xlim((-10,10)); plt.ylim((-10,10))
            plt.plot([-10,10],[10,-10], 'r')
        except:
            pass
        try:
            # histogram of theta from -45 to 45deg (are eye movements in resonable range?)
            plt.subplot(2,4,2)
            plt.title('hist of FM theta')
            plt.hist(uniquedf['fm1_theta'].iloc[0], range=[-45,45])
            # histogram of phi from -45 to 45deg (are eye movements in resonable range?)
            plt.subplot(2,4,3)
            plt.title('hist of FM phi')
            plt.hist(uniquedf['fm1_phi'].iloc[0], range=[-45,45])
            # histogram of gyro z (resonable range?)
            plt.subplot(2,4,4)
            plt.title('hist of FM gyro z')
            plt.hist(uniquedf['fm1_gz'].iloc[0], range=[2,4])
            # plot of contrast response functions on same panel scaled to max 30sp/sec
            # plot of average contrast reponse function across units
            plt.subplot(2,4,5)
            plt.title('contrast response functions')
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
                        plt.subplot(2,4,6)
                        plt.plot(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0][ch_num], color=colors[ch_num], linewidth=1)
                        plt.title('lfp trace, shank1'); plt.axvline(x=(0.1*30000))
                        plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                        plt.ylim([-1200,400])
                    if ch_num>31:
                        plt.subplot(2,4,7)
                        plt.plot(uniquedf['hf4_revchecker_revchecker_mean_resp_per_ch'].iloc[0][ch_num], color=colors[ch_num-32], linewidth=1)
                        plt.title('lfp trace, shank2'); plt.axvline(x=(0.1*30000))
                        plt.xticks(np.arange(0,18000,18000/5),np.arange(0,600,600/5))
                        plt.ylim([-1200,400])
            # fm spike raster
            plt.subplot(2,4,8)
            plt.title('FM spike raster')
            i = 0
            for ind, row in uniquedf.iterrows():
                plt.vlines(row['fm1_spikeT'],i-0.25,i+0.25)
                plt.xlim(0, 10); plt.xlabel('secs'); plt.ylabel('unit #')
                i = i+1
        except:
            pass
        pdf.savefig()
        plt.close()
    print('saving pdf')
    pdf.close()