"""
unit_summary.py

utilities for using ephys analysis outputs
"""
import pandas as pd
import numpy as np
import xarray as xr
import os
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

def make_unit_summary(df, savepath):
    pdf = PdfPages(os.path.join(savepath, 'unit_summary.pdf'))
    samprate = 30000
    for index, row in tqdm(df.iterrows()):

        # set which fm recording to use
        if not row['best_fm_rec']:
            fmA = 'fm1'
        else:
            fmA = row['best_fm_rec']

        unitfig = plt.figure(constrained_layout=True, figsize=(12,18))
        spec = gridspec.GridSpec(ncols=3, nrows=8, figure=unitfig)

        # waveform
        unitfig_wv = unitfig.add_subplot(spec[0, 0])
        wv = row['waveform']
        unitfig_wv.plot(np.arange(len(wv))*1000/samprate,wv)
        unitfig_wv.set_title(str(row['session'])+'_unit'+str(index)+' '+row['KSLabel']+' cont='+str(row['ContamPct']))

        try:
            unitfig_crf = unitfig.add_subplot(spec[4, 1])
            var_cent = row['hf1_wn_crf_cent']
            tuning = row['hf1_wn_crf_tuning']
            tuning_err = row['hf1_wn_crf_err']
            unitfig_crf.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_crf.set_title('WN contrast response')
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
            unitfig_ori_tuning.set_title('GRAT orientation tuning')
            ori_tuning = row['hf3_gratings_ori_tuning']
            drift_spont = row['hf3_gratings_drift_spont']
            unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,0],label = 'low sf')
            unitfig_ori_tuning.plot(np.arange(8)*45,ori_tuning[:,1],label = 'mid sf')
            unitfig_ori_tuning.plot(np.arange(8)*45,ori_tuning[:,2],label = 'hi sf')
            unitfig_ori_tuning.plot([0,315],[drift_spont,drift_spont],'r:',label='spont')
            unitfig_ori_tuning.legend()
        except:
            pass

        try:
            # fm1 eye movements
            unitfig_fm1saccavg = unitfig.add_subplot(spec[2, 2])
            trange = row[fmA+'_trange']
            upsacc_avg = row[fmA+'_upsacc_avg']; downsacc_avg = row[fmA+'_downsacc_avg']
            unitfig_fm1saccavg.set_title('FM1 upsacc/downsacc')
            unitfig_fm1saccavg.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_fm1saccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            maxval = np.max(np.maximum(upsacc_avg[i,:],downsacc_avg[i,:]))
            unitfig_fm1saccavg.ylim([0,maxval*1.2])
        except:
            pass

        try:
            # wn eye movements
            unitfig_wnsaccavg = unitfig.add_subplot(spec[2, 1])
            trange = row['hf1_wn_trange']
            upsacc_avg = row['hf1_wn_upsacc_avg']; downsacc_avg = row['hf1_wn_downsacc_avg']
            unitfig_wnsaccavg.set_title('WN upsacc/downsacc')
            unitfig_wnsaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
            unitfig_wnsaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
            unitfig_wnsaccavg.legend(['upsacc_avg','downsacc_avg'])
            maxval = np.max(np.maximum(upsacc_avg[i,:],downsacc_avg[i,:]))
            unitfig_wnsaccavg.ylim([0,maxval*1.2])
        except:
            pass

        try:
            # wn spike rate vs pupil radius
            unitfig_wnsrpupilrad = unitfig.add_subplot(spec[3, 1])
            var_cent = row['hf1_wn_spike_rate_vs_pupil_radius_cent']
            tuning = row['hf1_wn_spike_rate_vs_pupil_radius_tuning']
            tuning_err = row['hf1_wn_spike_rate_vs_pupil_radius_err']
            unitfig_wnsrpupilrad.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_wnsrpupilrad.set_title('WN spike rate vs pupil radius')
            unitfig_wnsrpupilrad.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # fm1 spike rate vs pupil radius
            unitfig_fm1srpupilrad = unitfig.add_subplot(spec[3, 2])
            var_cent = row[fmA+'_spike_rate_vs_pupil_radius_cent']
            tuning = row[fmA+'_spike_rate_vs_pupil_radius_tuning']
            tuning_err = row[fmA+'_spike_rate_vs_pupil_radius_err']
            unitfig_fm1srpupilrad.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srpupilrad.set_title('FM1 spike rate vs pupil radius')
            unitfig_fm1srpupilrad.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # fm1 spike rate vs theta
            unitfig_fm1srth = unitfig.add_subplot(spec[4, 2])
            var_cent = row[fmA+'_spike_rate_vs_theta_cent']
            tuning = row[fmA+'_spike_rate_vs_theta_tuning']
            tuning_err = row[fmA+'_spike_rate_vs_theta_err']
            unitfig_fm1srth.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srth.set_title('FM1 spike rate vs theta')
            unitfig_fm1srth.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # wn spike rate vs theta
            unitfig_wnsrth = unitfig.add_subplot(spec[4, 1])
            var_cent = row['hf1_wn_spike_rate_vs_theta_cent']
            tuning = row['hf1_wn_spike_rate_vs_theta_tuning']
            tuning_err = row['hf1_wn_spike_rate_vs_theta_err']
            unitfig_wnsrth.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_wnsrth.set_title('WN spike rate vs theta')
            unitfig_wnsrth.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # wn spike rate vs gz
            unitfig_wnsrvgz = unitfig.add_subplot(spec[4, 0])
            var_cent = row['hf1_wn_spike_rate_vs_gz_cent']
            tuning = row['hf1_wn_spike_rate_vs_gz_tuning']
            tuning_err = row['hf1_wn_spike_rate_vs_gz_err']
            unitfig_wnsrvgz.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_wnsrvgz.set_title('WN spike rate vs running speed')
            unitfig_wnsrvgz.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # fm1 spike rate vs gz
            unitfig_fm1srvgz = unitfig.add_subplot(spec[5, 0])
            var_cent = row[fmA+'_spike_rate_vs_gz_cent']
            tuning = row[fmA+'_spike_rate_vs_gz_tuning']
            tuning_err = row[fmA+'_spike_rate_vs_gz_err']
            unitfig_fm1srvgz.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srvgz.set_title('FM1 spike rate vs gyro_z')
            unitfig_fm1srvgz.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # fm1 spike rate vs gx
            unitfig_fm1srvgx = unitfig.add_subplot(spec[5, 1])
            var_cent = row[fmA+'_spike_rate_vs_gx_cent']
            tuning = row[fmA+'_spike_rate_vs_gx_tuning']
            tuning_err = row[fmA+'_spike_rate_vs_gx_err']
            unitfig_fm1srvgx.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srvgx.set_title('FM1 spike rate vs gyro_x')
            unitfig_fm1srvgx.set_ylim(0,np.nanmax(tuning[:]*1.2))
        except:
            pass

        try:
            # fm1 spike rate vs gy
            unitfig_fm1srvgy = unitfig.add_subplot(spec[5, 2])
            var_cent = row[fmA+'_spike_rate_vs_gy_cent']
            tuning = row[fmA+'_spike_rate_vs_gy_tuning']
            tuning_err = row[fmA+'_spike_rate_vs_gy_err']
            unitfig_fm1srvgy.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
            unitfig_fm1srvgy.set_title('FM1 spike rate vs gyro_z')
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
            unitfig_fm1upsacc_gazedEye.set_title('FM1 gaze shift dEye')
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
            unitfig_fm1upsacc_compdEye.set_title('FM1 comp dEye')
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
            unitfig_fm1upsacc_gazedHead.set_title('FM1 gaze shift dHead')
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
            unitfig_fm1upsacc_compdHead.set_title('FM1 comp dHead')
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
            unitfig_grat_psth.set_ylim([0,np.nanmax(psth)*1.2)])
        except:
            pass

        pdf.savefig(unitfig)
        plt.close()
    print('saving pdf')
    pdf.close()