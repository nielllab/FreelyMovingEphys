"""
ephys_population.py
"""
import numpy as np
import os
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

def modulation_index(tuning, zerocent=True, lbound=0, ubound=-1):
    tuning = tuning[~np.isnan(tuning)]
    if zerocent is False:
        return np.round((tuning[ubound] - tuning[lbound]) / (tuning[ubound] + tuning[lbound]), 3)
    elif zerocent is True:
        r0 = np.nanmean(tuning[4:6])
        modind_neg = np.round((tuning[lbound] - r0) / (tuning[lbound] + r0), 3)
        modind_pos = np.round((tuning[ubound] - r0) / (tuning[ubound] + r0), 3)
        return [modind_neg, modind_pos]

def saccade_modulation_index(trange, saccavg):
    t0ind = (np.abs(trange - 0)).argmin()
    t100ind = int((np.abs(trange - 0)).argmin()+(len(trange) * (1/10)))
    baseline = np.nanmean(saccavg[0:int(t100ind-((1/4)*t100ind))])
    r0 = np.round((saccavg[t0ind] - baseline) / (saccavg[t0ind] + baseline), 3)
    r100 = np.round((saccavg[t100ind] - baseline) / (saccavg[t100ind] + baseline), 3)
    return r0, r100

def make_unit_summary(df, savepath):
    pdf = PdfPages(os.path.join(savepath, 'unit_summary.pdf'))
    samprate = 30000
    for index, row in tqdm(df.iterrows()):

        # set which fm recording to use
        try:
            fmA = row['best_fm_rec']
        except:
            fmA = 'fm1'

        unitfig = plt.figure(constrained_layout=True, figsize=(30,22))
        spec = gridspec.GridSpec(ncols=5, nrows=6, figure=unitfig)

        # set up new h5 file to save out including new metrics
        newdf = df.copy()

        # waveform
        unitfig_wv = unitfig.add_subplot(spec[0, 0])
        wv = row['waveform']
        unitfig_wv.plot(np.arange(len(wv))*1000/samprate,wv)
        unitfig_wv.set_title(str(row['session'])+'_unit'+str(index)+' '+row['KSLabel']+'\ncont='+str(np.round(row['ContamPct'],3)))

        unitfig_crf = unitfig.add_subplot(spec[1, 0])
        crange = row['hf1_wn_c_range']
        var_cent = row['hf1_wn_crf_cent']
        tuning = row['hf1_wn_crf_tuning']
        tuning_err = row['hf1_wn_crf_err']
        unitfig_crf.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        new_crf = tuning[~np.isnan(tuning)]
        modind = np.round((new_crf[-1] - new_crf[0]) / (new_crf[-1] + new_crf[0]), 3)
        unitfig_crf.set_title('WN contrast response\nmodulation index='+str(modind))
        unitfig_crf.set_xlabel('contrast a.u.'); unitfig_crf.set_ylabel('sp/sec')
        unitfig_crf.set_ylim(0,np.nanmax(tuning[:]*1.2))#; unitfig_crf.set_xlim([0,1])
        newdf['hf1_wn_crf_modind'].iloc[index] = modind

        # wn sta
        unitfig_wnsta = unitfig.add_subplot(spec[0, 1])
        wnsta = np.reshape(row['hf1_wn_spike_triggered_average'],tuple(row['hf1_wn_sta_shape']))
        wnstaRange = np.max(np.abs(wnsta))*1.2
        if wnstaRange<0.25:
            wnstaRange=0.25
        unitfig_wnsta.set_title('WN spike triggered average')
        unitfig_wnsta.imshow(wnsta,vmin=-wnstaRange,vmax=wnstaRange,cmap='jet')
        unitfig_wnsta.axis('off')

        # wn stv
        unitfig_wnstv = unitfig.add_subplot(spec[1, 1])
        wnstv = np.reshape(row['hf1_wn_spike_triggered_variance'],tuple(row['hf1_wn_sta_shape']))
        unitfig_wnstv.imshow(wnstv,vmin=-1,vmax=1)
        unitfig_wnstv.set_title('WN spike triggered variance')
        unitfig_wnstv.axis('off')

        # fm1 sta
        unitfig_fm1sta = unitfig.add_subplot(spec[0, 2])
        fm1sta = np.reshape(row[fmA+'_spike_triggered_average'],tuple(row[fmA+'_sta_shape'])) #change to fm1!
        fm1staRange = np.max(np.abs(fm1sta))*1.2
        if fm1staRange<0.25:
            fm1staRange=0.25
        unitfig_fm1sta.set_title('FM1 spike triggered average')
        unitfig_fm1sta.imshow(fm1sta,vmin=-fm1staRange,vmax=fm1staRange,cmap='jet')
        unitfig_fm1sta.axis('off')

        # fm1 stv
        unitfig_fm1stv = unitfig.add_subplot(spec[1, 2])
        wnstv = np.reshape(row[fmA+'_spike_triggered_variance'],tuple(row[fmA+'_sta_shape']))
        unitfig_fm1stv.imshow(wnstv,vmin=-1,vmax=1)
        unitfig_fm1stv.set_title('FM1 spike triggered variance')
        unitfig_fm1stv.axis('off')

        # fm1 glm receptive field at five lags
        glm = row[fmA+'_glm_receptive_field']
        glm_cc = row[fmA+'_glm_cc']
        lag_list = [-4,-2,0,2,4]
        crange = np.max(np.abs(glm))
        for glm_lag in range(5):
            unitfig_glm = unitfig.add_subplot(spec[2, glm_lag])
            unitfig_glm.imshow(glm[glm_lag],vmin=-crange,vmax=crange,cmap='jet')
            unitfig_glm.set_title('FM1 GLM receptive field\n(lag='+str(lag_list[glm_lag])+' cc='+str(np.round(glm_cc[glm_lag],2))+')')
            unitfig_glm.axis('off')
    
        # orientation tuning curve
        unitfig_ori_tuning = unitfig.add_subplot(spec[0, 3])
        ori_tuning = row['hf3_gratings_ori_tuning']
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
        unitfig_ori_tuning.set_title('orientation tuning\nOSI low='+str(osi[0])+'mid='+str(osi[1])+'high='+str(osi[2]))
        unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,0],label = 'low sf')
        unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,1],label = 'mid sf')
        unitfig_ori_tuning.plot(np.arange(8)*45, ori_tuning[:,2],label = 'hi sf')
        unitfig_ori_tuning.plot([0,315],[drift_spont,drift_spont],'r:',label='spont')
        unitfig_ori_tuning.legend()
        unitfig_ori_tuning.set_ylim([0,np.nanmax(ori_tuning)*1.2])
        newdf['hf3_gratings_osi'].iloc[index] = osi

        # fm1 eye movements
        unitfig_fm1saccavg = unitfig.add_subplot(spec[0, 4])
        trange = row[fmA+'_trange']
        upsacc_avg = row[fmA+'_upsacc_avg']; downsacc_avg = row[fmA+'_downsacc_avg']
        modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
        unitfig_fm1saccavg.set_title('FM1 left/right saccades\nmod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
        unitfig_fm1saccavg.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
        unitfig_fm1saccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
        maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
        unitfig_fm1saccavg.set_ylim([0,maxval*1.2])
        newdf['hf3_gratings_right_modind'].iloc[index] = osi

        # wn eye movements
        unitfig_wnsaccavg = unitfig.add_subplot(spec[1, 4])
        trange = row['hf1_wn_trange']
        upsacc_avg = row['hf1_wn_upsacc_avg']; downsacc_avg = row['hf1_wn_downsacc_avg']
        modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
        unitfig_wnsaccavg.set_title('WN left/right saccades\nmod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
        unitfig_wnsaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
        unitfig_wnsaccavg.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
        unitfig_wnsaccavg.legend(['right','left'])
        maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
        unitfig_wnsaccavg.set_ylim([0,maxval*1.2])
        newdf['hf1_wn_upsacc_modind'].iloc[index] = modind_right; newdf['hf1_wn_downsacc_modind'].iloc[index] = modind_left

        # wn spike rate vs pupil radius
        unitfig_wnsrpupilrad = unitfig.add_subplot(spec[4, 4])
        var_cent = row['hf1_wn_spike_rate_vs_pupil_radius_cent']
        tuning = row['hf1_wn_spike_rate_vs_pupil_radius_tuning']
        tuning_err = row['hf1_wn_spike_rate_vs_pupil_radius_err']
        modind = modulation_index(tuning)
        unitfig_wnsrpupilrad.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        unitfig_wnsrpupilrad.set_title('WN spike rate vs pupil radius\nmod.ind.='+str(modind[0])+'/'+str(modind[1]))
        unitfig_wnsrpupilrad.set_ylim(0,np.nanmax(tuning[:]*1.2))
        newdf['hf1_wn_spike_rate_vs_pupil_radius_modind'].iloc[index] = modind

        # fm1 spike rate vs pupil radius
        unitfig_fm1srpupilrad = unitfig.add_subplot(spec[3, 4])
        var_cent = row[fmA+'_spike_rate_vs_pupil_radius_cent']
        tuning = row[fmA+'_spike_rate_vs_pupil_radius_tuning']
        tuning_err = row[fmA+'_spike_rate_vs_pupil_radius_err']
        modind = modulation_index(tuning)
        unitfig_fm1srpupilrad.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        unitfig_fm1srpupilrad.set_title('FM1 spike rate vs pupil radius\nmod.ind.='+str(modind[0])+'/'+str(modind[1]))
        unitfig_fm1srpupilrad.set_ylim(0,np.nanmax(tuning[:]*1.2))
        newdf['hf1_wn_spike_rate_vs_pupil_radius_modind'].iloc[index] = modind

        # fm1 spike rate vs theta
        unitfig_fm1srth = unitfig.add_subplot(spec[4, 0])
        var_cent = row[fmA+'_spike_rate_vs_theta_cent']
        tuning = row[fmA+'_spike_rate_vs_theta_tuning']
        tuning_err = row[fmA+'_spike_rate_vs_theta_err']
        modind = modulation_index(tuning)
        unitfig_fm1srth.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        unitfig_fm1srth.set_title('FM1 spike rate vs theta\nmod.ind.='+str(modind[0])+'/'+str(modind[1]))
        unitfig_fm1srth.set_ylim(0,np.nanmax(tuning[:]*1.2))
        newdf['fm1_wn_spike_rate_vs_theta_modind'].iloc[index] = modind

        # fm1 spike rate vs phi
        unitfig_fm1srphi = unitfig.add_subplot(spec[4, 1])
        var_cent = row[fmA+'_spike_rate_vs_phi_cent']
        tuning = row[fmA+'_spike_rate_vs_phi_tuning']
        tuning_err = row[fmA+'_spike_rate_vs_phi_err']
        modind = modulation_index(tuning)
        unitfig_fm1srphi.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        unitfig_fm1srphi.set_title('FM1 spike rate vs phi\nmod.ind.='+str(modind[0])+'/'+str(modind[1]))
        unitfig_fm1srphi.set_ylim(0,np.nanmax(tuning[:]*1.2))
        newdf['fm1_wn_spike_rate_vs_phi_modind'].iloc[index] = modind

        # fm1 spike rate vs roll
        unitfig_fm1srroll = unitfig.add_subplot(spec[4, 2])
        var_cent = row[fmA+'_spike_rate_vs_roll_cent']
        tuning = row[fmA+'_spike_rate_vs_roll_tuning']
        tuning_err = row[fmA+'_spike_rate_vs_roll_err']
        modind = modulation_index(tuning, lbound=5, ubound=-6)
        unitfig_fm1srroll.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        unitfig_fm1srroll.set_title('FM1 spike rate vs roll\nmod.ind.='+str(modind[0])+'/'+str(modind[1]))
        unitfig_fm1srroll.set_ylim(0,np.nanmax(tuning[:]*1.2)); unitfig_fm1srroll.set_xlim(-30,30)
        newdf['fm1_wn_spike_rate_vs_roll_modind'].iloc[index] = modind

        # fm1 spike rate vs pitch
        unitfig_fm1srpitch = unitfig.add_subplot(spec[4, 3])
        var_cent = row[fmA+'_spike_rate_vs_pitch_cent']
        tuning = row[fmA+'_spike_rate_vs_pitch_tuning']
        tuning_err = row[fmA+'_spike_rate_vs_pitch_err']
        modind = modulation_index(tuning, lbound=5, ubound=-6)
        unitfig_fm1srpitch.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        unitfig_fm1srpitch.set_title('FM1 spike rate vs pitch\nmod.ind.='+str(modind[0])+'/'+str(modind[1]))
        unitfig_fm1srpitch.set_ylim(0,np.nanmax(tuning[:]*1.2)); unitfig_fm1srpitch.set_xlim(-30,30)
        newdf['fm1_wn_spike_rate_vs_pitch_modind'].iloc[index] = modind

        # wn spike rate vs gx
        unitfig_wnsrvgz = unitfig.add_subplot(spec[3, 3])
        var_cent = row['hf1_wn_spike_rate_vs_spd_cent']
        tuning = row['hf1_wn_spike_rate_vs_spd_tuning']
        tuning_err = row['hf1_wn_spike_rate_vs_spd_err']
        modind = modulation_index(tuning)
        unitfig_wnsrvgz.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        unitfig_wnsrvgz.set_title('WN spike rate vs running speed\nmod.ind.='+str(modind[0])+'/'+str(modind[1]))
        unitfig_wnsrvgz.set_ylim(0,np.nanmax(tuning[:]*1.2))
        newdf['fm1_wn_spike_rate_vs_spd_modind'].iloc[index] = modind

        # fm1 spike rate vs gz
        unitfig_fm1srvgz = unitfig.add_subplot(spec[3, 2])
        var_cent = row[fmA+'_spike_rate_vs_gz_cent']
        tuning = row[fmA+'_spike_rate_vs_gz_tuning']
        tuning_err = row[fmA+'_spike_rate_vs_gz_err']
        modind = modulation_index(tuning)
        unitfig_fm1srvgz.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        unitfig_fm1srvgz.set_title('FM1 spike rate vs gyro_z\nmod.ind.='+str(modind[0])+'/'+str(modind[1]))
        unitfig_fm1srvgz.set_ylim(0,np.nanmax(tuning[:]*1.2))
        newdf['fm1_wn_spike_rate_vs_gz_modind'].iloc[index] = modind

        # fm1 spike rate vs gx
        unitfig_fm1srvgx = unitfig.add_subplot(spec[3, 0])
        var_cent = row[fmA+'_spike_rate_vs_gx_cent']
        tuning = row[fmA+'_spike_rate_vs_gx_tuning']
        tuning_err = row[fmA+'_spike_rate_vs_gx_err']
        modind = modulation_index(tuning)
        unitfig_fm1srvgx.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        unitfig_fm1srvgx.set_title('FM1 spike rate vs gyro_x\nmod.ind.='+str(modind[0])+'/'+str(modind[1]))
        unitfig_fm1srvgx.set_ylim(0,np.nanmax(tuning[:]*1.2))
        newdf['fm1_wn_spike_rate_vs_gx_modind'].iloc[index] = modind

        # fm1 spike rate vs gy
        unitfig_fm1srvgy = unitfig.add_subplot(spec[3, 1])
        var_cent = row[fmA+'_spike_rate_vs_gy_cent']
        tuning = row[fmA+'_spike_rate_vs_gy_tuning']
        tuning_err = row[fmA+'_spike_rate_vs_gy_err']
        modind = modulation_index(tuning)
        unitfig_fm1srvgy.errorbar(var_cent,tuning[:],yerr=tuning_err[:])
        unitfig_fm1srvgy.set_title('FM1 spike rate vs gyro_y\nmod.ind.='+str(modind[0])+'/'+str(modind[1]))
        unitfig_fm1srvgy.set_ylim(0,np.nanmax(tuning[:]*1.2))
        newdf['fm1_wn_spike_rate_vs_gy_modind'].iloc[index] = modind

        # gaze shift dEye
        unitfig_fm1upsacc_gazedEye = unitfig.add_subplot(spec[5, 0])
        upsacc_avg = row[fmA+'_upsacc_avg_gaze_shift_dEye']
        downsacc_avg = row[fmA+'_downsacc_avg_gaze_shift_dEye']
        trange = row[fmA+'_trange']
        maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
        modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
        unitfig_fm1upsacc_gazedEye.set_title('FM1 gaze shift dEye left/right saccades\nmod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
        unitfig_fm1upsacc_gazedEye.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
        unitfig_fm1upsacc_gazedEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
        unitfig_fm1upsacc_gazedEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
        unitfig_fm1upsacc_gazedEye.set_ylim([0,maxval*1.2])
        unitfig_fm1upsacc_gazedEye.set_ylabel('sp/sec')
        newdf['fm1_upsacc_avg_gaze_shift_dEye_modind'].iloc[index] = modind_right; newdf['fm1_downsacc_avg_gaze_shift_dEye_modind'].iloc[index] = modind_left


        # comp dEye
        unitfig_fm1upsacc_compdEye = unitfig.add_subplot(spec[5, 1])
        upsacc_avg = row[fmA+'_upsacc_avg_comp_dEye']
        downsacc_avg = row[fmA+'_downsacc_avg_comp_dEye']
        trange = row[fmA+'_trange']
        maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
        modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
        unitfig_fm1upsacc_compdEye.set_title('FM1 comp dEye left/right saccades\nmod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
        unitfig_fm1upsacc_compdEye.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
        unitfig_fm1upsacc_compdEye.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
        unitfig_fm1upsacc_compdEye.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
        unitfig_fm1upsacc_compdEye.set_ylim([0,maxval*1.2])
        unitfig_fm1upsacc_compdEye.set_ylabel('sp/sec')
        newdf['fm1_upsacc_avg_comp_dEye_modind'].iloc[index] = modind_right; newdf['fm1_downsacc_avg_comp_dEye_modind'].iloc[index] = modind_left

        # gaze shift dHead
        unitfig_fm1upsacc_gazedHead = unitfig.add_subplot(spec[5, 2])
        upsacc_avg = row[fmA+'_upsacc_avg_gaze_shift_dHead']
        downsacc_avg = row[fmA+'_downsacc_avg_gaze_shift_dHead']
        trange = row[fmA+'_trange']
        maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
        modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
        unitfig_fm1upsacc_gazedHead.set_title('FM1 gaze shift dHead left/right saccades\nmod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
        unitfig_fm1upsacc_gazedHead.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
        unitfig_fm1upsacc_gazedHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
        unitfig_fm1upsacc_gazedHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
        unitfig_fm1upsacc_gazedHead.set_ylim([0,maxval*1.2])
        unitfig_fm1upsacc_gazedHead.set_ylabel('sp/sec')
        newdf['fm1_upsacc_avg_gaze_shift_dHead_modind'].iloc[index] = modind_right; newdf['fm1_downsacc_avg_gaze_shift_dHead_modind'].iloc[index] = modind_left

        # gaze shift comp dHead
        unitfig_fm1upsacc_compdHead = unitfig.add_subplot(spec[5, 3])
        upsacc_avg = row[fmA+'_upsacc_avg_comp_dHead']
        downsacc_avg = row[fmA+'_downsacc_avg_comp_dHead']
        trange = row[fmA+'_trange']
        maxval = np.max(np.maximum(upsacc_avg[:],downsacc_avg[:]))
        modind_right = saccade_modulation_index(trange, upsacc_avg); modind_left = saccade_modulation_index(trange, downsacc_avg)
        unitfig_fm1upsacc_compdHead.set_title('FM1 comp dHead left/right saccades\nmod.ind.='+str(modind_right)+'(left)/'+str(modind_left)+'(right)')
        unitfig_fm1upsacc_compdHead.plot(0.5*(trange[0:-1]+ trange[1:]),upsacc_avg[:])
        unitfig_fm1upsacc_compdHead.plot(0.5*(trange[0:-1]+ trange[1:]),downsacc_avg[:],'r')
        unitfig_fm1upsacc_compdHead.vlines(0,0,np.max(upsacc_avg[:]*0.2),'r')
        unitfig_fm1upsacc_compdHead.set_ylim([0,maxval*1.2])
        unitfig_fm1upsacc_compdHead.set_ylabel('sp/sec')
        newdf['fm1_upsacc_comp_dHead_modind'].iloc[index] = modind_right; newdf['fm1_downsacc_comp_dHead_modind'].iloc[index] = modind_left


        # psth gratings
        unitfig_grat_psth = unitfig.add_subplot(spec[1, 3])
        lower = -0.5; upper = 1.5; dt = 0.1
        bins = np.arange(lower,upper+dt,dt)
        psth = row['hf3_gratings_grating_psth']
        unitfig_grat_psth.plot(bins[0:-1]+ dt/2,psth)
        unitfig_grat_psth.set_title('gratings psth')
        unitfig_grat_psth.set_xlabel('time'); unitfig_grat_psth.set_ylabel('sp/sec')
        unitfig_grat_psth.set_ylim([0,np.nanmax(psth)*1.2])

        # LFP trace relative to center of layer 4
        unitfig_lfp = unitfig.add_subplot(spec[5, 4])
        if np.size(row['hf4_revchecker_revchecker_mean_resp_per_ch'],0) == 64:
            shank_channels = [c for c in range(np.size(row['hf4_revchecker_revchecker_mean_resp_per_ch'], 0)) if int(np.floor(c/32)) == int(np.floor(int(row['ch'])/32))]
            whole_shank = row['hf4_revchecker_revchecker_mean_resp_per_ch'][shank_channels]
            unitfig_lfp.plot(whole_shank.T, color='k', alpha=0.1, linewidth=1)
            unitfig_lfp.plot(whole_shank[np.argmin(np.min(whole_shank, axis=1)),:], label='layer4', color='r')
        else:
            unitfig_lfp.plot(row['hf4_revchecker_revchecker_mean_resp_per_ch'].T, color='k', alpha=0.1, linewidth=1)
        unitfig_lfp.plot(row['hf4_revchecker_revchecker_mean_resp_per_ch'][row['ch']], label='this channel', color='b')
        try:
            unitfig_lfp.set_title('ch='+str(row['ch'])+'\npos='+str(row['hf4_revchecker_lfp_rel_depth'][(row['ch'])]))
            newdf['hf4_revchecker_ch_lfp_relative_depth'].iloc[index] = row['hf4_revchecker_lfp_rel_depth'][(row['ch'])]
        except KeyError:
            unitfig_lfp.set_title('ch='+str(row['ch']))
        unitfig_lfp.legend(); unitfig_lfp.axvline(x=(0.1*30000), color='k', linewidth=1)
        unitfig_lfp.set_xticks(np.arange(0,18000,18000/8))
        unitfig_lfp.set_xticklabels(np.arange(-100,500,75))
        unitfig_lfp.set_xlabel('msec'); unitfig_lfp.set_ylabel('uvolts')

        plt.tight_layout()

        pdf.savefig(unitfig)
        plt.close()

    print('saving unit summary pdf')
    pdf.close()

    print('saving an updated h5 ephys file')
    newdf.to_hdf(os.path.join(savepath, 'updated_ephys_props.h5'), 'w')

def make_session_summary(df, savepath):
    pdf = PdfPages(os.path.join(savepath, 'session_summary.pdf'))
    df['unit'] = df.index.values
    df = df.set_index('session')

    unique_inds = sorted(list(set(df.index.values)))

    for unique_ind in tqdm(unique_inds):
        uniquedf = df.loc[unique_ind]
        # set up subplots
        plt.subplots(3,4,figsize=(15,15))
        plt.suptitle(unique_ind+'eye fit: m='+str(uniquedf['best_ellipse_fit_m'].iloc[0])+' r='+str(uniquedf['best_ellipse_fit_r'].iloc[0]))
        # eye position vs head position
        try:
            plt.subplot(3,4,1)
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
            accT = uniquedf['fm1_accT'].iloc[0]
            roll_interp = uniquedf['fm1_roll_interp'].iloc[0]
            pitch_interp = uniquedf['fm1_pitch_interp'].iloc[0]
            th = uniquedf['fm1_theta'].iloc[0]
            phi = uniquedf['fm1_phi'].iloc[0]
            plt.subplot(3,4,2)
            plt.plot(pitch_interp[::100], th[::100], '.'); plt.xlabel('pitch'); plt.ylabel('theta')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
            plt.subplot(3,4,3)
            plt.plot(roll_interp[::100], phi[::100], '.'); plt.xlabel('roll'); plt.ylabel('phi')
            plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')
        except:
            pass
        try:
            # histogram of theta from -45 to 45deg (are eye movements in resonable range?)
            plt.subplot(3,4,4)
            plt.title('hist of FM theta')
            plt.hist(uniquedf['fm1_theta'].iloc[0], range=[-45,45])
            # histogram of phi from -45 to 45deg (are eye movements in resonable range?)
            plt.subplot(3,4,5)
            plt.title('hist of FM phi')
            plt.hist(uniquedf['fm1_phi'].iloc[0], range=[-45,45])
            # histogram of gyro z (resonable range?)
            plt.subplot(3,4,6)
            plt.title('hist of FM gyro z')
            plt.hist(uniquedf['fm1_gz'].iloc[0], range=[2,4])
            # plot of contrast response functions on same panel scaled to max 30sp/sec
            # plot of average contrast reponse function across units
            plt.subplot(3,4,7)
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
            plt.subplot(3,4,8)
            plt.title('FM spike raster')
            i = 0
            for ind, row in uniquedf.iterrows():
                plt.vlines(row['fm1_spikeT'],i-0.25,i+0.25)
                plt.xlim(0, 10); plt.xlabel('secs'); plt.ylabel('unit #')
                i = i+1
        except:
            pass
        # all psth plots in a single panel, with avg plotted over the top
        plt.subplot(3,4,9)
        lower = -0.5; upper = 1.5; dt = 0.1
        bins = np.arange(lower,upper+dt,dt)
        for ind, row in uniquedf.iterrows():
            plt.plot(bins[0:-1]+dt/2,row['hf3_gratings_grating_psth'])
        avg_psth = np.mean(uniquedf['hf3_gratings_grating_psth'], axis=1)
        plt.plot(bins[0:-1]+dt/2,avg_psth)
        plt.set_title('gratings psth'); plt.set_xlabel('time'); plt.set_ylabel('sp/sec')
        plt.set_ylim([0,np.nanmax(avg_psth)*1.5])
        
        pdf.savefig()
        plt.close()
    print('saving session summary pdf')
    pdf.close()