"""
polished_figures.py
"""
import figurefirst as fifi
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

def make_fig1(parentpath, df1):
    layout = fifi.FigureLayout(os.path.join(parentpath,'fig1.svg'), make_mplfigures=True)
    matplotlib.rcParams.update({'font.size': 18})

    example128ch =  df1[df1['session']=='070921_J553RT_control_Rig2']
    start = 1
    rangeT = np.arange(start,start+(60*20))

    layout.axes['theta'].plot(example128ch['fm1_eyeT'].iloc[0][rangeT], example128ch['fm1_theta'].iloc[0][rangeT])
    layout.axes['theta'].get_xaxis().set_visible(False)
    layout.axes['theta'].set_xlim([0,20])
    layout.axes['theta'].set_ylabel('theta')

    layout.axes['phi'].plot(example128ch['fm1_eyeT'].iloc[0][rangeT], example128ch['fm1_phi'].iloc[0][rangeT])
    layout.axes['phi'].get_xaxis().set_visible(False)
    layout.axes['phi'].set_xlim([0,20])
    layout.axes['phi'].set_ylabel('phi')

    layout.axes['pitch'].plot(example128ch['fm1_eyeT'].iloc[0][rangeT], example128ch['fm1_pitch_interp'].iloc[0][rangeT])
    layout.axes['pitch'].get_xaxis().set_visible(False)
    layout.axes['pitch'].set_xlim([0,20])
    layout.axes['pitch'].set_ylabel('pitch')

    layout.axes['roll'].plot(example128ch['fm1_eyeT'].iloc[0][rangeT], example128ch['fm1_roll_interp'].iloc[0][rangeT])
    layout.axes['roll'].get_xaxis().set_visible(False)
    layout.axes['roll'].set_xlim([0,20])
    layout.axes['roll'].set_ylabel('roll')

    sh_num = 4
    sh0 = np.arange(0,len(example128ch.index)+sh_num,sh_num)
    full_raster = np.array([]).astype(int)
    for sh in range(sh_num):
        full_raster = np.concatenate([full_raster, sh0+sh])
    for i, ind in enumerate(example128ch.index):
        i = full_raster[i]
        layout.axes['raster'].vlines(example128ch.at[ind,'fm1_spikeT'],i-0.25,i+0.25,'k',linewidth=0.5)
    layout.axes['raster'].set_ylim(len(example128ch)+0.5,-.5)
    layout.axes['raster'].set_xlabel('secs')
    layout.axes['raster'].set_xlim([start,start+20])
    layout.axes['raster'].set_xticks(np.arange(1,21,2))
    layout.axes['raster'].set_xticklabels(np.arange(0,20,2))
    layout.axes['raster'].set_ylabel('unit')

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
    layout.axes['spikerates'].bar(x - width/2, np.nanmedian(exc_rates,a), yerr=np.nanstd(exc_rates,a)/np.sqrt(np.size(exc_rates,a)), color='b', width=width, label='exc')
    layout.axes['spikerates'].bar(x + width/2, np.nanmedian(inh_rates,a), yerr=np.nanstd(inh_rates,a)/np.sqrt(np.size(inh_rates,a)), color='g', width=width, label='inh')
    layout.axes['spikerates'].set_xticks(x)
    layout.axes['spikerates'].set_xticklabels(labels)
    layout.axes['spikerates'].set_ylim([0,20])
    layout.axes['spikerates'].legend()
    layout.axes['spikerates'].set_ylabel('sp/sec')

    for ind, row in df1['norm_waveform'][df1['waveform_km_label']==0].iteritems():
        layout.axes['waveforms'].plot(row, 'b', linewidth=1)
    for ind, row in df1['norm_waveform'][df1['waveform_km_label']==1].iteritems():
        layout.axes['waveforms'].plot(row, 'g', linewidth=1)
    bluepatch = mpatches.Patch(color='g', label='inhibitory')
    greenpatch = mpatches.Patch(color='b', label='excitatory')
    layout.axes['waveforms'].legend(handles=[bluepatch, greenpatch])
    layout.axes['waveforms'].set_ylabel('milivolts'); layout.axes['waveforms'].set_xlabel('msec')

    layout.axes['waveform_props'].plot(df1['waveform_trough_width'][df1['waveform_peak'] < 0][df1['waveform_km_label']==0], df1['AHP'][df1['waveform_peak'] < 0][df1['waveform_km_label']==0], 'g.')
    layout.axes['waveform_props'].plot(df1['waveform_trough_width'][df1['waveform_peak'] < 0][df1['waveform_km_label']==1], df1['AHP'][df1['waveform_peak'] < 0][df1['waveform_km_label']==1], 'b.')
    layout.axes['waveform_props'].set_ylabel('AHP'); layout.axes['waveform_props'].set_xlabel('trough width')

    layout.axes['depth'].hist(df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==1],color='b',bins=np.arange(-600,600,100),alpha=0.3,orientation='horizontal')
    layout.axes['depth'].hist(df1['hf1_wn_depth_from_layer5'][df1['waveform_km_label']==0],color='g',bins=np.arange(-600,600,100),alpha=0.3,orientation='horizontal')
    layout.axes['depth'].set_ylabel('um relative to layer 5'); layout.axes['depth'].set_xlabel('unit count'); layout.axes['depth'].invert_yaxis()
    layout.axes['depth'].plot([0,10],[0,0],'k')

    layout.save(os.path.join(parentpath, 'fig1_output.svg'))

if __name__ == '__main__':
    parentpath = '/home/niell_lab/Documents/figures/polished_figs'
    picklepath = '/home/niell_lab/data/freely_moving_ephys/batch_files/091321/pooled_ephys_population_update_091321.pickle'
    df1 = pd.read_pickle(picklepath)
    make_fig1(parentpath, df1)