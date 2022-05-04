"""
summaryfigs.py
"""
import argparse, os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from projects.ephys.population import Population
from src.utils.path import find
from src.utils.auxiliary import flatten_series
from scipy.interpolate import interp1d
from scipy.stats import linregress
from src.utils.filter import convfilt
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hffm', type=str, default='/home/niell_lab/data/freely_moving_ephys/batch_files/021022/hffm')
    parser.add_argument('--ltdk', type=str, default='/home/niell_lab/data/freely_moving_ephys/batch_files/021022/ltdk')
    parser.add_argument('--savepath', type=str, default='/home/niell_lab/Desktop/redef_using_dHead/')
    args = parser.parse_args()
    return args

def open_data(hffm_savepath, ltdk_savepath):
    hffm = Population(savepath=hffm_savepath)
    hffm.load(fname='hffm_pop2'); hffm.exptype = 'hffm'

    ltdk = Population(savepath=ltdk_savepath)
    ltdk.load(fname='ltdk_pop2'); hffm.exptype = 'ltdk'

    return hffm, ltdk

def cropPSTH(x):
    model_dt = 0.025
    trange = np.arange(-1, 1.1, model_dt)
    trange_x = 0.5*(trange[0:-1]+ trange[1:])
    # crop PSTH for hf sparse noise to seperate out repeated stimulus presentations
    x[trange_x<0] = np.nan
    x[trange_x>.2] = np.nan
    return x

def multipanel_figs(savepath, hffm, ltdk):
    ...

def small_figs(savepath, hffm, ltdk):
        pdf = PdfPages(os.path.join(savepath, 'summaryfigs_'+datetime.today().strftime('%m%d%y')+'.pdf'))

        # example neuron eye movement PSTH
        ax_ex_psth = plt.subplot(1,2,1)
        ex_unit = 474
        right = hffm.data['FmLt_rightsacc_avg_gaze_shift_dEye'][hffm.data['session']=='102621_J558NC_control_Rig2'][hffm.data['index']==ex_unit].iloc[0]
        left = hffm.data['FmLt_leftsacc_avg_gaze_shift_dEye'][hffm.data['session']=='102621_J558NC_control_Rig2'][hffm.data['index']==ex_unit].iloc[0]
        ax_ex_psth.plot(hffm.trange_x, right, color=hffm.cmap_special2[1])
        ax_ex_psth.plot(hffm.trange_x, left, color='dimgray', linestyle='dashed')
        maxval = np.max(np.maximum(right, left))*1.2
        ax_ex_psth.set_ylim([0, maxval])
        ax_ex_psth.set_xlim([-0.2, 0.4])
        ax_ex_psth.set_ylabel('sp/sec')
        ax_ex_psth.set_xlabel('sec')
        ax_ex_psth.annotate('right', xy=[0.18,5], color=hffm.cmap_special2[1], fontsize=16)
        ax_ex_psth.annotate('left', xy=[0.18,4.2], color='dimgray', fontsize=16)
        ax_all_norm = plt.subplot(1,2,2)
        pref = hffm.data['pref_gazeshift_psth'][hffm.data['session']=='102621_J558NC_control_Rig2'][hffm.data['index']==ex_unit].iloc[0]
        allpsth = hffm.data['pref_gazeshift_psth'][hffm.data['sacccluster_no_movement'].isin(['early','late','biphasic','negative','unresponsive'])]
        for ind, psth in allpsth.iteritems():
            ax_all_norm.plot(hffm.trange_x, psth, linewidth=1, alpha=0.1)
        ax_all_norm.set_ylim([-1,1]); ax_all_norm.set_xlim([-0.2,0.4])
        ax_all_norm.plot(hffm.trange_x, pref, color=hffm.cmap_special2[1])
        ax_all_norm.plot(hffm.trange_x, np.mean(flatten_series(allpsth),0), 'k', linewidth=4)
        ax_all_norm.set_ylabel('norm. spike rate')
        ax_all_norm.set_xlabel('sec')
        plt.tight_layout(); pdf.savefig(); plt.close()

        # saccade clusters (WITHOUT the movement cluster)
        fig = plt.figure(constrained_layout=True, figsize=(6,6))
        spec = gridspec.GridSpec(nrows=4, ncols=1, figure=fig)

        ax_cluster_sz = fig.add_subplot(spec[0,0])
        ax_four_clusters = fig.add_subplot(spec[1:,0])

        vcounts = hffm.data['sacccluster_no_movement'].value_counts()
        names = ['early','late','biphasic','negative','unresponsive']
        for i, name in enumerate(names):
            ax_cluster_sz.bar(i, vcounts[name], color=hffm.cmap_movclusts[name])
        ax_cluster_sz.set_xticks(ticks=range(5), labels=['early','late','biphasic','neg.','unresp.'])
        ax_cluster_sz.set_ylabel('# neurons')

        for count, name in enumerate(names):
            cluster_psths = flatten_series(hffm.data['pref_gazeshift_psth'][hffm.data['sacccluster_no_movement']==name])
            clustmean = np.mean(cluster_psths, 0)
            clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
            ax_four_clusters.plot(hffm.trange_x, clustmean, '-', linewidth=2, color=hffm.cmap_movclusts[name])
            ax_four_clusters.fill_between(hffm.trange_x, clustmean-clusterr, clustmean+clusterr, color=hffm.cmap_movclusts[name], alpha=0.3)
        ax_four_clusters.set_xlim([-0.1,0.3]); ax_four_clusters.set_ylim([-.5,.6])
        # ax_four_clusters.annotate('movement', xy=[-0.09,0.55], color=hffm.cmap_movclusts['movement'], fontsize=15)
        ax_four_clusters.annotate('early', xy=[-0.09,0.50], color=hffm.cmap_movclusts['early'], fontsize=15)
        ax_four_clusters.annotate('late', xy=[-0.09,0.45], color=hffm.cmap_movclusts['late'], fontsize=15)
        ax_four_clusters.annotate('biphasic', xy=[-0.09,0.40], color=hffm.cmap_movclusts['biphasic'], fontsize=15)
        ax_four_clusters.annotate('negative', xy=[-0.09,0.35], color=hffm.cmap_movclusts['negative'], fontsize=15)
        ax_four_clusters.annotate('unresponsive', xy=[-0.09,0.30], color=hffm.cmap_movclusts['unresponsive'], fontsize=15)
        ax_four_clusters.set_ylabel('norm. spike rate'); ax_four_clusters.set_xlabel('sec')
        pdf.savefig(); plt.close()

        # saccade clusters (WITH the movement cluster)
        fig = plt.figure(constrained_layout=True, figsize=(6,6))
        spec = gridspec.GridSpec(nrows=4, ncols=1, figure=fig)
        ax_cluster_sz = fig.add_subplot(spec[0,0])
        ax_four_clusters = fig.add_subplot(spec[1:,0])
        vcounts = hffm.data['sacccluster'].value_counts()
        names = ['movement','early','late','biphasic','negative','unresponsive']
        for i, name in enumerate(names):
            ax_cluster_sz.bar(i, vcounts[name], color=hffm.cmap_movclusts[name])
        ax_cluster_sz.set_xticks(ticks=range(6), labels=['mov.','early','late','biphasic','neg.','unresp.'])
        ax_cluster_sz.set_ylabel('# neurons')
        for count, name in enumerate(names):
            cluster_psths = flatten_series(hffm.data['pref_gazeshift_psth'][hffm.data['sacccluster']==name])
            clustmean = np.mean(cluster_psths, 0)
            clusterr = np.std(cluster_psths, 0) / np.sqrt(np.size(cluster_psths,0))
            ax_four_clusters.plot(hffm.trange_x, clustmean, '-', linewidth=2, color=hffm.cmap_movclusts[name])
            ax_four_clusters.fill_between(hffm.trange_x, clustmean-clusterr, clustmean+clusterr, color=hffm.cmap_movclusts[name], alpha=0.3)
        ax_four_clusters.set_xlim([-0.1,0.3]); ax_four_clusters.set_ylim([-.5,.6])
        ax_four_clusters.annotate('movement', xy=[-0.09,0.55], color=hffm.cmap_movclusts['movement'], fontsize=15)
        ax_four_clusters.annotate('early', xy=[-0.09,0.50], color=hffm.cmap_movclusts['early'], fontsize=15)
        ax_four_clusters.annotate('late', xy=[-0.09,0.45], color=hffm.cmap_movclusts['late'], fontsize=15)
        ax_four_clusters.annotate('biphasic', xy=[-0.09,0.40], color=hffm.cmap_movclusts['biphasic'], fontsize=15)
        ax_four_clusters.annotate('negative', xy=[-0.09,0.35], color=hffm.cmap_movclusts['negative'], fontsize=15)
        ax_four_clusters.annotate('unresponsive', xy=[-0.09,0.30], color=hffm.cmap_movclusts['unresponsive'], fontsize=15)
        ax_four_clusters.set_ylabel('norm. spike rate'); ax_four_clusters.set_xlabel('sec')
        pdf.savefig(); plt.close()

        # laminar depth
        fig = plt.figure(constrained_layout=True, figsize=(15,5))
        spec = gridspec.GridSpec(nrows=1, ncols=7, figure=fig)

        ax_ex_depth = fig.add_subplot(spec[0,0])
        ax_movement_depth = fig.add_subplot(spec[0,1])
        ax_early_depth = fig.add_subplot(spec[0,2])
        ax_late_depth = fig.add_subplot(spec[0,3])
        ax_biphasic_depth = fig.add_subplot(spec[0,4])
        ax_negative_depth = fig.add_subplot(spec[0,5])
        ax_unresp_depth = fig.add_subplot(spec[0,6])

        mua_power = hffm.data['Wn_lfp_power'][hffm.data['session']=='101521_J559NC_control_Rig2'].iloc[0]
        layer5 = hffm.data['Wn_layer5cent_from_lfp'][hffm.data['session']=='101521_J559NC_control_Rig2'].iloc[0]
        ch_spacing = 25
        for sh in range(4):
            ax_ex_depth.plot(mua_power[sh], np.arange(0,32)-layer5[sh], 'tab:red')
        # ax_ex_depth.set_title('example recording depth')
        ax_ex_depth.hlines(0,np.min(mua_power),np.max(mua_power), 'k', linestyle='dashed')
        ax_ex_depth.set_ylim([18,-19])
        ax_ex_depth.set_yticks(ticks=np.arange(18,-19,-3), labels=(ch_spacing*np.arange(18,-19,-3)))
        ax_ex_depth.set_ylabel('depth (um)'); ax_ex_depth.set_xlabel('LFP MUA power')
        ax_ex_depth.annotate('layer 5', xy=[0.75, -.5], color='k', fontsize=12)

        panels = [ax_movement_depth, ax_early_depth, ax_late_depth, ax_biphasic_depth, ax_negative_depth, ax_unresp_depth]
        names = ['movement','early','late','biphasic','negative','unresponsive']
        for i, panel in enumerate(panels):
            name = names[i]
        #     panel.set_title(name)
            panel.hist(hffm.data['Wn_depth_from_layer5'], color='k', bins=np.linspace(-650,750,8), orientation='horizontal', density=True, histtype='step', linewidth=2)
            panel.hist(hffm.data['Wn_depth_from_layer5'][hffm.data['sacccluster']==name],
                                color=hffm.cmap_movclusts[name], bins=np.linspace(-650,750,8), orientation='horizontal', density=True, histtype='stepfilled')
            panel.set_ylabel('depth (um)')
            panel.set_xlabel('fraction of neurons')
            panel.set_title(name)
            panel.invert_yaxis()
            panel.set_xlim(0,0.003)
        pdf.savefig(); plt.close()


        # cluster PSTHs, also showing non-pref direction
        fig = plt.figure(constrained_layout=True, figsize=(15,5))
        spec = gridspec.GridSpec(nrows=2, ncols=6, figure=fig)

        ax_movement = fig.add_subplot(spec[0,0])
        ax_early = fig.add_subplot(spec[0,1])
        ax_late = fig.add_subplot(spec[0,2])
        ax_biphasic = fig.add_subplot(spec[0,3])
        ax_negative = fig.add_subplot(spec[0,4])
        ax_unresp = fig.add_subplot(spec[0,5])

        ax_movement_pnp = fig.add_subplot(spec[1,0])
        ax_early_pnp = fig.add_subplot(spec[1,1])
        ax_late_pnp = fig.add_subplot(spec[1,2])
        ax_biphasic_pnp = fig.add_subplot(spec[1,3])
        ax_negative_pnp = fig.add_subplot(spec[1,4])
        ax_unresp_pnp = fig.add_subplot(spec[1,5])

        panels = [ax_movement, ax_early, ax_late, ax_biphasic, ax_negative, ax_unresp]
        pnp_panels = [ax_movement_pnp, ax_early_pnp, ax_late_pnp, ax_biphasic_pnp, ax_negative_pnp, ax_unresp_pnp]
        movtypes = ['movement','early','late','biphasic','negative','unresponsive']
        for count in range(6):
            panel = panels[count]
            movtype = movtypes[count]
            pnp_panel = pnp_panels[count]
            
            thisclust = hffm.data['pref_gazeshift_psth'][hffm.data['movcluster1']==movtype]
            for i, psth in enumerate(thisclust):
                panel.plot(hffm.trange_x, psth, '-', linewidth=1, alpha=0.1)
            clustmean = np.nanmean(flatten_series(thisclust),0)
            panel.plot(hffm.trange_x, clustmean, '-', linewidth=4, color=hffm.cmap_movclusts[movtype])
            panel.set_xlim([-0.2,0.4]); panel.set_ylim([-.8,.8])
            panel.set_title(movtype)
            panel.set_ylabel('norm. spike rate'); panel.set_xlabel('sec')
            # panel.vlines(0,-.8,.8, colors='k', linestyle='dotted', alpha=0.3)
            
            gaze_pref = np.nanmean(flatten_series(hffm.data['pref_gazeshift_psth'][hffm.data['movcluster1']==movtype]),0)
            gaze_nonpref = np.nanmean(flatten_series(hffm.data['nonpref_gazeshift_psth'][hffm.data['movcluster1']==movtype]),0)
            comp_pref = np.nanmean(flatten_series(hffm.data['pref_comp_psth'][hffm.data['movcluster1']==movtype]),0)
            comp_nonpref = np.nanmean(flatten_series(hffm.data['nonpref_comp_psth'][hffm.data['movcluster1']==movtype]),0)
            
            pnp_panel.plot(hffm.trange_x, gaze_pref, '-', linewidth=2, color='tab:blue')
            pnp_panel.plot(hffm.trange_x, gaze_nonpref, '-', linewidth=2, color='lightblue')
            pnp_panel.plot(hffm.trange_x, comp_pref, '-', linewidth=2, color='tab:red')
            pnp_panel.plot(hffm.trange_x, comp_nonpref, '-', linewidth=2, color='lightcoral')
            pnp_panel.set_xlim([-0.2,0.4]); pnp_panel.set_ylim([-.5,.5])
            pnp_panel.set_ylabel('norm. spike rate'); pnp_panel.set_xlabel('sec')
        pdf.savefig(); plt.close()

def main(args):
    hffm, ltdk = open_data(args.hffm, args.ltdk)
    small_figs(args.savepath, hffm, ltdk)
    multipanel_figs(args.savepath, hffm, ltdk)

if __name__ == '__main__':
    args = get_args()
    main(args)