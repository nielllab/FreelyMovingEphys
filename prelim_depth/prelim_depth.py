"""
prelim_depth.py
"""
from glob import glob
import os, platform
import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils.ephys import read_ephys_bin, butter_bandpass

def main(binary_path, probe_type):
    print('opening pdf')
    recording_directory, _ = os.path.split(binary_path)
    pdf = PdfPages(os.path.join(recording_directory, 'prelim_depth.pdf'))
    print('starting continuous LFP laminar depth estimation')
    print('loading ephys binary file')
    # read in ephys binary
    if platform.system() == 'Linux':
        mapping_json = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/probes/channel_maps.json'
    else:
        mapping_json = '/'.join(os.path.abspath(__file__).split('\\')[:-2]) + '/probes/channel_maps.json'
    lfp_ephys = read_ephys_bin(binary_path, probe_type, do_remap=True, mapping_json=mapping_json)
    print('applying bandpass filter')
    # subtract mean in time dim and apply bandpass filter
    ephys_center_sub = lfp_ephys - np.mean(lfp_ephys,0)
    filt_ephys = butter_bandpass(ephys_center_sub, lowcut=600, highcut=6000, fs=30000, order=6)
    print('getting lfp power profile across channels')
    # get lfp power profile for each channel
    ch_num = np.size(filt_ephys,1)
    lfp_power_profiles = np.zeros([ch_num])
    for ch in range(ch_num):
        lfp_power_profiles[ch] = np.sqrt(np.mean(filt_ephys[:,ch]**2))
    # median filter
    print('applying median filter')
    lfp_power_profiles_filt = medfilt(lfp_power_profiles)
    if probe_type == 'DB_P64-8':
        ch_spacing = 25/2
    else:
        ch_spacing = 25
    print('making figures')
    if ch_num == 64:
        norm_profile_sh0 = lfp_power_profiles_filt[:32]/np.max(lfp_power_profiles_filt[:32])
        layer5_cent_sh0 = np.argmax(norm_profile_sh0)
        norm_profile_sh1 = lfp_power_profiles_filt[32:64]/np.max(lfp_power_profiles_filt[32:64])
        layer5_cent_sh1 = np.argmax(norm_profile_sh1)
        plt.subplots(1,2, figsize=(10,8))
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
        pdf.savefig(); plt.close()
    elif ch_num == 16:
        norm_profile_sh0 = lfp_power_profiles_filt[:16]/np.max(lfp_power_profiles_filt[:16])
        layer5_cent_sh0 = np.argmax(norm_profile_sh0)
        plt.figure()
        plt.plot(norm_profile_sh0,range(0,16))
        plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
        plt.ylim([17,-1]); plt.yticks(ticks=list(range(-1,17)),labels=(ch_spacing*np.arange(18)-(layer5_cent_sh0*ch_spacing)))
        plt.title('shank0')
        pdf.savefig(); plt.close()
    elif ch_num == 128:
        norm_profile_sh0 = lfp_power_profiles_filt[:32]/np.max(lfp_power_profiles_filt[:32])
        layer5_cent_sh0 = np.argmax(norm_profile_sh0)
        norm_profile_sh1 = lfp_power_profiles_filt[32:64]/np.max(lfp_power_profiles_filt[32:64])
        layer5_cent_sh1 = np.argmax(norm_profile_sh1)
        norm_profile_sh2 = lfp_power_profiles_filt[64:96]/np.max(lfp_power_profiles_filt[64:96])
        layer5_cent_sh2 = np.argmax(norm_profile_sh2)
        norm_profile_sh3 = lfp_power_profiles_filt[96:128]/np.max(lfp_power_profiles_filt[96:128])
        layer5_cent_sh3 = np.argmax(norm_profile_sh3)
        plt.subplots(1,4, figsize=(20,8))
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
        pdf.savefig(); plt.close()
    print('closing pdf')
    pdf.close()