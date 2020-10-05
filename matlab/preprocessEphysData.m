%% Preprocess ephys data
%%% runs applyCARtoDat_subset on data with user params
fname = '093020_J524RT_wn_hf1_Ephys.bin';
outpath = '\\new-monster\t\freely_moving_ephys\ephys_recordings\093020\J524RT\wn_hf1';
nchan = 32;
medfilt = 1;
subset = [9:24];
isuint16 = 1;

applyCARtoDat_subset(fname,nchan,outpath,medfilt,subset,isuint16);