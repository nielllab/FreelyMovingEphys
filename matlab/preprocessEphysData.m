%% Preprocess ephys data
%%% runs applyCARtoDat_subset on data with user params
fname = '101220_G6H28P6LT_hf1_wn_Ephys.bin';
outpath = 'T:\freely_moving_ephys\ephys_recordings\101220\G6H28P6LT\hf1_wn';
nchan = 32;
medfilt = 1;
subset = [9:24];
isuint16 = 1;

applyCARtoDat_subset(fname,nchan,outpath,medfilt,subset,isuint16);