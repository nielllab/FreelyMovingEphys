%% Preprocess ephys data
%%% runs applyCARtoDat_subset on data with user params
fname = '101320_G6H28P6LT_hf3_wn_Ephys.bin';
outpath = 'T:\freely_moving_ephys\ephys_recordings\101320\G6H28P6LT\hf3_wn';
nchan = 32;
medfilt = 1;
subset = [9:24];
isuint16 = 1;

applyCARtoDat_subset(fname,nchan,outpath,medfilt,subset,isuint16);