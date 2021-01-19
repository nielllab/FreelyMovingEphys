%% Preprocess ephys data
%%% runs applyCARtoDat_subset on data with user params
fname = '101320_G6H28P6LT_hf3_wn_Ephys.bin';
outpath = 'T:\freely_moving_ephys\ephys_recordings\101320\G6H28P6LT\hf3_wn';
nchan = 64;
medfilt = 1;
subset = 1:64;
isuint16 = 1;


chanMap = 1:64;
%%% channel map for 2x32 custom probe.
chanMap = 1 + [32 62 33 63 34 60 36 61 37 58 38 59 40 56 41 57 42 54 44 55 45 52 46 53 47 50 43 51 39 48 35 49 0 30 1 31 2 28 3 26 4 27 5 24 6 22 7 23 8 20 9 18 10 19 11 16 12 17 13 21 14 25 15 29];

%single dataset
%applyCARtoDat_subset(fname,nchan,outpath,medfilt,subset,isuint16);

%merge datasets
applyCARtoDat_subset_multi(nchan,medfilt,subset,isuint16, chanMap);