%% Preprocess ephys data
%%% runs applyCARtoDat_subset on data with user params
% nchan = 64;
% medfilt = 1;
% subset = 1:64;
% isuint16 = 1;
% 
% 
% % % % chanMap = 1:64;
% %%% channel map for 2x32 custom probe.
% chanMap = 1 + [32 62 33 63 34 60 36 61 37 58 38 59 40 56 41 57 42 54 44 55 45 52 46 53 47 50 43 51 39 48 35 49 0 30 1 31 2 28 3 26 4 27 5 24 6 22 7 23 8 20 9 18 10 19 11 16 12 17 13 21 14 25 15 29];
% 
% % %single dataset
% % applyCARtoDat_subset(nchan,outpath,medfilt,subset,isuint16,chanMap);
% 
% %merge datasets
% applyCARtoDat_subset_multi(nchan,medfilt,subset,isuint16,chanMap);


%% 16-channel version

nchan = 32;
medfilt = 1;
subset = 9:24;
isuint16 = 1;

chanMap = [15 18 10 23 11 22 12 21 9 24 13 20 14 19 16 17];

applyCARtoDat_subset_multi(nchan,medfilt,subset,isuint16,chanMap);
