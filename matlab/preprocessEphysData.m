% Preprocess ephys data
%% 64 ch version, runs applyCARtoDat_subset on data with user params

nchan = 64;
medfilt = 1;
subset = 1:64;
isuint16 = 1;

%%% default channel map
%chanMap = 1:64;

%%% NN H64-LP
% chanMap = 1 + [32 62 33 63 34 60 36 61 37 58 38 59 40 56 41 57 42 54 44 55 45 52 46 53 47 50 43 51 39 48 35 49 0 30 1 31 2 28 3 26 4 27 5 24 6 22 7 23 8 20 9 18 10 19 11 16 12 17 13 21 14 25 15 29];

%%% DB P64-3
chanMap = [1 3 5 7 9 11 13 15 16 14 12 10 8 6 4 2 17 19 21 23 25 27 29 31 32 30 28 26 24 22 20 18 48 46 44 42 40 38 36 34 33 35 37 39 41 43 45 47 64 62 60 58 56 54 52 50 49 51 53 55 57 59 61 63];

%%% DB P64-8
% chanMap = [12 1 10 3 8 5 6 7 4 9 2 11 17 13 19 15 21 16 23 14 25 27 29 31 32 30 28 26 24 22 20 18 37 48 39 46 41 44 43 42 45 40 47 38 64 36 62 34 60 33 58 35 56 54 52 50 49 51 53 55 57 59 61 63]

% merge datasets and output single bin file
applyCARtoDat_subset_multi(nchan,medfilt,subset,isuint16,chanMap);


%% 16-channel version

% nchan = 32;
% medfilt = 0;
% subset = 9:24;
% isuint16 = 1;
% 
%%%% NN 16ch map
% chanMap = [15 18 10 23 11 22 12 21 9 24 13 20 14 19 16 17] - 8;
% 
% applyCARtoDat_subset_multi(nchan,medfilt,subset,isuint16,chanMap);
