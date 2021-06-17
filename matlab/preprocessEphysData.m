%% Preprocess ephys data

medfilt = 1;
isuint16 = 1;
% json_path should to .json file with mappings of each probe i.e. {'probe':[1,2,3]}
json_path = 'C:/Users/Niell Lab/Documents/GitHub/FreelyMovingEphys/matlab/channel_maps.json';
% options for probe name are: [default16, NN_H16, default64, NN_H64-LP, DB_P64-3, DB_P64-8]
% default16 and default64 are ordered sequences (i.e. no remaping done)
% all remappings should be 1 (not 0) referenced
probe = 'DB_P64_3';

% get required params for applyCARtoDat
[chanMap, nchan, subset] = getProbeMap(probe, json_path);

% merge datasets and output single bin file
applyCARtoDat_subset_multi(nchan,medfilt,subset,isuint16,chanMap);