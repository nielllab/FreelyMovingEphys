% When data are moved, the metadata .mat for that ephys session will still
% point python code (to split ephys data back into individual recordings) to
% the old path.
% This script reads in a .mat and updates the paths for each directory for
% the new location of the data. It'll use the current path of the .mat as
% the new directory for the recording and update the cell array inside of
% that mat to reflect the new location.
% The old .mat will NOT be overwritten. Read in the .mat ending in
% '_update.mat' when splitting the recordings back out.

existingMatFile = '\\goeppert\nlab-nas\Dylan\freely_moving_ephys\ephys_recordings\100821\J559TT\100821_J559TT_dark_light.mat';

loadmat(existingMatFile);
[newpath, oldmatname, ext] = fileparts(existingMatFile);

newPathList = {};
newMatFile = fullfile(newpath, append(oldmatname, '_update', ext));

for i = 1:size(pathList,2)
    split_fname = split(string(pathList(i)),'\'); % split file path
    split_fname(cellfun('isempty',split_fname)) = []; % drop empty elements
    newPathList(i) = fullfile(newpath, split_fname(end)); % add .mat path to the existing directory name
end

pathList = cellstr(newPathList); % overwrite the old paths

save(newMatFile, 'doMedian', 'subChans', 'fileList', 'pathList', 'nSamps');