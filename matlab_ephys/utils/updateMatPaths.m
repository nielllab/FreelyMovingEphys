function updateMatPaths(matpath)
% updateMatPaths   Update paths in a preprocessing .mat parameter file.
%
%   When data are moved, the metadata .mat for that ephys session will
%   still point python code (to split ephys data back into individual
%   recordings) to the old path.
%   
%   This script reads in a .mat and updates the paths for each directory
%   for the new location of the data. It'll use the current path of the
%   .mat as the new directory for the recording and update the cell array
%   inside of that mat to reflect the new location.
%   
%   The old .mat will be overwritten.
%
%
% Niell lab - FreelyMovingEphys
% Written by DMM, Feb 2022
%

if ~exist('matpath', 'var')
    [tmp_path, tmp_file] = matpath = uigetfile();
    matpath = join([tmp_file, tmp_path], '\');
end 

display(matpath);
load(matpath);
[newpath, ~, ~] = fileparts(matpath);

newPathList = strings(1,size(pathList,2));
for i = 1:size(pathList,2)
    split_fname = split(string(pathList(i)),'\'); % split file path
    split_fname(cellfun('isempty',split_fname)) = []; % drop empty elements
    newPathList(i) = fullfile(newpath, split_fname(end)); % add .mat path to the existing directory name
end

pathList = cellstr(newPathList); % overwrite the old paths

save(matpath, 'doMedian', 'subChans', 'fileList', 'pathList', 'nSamps');

end