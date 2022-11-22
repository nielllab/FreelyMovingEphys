function [names] = listProbes(probe_json_path)
% listProbes  Return a list of possible ephys probes.
%
%   If the argument `probe_json_path` is given, the file will be read in
%   and keys will be listed, assumed to be the name of acceptable ephys
%   probes. Without this argument, the function will look for it in the
%   expected filepath in this repository:
%       /FreelyMovingEphys/fmEphys/utils/probes.json
%   To add new options for ephys preprocessing, add keys to the .json file
%   in the repository.
%   
%   Outputs a 1D array of probe names.
%
% Niell lab - Freely Moving Ephys
% Written by DMM, Nov 2022
%

if ~exist('probe_json_path', 'var')

    filePath = matlab.desktop.editor.getActiveFilename;
    pathparts = strsplit(filePath,filesep);
    repoPath = strjoin(pathparts(1:end-3),filesep);
    probe_json_path = [repoPath,filesep,'fmEphys',filesep,'utils',filesep,'probes.json'];

end

% Add the .json to the Matlab path, if it is not already added.
if ~isfile(probe_json_path)
    addpath(genpath(probe_json_path));
end

% Read the probe data
probeData = readJSON(probe_json_path);

% Get the probe names as a cell array
names1 = fieldnames(probeData);


% Convert to a 1D array
names = strings(1,size(names1,1));
for n = 1:size(names1,1)
    names(n) = cell2mat(names1(n,:));
end

end