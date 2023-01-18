function [map, nCh, subset] = getProbeMap(probe_name)
% getProbeMap  Get the sequence of recording sites for a linear probe.
% 
%   You can run this without giving the path to a .json file (`json_path`).
%   Without this, the function will look for that file in the expected
%   path in the FreelyMovingEphys repository.
%
%   The .json file should be readable as a cell array with the
%   dictionary-like structure:
%
%       {
%           name : {
%               map : [1,2,3,4,5]
%               nCh: 16
%               channel_spacing : 25    % microns
%            }
%       }
%
% `subset` is the subset of channels to include for applyCARtoDat
%
% probe_remapping, nCh, subset
%
% and path to json storing probe names and their channel remappings
%   probe name should be str, name of probe
%   only options are probe maps listed in json, which is read in from the
%   provided file path, json_path
%
% set arguments for later analysis, number of channels and subset of channels
% searches for string 16 or 64 in probe name, which should always be there
%
%   Inputs
%
%       probe_name  :  
%
% Niell lab - FreelyMovingEphys
% Written by DMM, Apr 2021
% Modified Nov 2022
%

% Use expected yml path, if one was not given.
%if ~exist('json_path', 'var')
filePath = mfilename('fullpath');
pathparts = strsplit(filePath,filesep);
repoPath = strjoin(pathparts(1:end-3),filesep);
json_path = [repoPath,filesep,'fmEphys',filesep,'utils',filesep,'probes.json'];
%end

if ~isfile(json_path)
    addpath(genpath(json_path));
end

% Read and decode yml, get the struct of all channel maps
allProbeData = readJSON(json_path);

% Get info for the current probe
probeData = allProbeData.(probe_name);
map = probeData.map;
nCh = probeData.nCh;

% Choose the channels to subset
if nCh == 16
    subset = 9:24;
elseif nCh == 64
    subset = 1:64;
elseif nCh == 128
    subset = 1:128;
end

end
