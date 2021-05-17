function [probe_remapping, nch, subset] = getProbeMap(probe_name, json_path)
%getProbeMap.m Get the sequence of recording sites for a probe, given the probe
%name and path to json storing probe names and their channel remappings
%   probe name should be str, name of probe
%   only options are probe maps listed in json, which is read in from the
%   provided file path, json_path

% set arguments for later analysis, number of channels and subset of channels
% searches for string 16 or 64 in probe name, which should always be there
if contains(probe_name, '16')
    nch = 16;
    subset = 9:24;
elseif contains(probe_name, '64')
    nch = 64;
    subset = 1:64;
else
    error('Could not find usable ch num using given probe name')
end

% read and decode json, get the struct of all channel maps
probe_maps = jsondecode(char(fread(fopen(json_path))'));

% get the map for the current probe
probe_remapping = probe_maps.(probe_name);