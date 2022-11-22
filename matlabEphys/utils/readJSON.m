function [data] = readJSON(json_path)
% readJSON Read .json file and format data as a string array.
%
% Niell lab - FreelyMovingEphys
% Written by DMM, Nov 2022
%

data = jsondecode(char(fread(fopen(json_path))'));

end