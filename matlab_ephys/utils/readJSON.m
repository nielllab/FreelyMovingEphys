function [data] = readJSON(json_path)
% readJSON Read .json file and format data as a string array.
%
% Niell lab - FreelyMovingEphys
% Written by DMM, Nov 2022
%
f = fopen(json_path);
data = jsondecode(char(fread(f)'));

end