function PreprocessEphys(probe, doMedFilt, isuint16)
% PreprocessEphys
%
%   This is an alternative to the GUI, `PreprocessEphysGUI.mlapp`
%   
%   You can run this function without any arguments. Defaults will be
%   filled in, and dialog boxes will be opened for required information and
%   file paths.
%
%   Once this function is run (assuming no arguments were given as
%   arguments:
%       1. Select the probe name from the dropdown menu.
%       2. Select recording .bin files.
%           A window will open to select a .bin files for recordings in
%           chronological order.
%           Each time a recording file is opened, the window will reopen to
%           allow you to select another. Once you have selected all the
%           recordings and the window reopens, hit 'cancel' to close it.
%       3. Select the save path for the merged .bin file.
%
%
%   Input
%       
%       probe       :  Name of the probe to use. This is optional, and a
%                      dialog box will open to choose from options in the
%                      repo .json.
%                      Options: [ default16, NN_H16, default64, NN_H64_LP,
%                              DB_P64_3, DB_P64_8, DB_P128_6, DB_P64_10_D ]
%                      The 'default16' and 'default64' options are ordered
%                      sequences (i.e., no remaping done). All remappings
%                      should be 1 (not 0) referenced.
%       doMedFilt   :  Should the data be median filtered? (1=True, 0=False).
%                      Default = 1.
%       isuint16    :  Is the raw data of the type uint16? If so, it will
%                      be converted to int16. (1=True, 0=False). Default = 1.
%
%
% Niell lab - FreelyMovingEphys
% Written by DMM, 2020
%

if ~exist('doMedFilt', 'var')
    doMedFilt = 1;
end

if ~exist('isuint16', 'var')
    isuint16 = 1;
end

% Get a list of names of probes
if ~exist('probe', 'var')

    names = listProbes();

    fig = uifigure(Name='Select probe');
    dd = uidropdown(fig);
    dd.Items = names;
    dd.Value = '';
    
    if dd.ValueChangedFcn.Edited == 1
        probe = @dd.Value;
    end
end

% Get required probe parameters from the name. This expects the matlab
% script to be located in the FreelyMovingEphys repo and will look for a
% .json file in the path /FreelyMovingEphys/fmEphys/utils/probes.json
[chanMap, nCh, subset] = getProbeMap(probe);

% Median filter and merge datasets as a single .bin file.
applyMedianFilt(nCh, doMedFilt, subset, isuint16, chanMap);

end