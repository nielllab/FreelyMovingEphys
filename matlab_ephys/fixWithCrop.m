% fix recording by cropping off a section of an ephys binary file

% will rename the original recording to end in _full.bin and save the new
% file with the original name

%% set up options
% path to the binary to fix
binaryName = '/home/niell_lab/data/freely_moving_ephys/ephys_binary_files/070921_J553RT_control_Rig2_fm1_Ephys.bin';
% number of channels
chNum = 128;
% how many seconds to drop from the recording
secToDrop = 1;
% trim that amount of time from the start of the recording? true=1, false=0
% it is possible to trim both the start and end
trimStart = 0;
% trim from the end?
trimEnd = 1;

%% trim the ephys file
display('making a copy of file')
[pathstr, name, ext] = fileparts(binaryName);
newBinaryName = strcat(pathstr, '/', name, '_full', ext);
copyfile(binaryName, newBinaryName)
delete(binaryName)
%%
display('trimming ephys data')
trimmedData = trimEphys(newBinaryName, chNum, secToDrop, trimStart, trimEnd);
%%
display('saving')
out_fileID = fopen(binaryName, 'w');
fwrite(out_fileID, trimmedData, 'uint16');
fclose(out_fileID);
display('done')
