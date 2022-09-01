%% open ephys binary and check length

full_binary = '/home/niell_lab/data/freely_moving_ephys/ephys_binary_files/070921_J553RT_control_Rig2_fm1_Ephys_full.bin';
part_binary = '/home/niell_lab/data/freely_moving_ephys/ephys_binary_files/070921_J553RT_control_Rig2_fm1_Ephys.bin';
%%
openFile = fopen(full_binary, 'r');
data = fread(openFile, [chNum Inf], '*uint16');
display(size(data))

%%
openFile = fopen(part_binary, 'r');
data = fread(openFile, [chNum Inf], '*uint16');
display(size(data))
