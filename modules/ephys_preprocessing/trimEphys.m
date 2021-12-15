function trimmedData = trimEphys(binaryPath, chNum, secToDrop, trimStart, trimEnd)
% trimEphys indexes off a section of the binary, specified in seconds, from
% either the start, end, or both the start and end of a recording
% INPUTS
% binaryPath: path to an ephys binary file
% chNum: number of channels in the binary
% secToDrop: how many seconds to discard
% trimStart: 1 if window should be trimmed from the start of the binary
% trimEnd: 1 if window should be trimmed from the end of the binary
% OUTPUTS
% trimmedData: ephys data for each channel across time, with samples within
% the range of secToDrop removed from either the start, end, or both
samprate = 30000;
openFile = fopen(binaryPath, 'r');
data = fread(openFile, [chNum Inf], '*uint16');
win = secToDrop * samprate;
if trimEnd == 1 && trimStart == 0
    trimmedData = data(:, 1:(end-win));
elseif trimStart == 1 && trimEnd == 0
    trimmedData = data(:, win:end);
elseif trimStart == 1 && trimEnd == 1
    trimmedData = data(:, win:(end-win));
elseif trimStart == 0 && trimEnd == 0
    trimmedData = data;
end
