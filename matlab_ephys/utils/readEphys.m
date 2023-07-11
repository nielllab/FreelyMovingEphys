function ephysData = readEphys(json_path)
% readEphys
% Read ephys .json file and correct the
% offset and drift in timing.
%
% Written by DMM, July 2023
%

% Drift and offset values
drift_rate = -0.000114; % sec/sample
offset_val = 0.1; % sec

% Read .json file.
ephysData = readJSON(json_path);

% Select only the good cells (based on spike sorting labels).
useInds = find(struct2cell(ephysData.group)=="good");

% Keep copy of raw spikes.
ephysData.spikeT_raw = ephysData.spikeT;
% Overwrite the original spikes so that we only do timing correction for
% good cells.
ephysData = rmfield(ephysData, "spikeT");
ephysData.spikeT = struct;

% Apply timing correction
for i = 1:size(useInds, 1)
    
    % Index with unit number (e.g., unit 7 is indexed as "x7")
    unitNum = "x"+useInds(i);

    % sp = sp - (offset + sp * drift)
    ephysData.spikeT.(unitNum) = ephysData.spikeT_raw.(unitNum) -     ...
               (offset_val + ephysData.spikeT_raw.(unitNum) * drift_rate);

end

% Add the list of 'good' cells to the returned struct
ephysData.useInds = useInds;

end