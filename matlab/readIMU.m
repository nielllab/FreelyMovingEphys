function [allData medianTrace] = readIMU(nChans, imuFile)
% Subtracts median of each channel, then subtracts median of each time
% point. Also, can select a subset of channels first (so don't include
% noise in median filter)
%
% this version will merge multiple recordings into one .bin for kilosort
% and creates a corresponding .mat file that will allow separation
%
% based on cortex repository by N. Steinmetz
% edited by cmn 2020
%
% should make chunk size as big as possible so that the medians of the
% channels differ little from chunk to chunk.
%
% doMedian = option to subtract medians (1) or not (0) - latter is if you
% only want to subset
% subChans = subset of channels to includie in output
% isUint16 = raw data is uint16,so convert to int16
% returns processed traces (allData) and CAR median (medianTrace)
%
% gui will ask for .bin files to merge. 'cancel' when done'
% next, will ask for .bin output file to save merged data into

if ~exist('nChans','var') | isempty(nChans)
    nChans = 8;
end

if ~exist('imuFile','var') | isempty(imuFile)
    [f p] = uigetfile('*.bin');
    imuFile = fullfile(p,f)
end

chunkSize = 1000000;
done=0;
nf = 0; %%% number of files

outputFilename = [imuFile(1:end-4) '.mat'];
try
    fid = fopen(imuFile, 'r');
    

    allData = fread(fid, [nChans Inf], 'uint16');  
    allData = 10 * (double(allData)/(2^16) - 0.5); % convert to 0-5V
    
    %%% plot trace of each channel
    samprate = 30000;
    figure
    for i = 1:nChans
        subplot(nChans,1,i)
        plot((1:length(allData(i,1:30:end)))*30/samprate, allData(i,1:30:end));
        if i==1
            title(imuFile)
        end
      %  xlim([0 30])
        ylim([0 5])
    end
    xlabel('secs'); ylabel('volts')
   % savefig([imuFile(1:end-4) '_IMU_fig1'])
    
    %%% bar plot of stdev for each channel (noise measure)
    figure
    bar(std(double(allData),[],2));
    xlabel('chan'); ylabel('stdev')
    title(imuFile)
   % savefig([imuFile(1:end-4) '_IMU_fig2'])
    
    fclose(fid);
    
    save(outputFilename,'allData','-v7.3');
    
catch me

    rethrow(me)
    
end
