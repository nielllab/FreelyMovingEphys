function [allData] = readIMU(nChans, imuFile, sampFreq, downSamp)
% reads IMU data as recorded from openEphys ADC in Bonsai

% input:
%nChans = # of channels (this must be correct or data will not be read in properly (default = 8)
%imuFile = .bin file containing data
%sampFreq = sampling frequency of ADC
% downSamp = fraction to downsample by

%output:
%allData = [nchans x nsamps] voltages (converted form uint16 to -5 to 5V)
% also saves allData into .mat file

% cmn 2020


if ~exist('nChans','var') | isempty(nChans)
    nChans = 8;
end

if ~exist('imuFile','var') | isempty(imuFile)
    [f p] = uigetfile('*.bin');
    imuFile = fullfile(p,f)
end

if ~exist('sampFreq','var') | isempty(sampFreq)
    sampFreq = 30000;
end

if ~exist('downSamp','var') | isempty(downSamp)
    downSamp = 10;
end

outputFilename = [imuFile(1:end-4) '.mat'];

try    
    % open file and read data
    fid = fopen(imuFile, 'r');
    allData = fread(fid, [nChans Inf], 'uint16');
    fclose(fid);
    
    % convert to -5 to 5V
    allData = 10 * (double(allData)/(2^16) - 0.5);
    
    % downsample data
    allData = allData(:,1:downSamp:end);
    sampFreq = sampFreq/downSamp;
    
    %%% plot trace of each channel
    figure
    for i = 1:nChans
        subplot(nChans,1,i)
        plot((1:length(allData(i,1:30:end)))*30/sampFreq, allData(i,1:30:end));
        if i==1
            title(imuFile)
        end
        %  xlim([0 30])
        ylim([0 5])
    end
    xlabel('secs'); ylabel('volts')
    % savefig([imuFile(1:end-4) '_IMU_fig1']) % not saving for some reason
    
    %%% bar plot of stdev for each channel (noise measure)
    figure
    bar(std(double(allData),[],2));
    xlabel('chan'); ylabel('stdev')
    title(imuFile)
    % savefig([imuFile(1:end-4) '_IMU_fig2']) % not saving for some reason
    
    
    save(outputFilename,'allData','-v7.3');
    
catch me
    
    rethrow(me)
    
end
