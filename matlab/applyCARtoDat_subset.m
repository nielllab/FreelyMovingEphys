function [allData medianTrace] = applyCARtoDat_subset(filename, nChansTotal, outputDir, doMedian, subChans, isUint16)
% Subtracts median of each channel, then subtracts median of each time
% point. Also, can select a subset of channels first (so don't include
% noise in median filter)
% based on cortex repository by N. Steinmetz
% edited by cmn 2020
%
% filename should include the extension
% outputDir is optional (can leave empty), by default will write to the directory of the input file
%
% should make chunk size as big as possible so that the medians of the
% channels differ little from chunk to chunk.
%
% doMedian = option to subtract medians (1) or not (0) - latter is if you
% only want to subset
% subChans = subset of channels to includie in output
% isUint16 = raw data is uint16,so convert to int16
% returns processed traces (allData) and CAR median (medianTrace)

if ~exist('doMedian','var') | isempty(doMedian)
    doMedian = 1;
end

if ~exist('subChans','var') | isempty(subChans)
    subChans = 1:nChansTotal;
end

if ~exist('isUint16','var')
    isUint16=0;
end
chunkSize = 1000000;

fid = []; fidOut = [];

d = dir(filename);
nSampsTotal = d.bytes/nChansTotal/2;
nChunksTotal = ceil(nSampsTotal/chunkSize);
try
    
    [pathstr, name, ext] = fileparts(filename);
    fid = fopen(filename, 'r');
    suffix = sprintf('_int16_med%d_nch%d',doMedian,length(subChans));
    if nargin < 3 | isempty(outputDir)
        outputFilename  = [pathstr filesep name suffix ext];
        mdTraceFilename = [pathstr filesep name '_medianTrace.mat'];
    else
        outputFilename  = [outputDir filesep name suffix ext];
        mdTraceFilename = [outputDir filesep name '_medianTrace.mat'];
    end
    fidOut = fopen(outputFilename, 'w');
    
    % theseInds = 0;
    chunkInd = 1;
    medianTrace = zeros(1, nSampsTotal);
    while 1
        
        fprintf(1, 'chunk %d/%d\n', chunkInd, nChunksTotal);
        
        if isUint16
        dat = fread(fid, [nChansTotal chunkSize], '*uint16');
        dat = int16(double(dat)-2^15); %%% convert to int16
        else
          dat = fread(fid, [nChansTotal chunkSize], '*int16');
        end
        
        if ~isempty(dat)
           % keyboard
            %         theseInds = theseInds(end):theseInds(end)+chunkSize-1;
            dat = dat(subChans,:);
            
            dat = bsxfun(@minus, dat, median(dat,2)); % subtract median of each channel
            tm = median(dat,1);
            if doMedian
                dat = bsxfun(@minus, dat, tm); % subtract median of each time point
            end
            medianTrace((chunkInd-1)*chunkSize+1:(chunkInd-1)*chunkSize+numel(tm)) = tm;
            
            fwrite(fidOut, dat, 'int16');
            allData(1:length(subChans),(chunkInd-1)*chunkSize+1:(chunkInd-1)*chunkSize+numel(tm)) = dat;
        else
            break
        end
        
        chunkInd = chunkInd+1;
    end
    
    save(mdTraceFilename, 'medianTrace', '-v7.3');
    fclose(fid);
    fclose(fidOut);
    
    figure
    for i = 1:length(subChans)
        subplot(ceil(length(subChans)/2),2,i)
        plot(allData(i,1:3000));
        axis off
    end
    figure
    bar(std(double(allData),[],2));
    xlabel('chan'); ylabel('stdev')
    
catch me
    
    if ~isempty(fid)
        fclose(fid);
    end
    
    if ~isempty(fidOut)
        fclose(fidOut);
    end
    
    
    rethrow(me)
    
end
