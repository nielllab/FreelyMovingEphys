function [allData medianTrace] = applyCARtoDat_subset(filename, nChansTotal, outputDir, doMedian, subChans)
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
%
% returns processed traces (allData) and CAR median (medianTrace)

if ~exist('doMedian','var') | isempty(doMedian)
    doMedian = 1;
end

if ~exist('subChans','var') | isempty(subChans)
    subChans = 1:nChansTotal;
end
chunkSize = 1000000;

fid = []; fidOut = [];

d = dir(filename);
nSampsTotal = d.bytes/nChansTotal/2;
nChunksTotal = ceil(nSampsTotal/chunkSize);
try
    
    [pathstr, name, ext] = fileparts(filename);
    fid = fopen(filename, 'r');
    if nargin < 3 | isempty(outputDir)
        outputFilename  = [pathstr filesep name '_CARsub' ext];
        mdTraceFilename = [pathstr filesep name '_medianTrace.mat'];
    else
        outputFilename  = [outputDir filesep name '_CARsub' ext];
        mdTraceFilename = [outputDir filesep name '_medianTrace.mat'];
    end
    fidOut = fopen(outputFilename, 'w');
    
    % theseInds = 0;
    chunkInd = 1;
    medianTrace = zeros(1, nSampsTotal);
    while 1
        
        fprintf(1, 'chunk %d/%d\n', chunkInd, nChunksTotal);
        
        dat = fread(fid, [nChansTotal chunkSize], '*int16');
        
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
    
catch me
    
    if ~isempty(fid)
        fclose(fid);
    end
    
    if ~isempty(fidOut)
        fclose(fidOut);
    end
    
    
    rethrow(me)
    
end
