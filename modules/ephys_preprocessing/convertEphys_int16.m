function [allData ] = convertEphys_int16(filename, nChansTotal, outputDir)
%  convert binary ephys data from uint16 to int16
% kilosort and other analysis expects int16
% but Bonsai was initially set up to record uint16
% this will convert between them and save it back out
%
% returns processed traces (allData)

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
        outputFilename  = [pathstr filesep name '_int16' ext];
        mdTraceFilename = [pathstr filesep name '_medianTrace.mat'];
    else
        outputFilename  = [outputDir filesep name '_int16' ext];
        mdTraceFilename = [outputDir filesep name '_medianTrace.mat'];
    end
   outputFilename
   fidOut = fopen(outputFilename, 'w');
    
    % theseInds = 0;
    chunkInd = 1;

    while 1
        
        fprintf(1, 'chunk %d/%d\n', chunkInd, nChunksTotal);
        
        dat = fread(fid, [nChansTotal chunkSize], '*uint16');
        
        if ~isempty(dat)
           % keyboard
            %         theseInds = theseInds(end):theseInds(end)+chunkSize-1;
            dat16 = int16(double(dat)-2^15);
            tm = median(dat16,1);
            fwrite(fidOut, dat16, 'int16');
            allData(1:length(subChans),(chunkInd-1)*chunkSize+1:(chunkInd-1)*chunkSize+numel(tm)) = dat16;
        else
            break
        end
        
        chunkInd = chunkInd+1;
    end
    
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
