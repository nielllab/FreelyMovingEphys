function [allData medianTrace] = applyCARtoDat_subset_multi(nChansTotal, doMedian, subChans, isUint16, chanMap)
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
% doMedian = option to subtract medians (1) or not (0) - latter is if you only want to subset
% subChans = subset of channels to includie in output (note - subsetting occurs after channel remapping, so select probe sites to include)
% isUint16 = raw data is uint16,so convert to int16
% chanMap = list of data channels to be mapped to each probe site (e.g. chanmap(1) = 42 means that the data recorded in channel 42 is assigned to probe site 1
%
% returns processed traces (allData) and CAR median (medianTrace)
%
% gui will ask for .bin files to merge. 'cancel' when done'
% next, will ask for .bin output file to save merged data into

if ~exist('doMedian','var') | isempty(doMedian)
    doMedian = 1;
end

if ~exist('subChans','var') | isempty(subChans)
    subChans = 1:nChansTotal;
end

if ~exist('chanMap','var') | isempty(chanMap)
    chanMap = 1:nChansTotal;
end

if ~exist('isUint16','var')
    isUint16=0;
end

chunkSize = 1000000;
done=0;
nf = 0; %%% number of files

%%% select files to merge
while ~done
    [f p] = uigetfile('*.bin','ephys file to merge')
    if f~=0
        nf =nf+1;
        fileList{nf} = f;
        pathList{nf} = p;
    else
        done = 1;
    end
end

%%% select output file
[f p ]= uiputfile('*.bin', 'merged output file');
outputFilename = fullfile(p,f);

try
    
    fidOut = fopen(outputFilename, 'w');
    
    for fnum = 1:nf
        filename = fullfile(pathList{fnum},fileList{fnum})
        fid = fopen(filename, 'r');
        
        d = dir(filename);
        nSampsTotal = d.bytes/nChansTotal/2;
        nChunksTotal = ceil(nSampsTotal/chunkSize);
        nSamps(fnum) = nSampsTotal;
        
        % theseInds = 0;
        chunkInd = 1;
        medianTrace = zeros(1, nSampsTotal);
       
        % load data for this file, filter, and save out
        while 1
            
            fprintf(1, 'chunk %d/%d\n', chunkInd, nChunksTotal);
            
            if isUint16
                dat = fread(fid, [nChansTotal chunkSize], '*uint16');
                dat = int16(double(dat)-2^15); %%% convert to int16
            else
                dat = fread(fid, [nChansTotal chunkSize], '*int16');
            end
            
            if ~isempty(dat)
                
                %%% select appropriate channels
                dat = dat(subChans,:);
                
                % perform remapping
                dat = dat(chanMap,:);
                
                %%% filtering
                dat = bsxfun(@minus, dat, median(dat,2)); % subtract median of each channel
                tm = median(dat,1);
                if doMedian
                    dat = bsxfun(@minus, dat, tm); % subtract median of each time point
                end
                medianTrace((chunkInd-1)*chunkSize+1:(chunkInd-1)*chunkSize+numel(tm)) = tm;
                
                %%% save data
                fwrite(fidOut, dat, 'int16');
                allData(1:length(subChans),(chunkInd-1)*chunkSize+1:(chunkInd-1)*chunkSize+numel(tm)) = dat;
            else
                break
            end       
            chunkInd = chunkInd+1;
        end
        
        %%% save out median trace
        save([filename(1:end-4) '_medianTrace.mat'], 'medianTrace', '-v7.3');
     
        %%% plot trace of each channel
        figure
        map64 = [1:2:64 2:2:64];
        map128 = [1:4:128 2:4:128 3:4:128 4:4:128];
        
        for i = 1:length(subChans)
            if length(subChans)==64
                subplot(32,2,map64(i));
            elseif length(subChans)==128
                subplot(32,4,map128(i));
            else
                subplot(length(subChans),1,i);
            end
            plot(allData(i,1:3000));
            axis off
            if i==1
                title(fileList{fnum});
            end
            xlabel(num2str(i));
        end
        savefig(['CAR_' fileList{fnum}(1:end-4) '_fig1'])

        %%% bar plot of stdev for each channel (noise measure)
        figure
        stdByChan = zeros(length(subChans),1);
        for i = 1:length(subChans)
            stdByChan(i,:) = std(double(allData(i,:)),[],2);
        end
        bar(stdByChan);
        xlabel('chan'); ylabel('stdev')
        title(fileList{fnum})
        savefig(['CAR_' fileList{fnum}(1:end-4) '_fig2'])
        
        fclose(fid);
        
    end  %fnum
    fclose(fidOut);
    
    %%% save .mat with info to separate phy output
    matFname = [outputFilename(1:end-4) '.mat'];
    save(matFname,'pathList','fileList','nSamps','doMedian','subChans');
    
catch me
    
    if ~isempty(fid)
        fclose(fid);
    end
    
    if ~isempty(fidOut)
        fclose(fidOut);
    end
    
    rethrow(me)
    
end
