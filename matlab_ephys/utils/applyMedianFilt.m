function [allData, medianTrace] = applyMedianFilt(nChansTotal, doMedian, subChans, isUint16, chanMap, doFileChoose)
% applyMedianFilt Subtract median of each channel. Then, subtract median of
% each time point.
% 
%   Previously named: `applyCARtoDat_subset_multi`
%
%   This function merges multiple recordings into one .bin for Kilosort. It
%   creates a .mat that will allow recordings to be split apart again. If
%   the data is moved between computers or it is moved to a new disk/path
%   on the same computer, file paths saved in the .mat will be wrong and
%   need to be corrected using the script updateEphysPaths.m
%   
%   When this function is run, a GUI will open and ask for the .bin files
%   to merge. Pick the recordings in chronological order. After each file,
%   a new window will open. Select all files, and when the last one has
%   been selected and the GUI reopens, hit 'cancel'.
%   Another GUI will open. Enter the .bin output file path to save merged
%   data into.
%
%   Can select a subset of channels first, to avoid including noise in the
%   median filter. 16-channel probes are read in with the dimensions of a
%   32-channel probe, so these should use the subset 9:24 to ignore the
%   empty channels. 64- and 128-channel probes should use the subset 1:64
%   and 1:128
%
%
%   Inputs
%
%       nChansTotal     :  Number of channels on the probe.
%       doMedian        :  Option to subtract medians. Yes=1, No=0.
%       subChans        :  Subset of channels to includie in output. Subsetting
%                          occurs after channel remapping, so select probe
%                          sites to include.
%       isUint16        :  If the raw data is uint16 it needs to be converted
%                          to int16. Yes=1, No=0.
%       chanMap         :  List of data channels to be mapped to each probe
%                          site. e.g., chanmap(1) = 42 means that the data
%                          recorded in channel 42 is assigned to probe site 1
%
%   Returns
%       
%       allData         :  Processed trace with shape (channels, time).
%       medianTrace     :  CAR median of allData
%
%
% Niell lab - FreelyMovingEphys
% Written by CMN Dec 2020
% Based on cortex repository by N. Steinmetz
%

if ~exists('doFileChoose', 'var')
    doFileChoose = 1;
end

if ~exist('doMedian','var') || isempty(doMedian)
    doMedian = 1;
end

if ~exist('subChans','var') || isempty(subChans)
    subChans = 1:nChansTotal;
end

if ~exist('chanMap','var') || isempty(chanMap)
    chanMap = 1:nChansTotal;
end

if ~exist('isUint16','var')
    isUint16 = 0;
end

% Should make chunk size as big as possible so that the medians of the
% channels differ little from chunk to chunk.
chunkSize = 1000000;

% Choose files
if doFileChoose==1
    
    done=0;
    % Number of files.
    nf = 0;
    
    % Select .bin files to merge.
    while ~done
        [f, p] = uigetfile('*.bin', 'ephys file to merge');
        
        sprintf(['Adding to merge list: ' f])
    
        if f~=0
            nf = nf+1;
            fileList{nf} = f;
            pathList{nf} = p;
        else
            done = 1;
        end
    end
    
    % Select output .bin file
    [f, p]= uiputfile('*.bin', 'merged output file');
    outputFilename = fullfile(p,f);
end

if ~exists('nf', 'val')
    nf = size(fileList);
end


try
    
    fidOut = fopen(outputFilename, 'w');
    
    for fnum = 1:nf
        filename = fullfile(pathList{fnum},fileList{fnum});
        fid = fopen(filename, 'r');
        
        d = dir(filename);
        nSampsTotal = d.bytes/nChansTotal/2;
        nChunksTotal = ceil(nSampsTotal/chunkSize);
        nSamps(fnum) = nSampsTotal;
        
        % theseInds = 0;
        chunkInd = 1;
        medianTrace = zeros(1, nSampsTotal);
       
        % Load data for this file, filter, and save out
        while 1
            
            fprintf(1, 'chunk %d/%d\n', chunkInd, nChunksTotal);
            
            if isUint16
                dat = fread(fid, [nChansTotal chunkSize], '*uint16');
                dat = int16(double(dat)-2^15); %%% convert to int16
            else
                dat = fread(fid, [nChansTotal chunkSize], '*int16');
            end
            
            if ~isempty(dat)
                
                % Select appropriate channels
                dat = dat(subChans,:);
                
                % Perform remapping
                dat = dat(chanMap,:);
                
                % Filtering
                % Subtract median of each channel
                dat = bsxfun(@minus, dat, median(dat,2));
                tm = median(dat,1);
                if doMedian
                    % Subtract median of each time point
                    dat = bsxfun(@minus, dat, tm);
                end
                medianTrace((chunkInd-1)*chunkSize+1:(chunkInd-1)*chunkSize+numel(tm)) = tm;
                
                % Save data
                fwrite(fidOut, dat, 'int16');
                allData(1:length(subChans),(chunkInd-1)*chunkSize+1:(chunkInd-1)*chunkSize+numel(tm)) = dat;
            else
                break
            end       
            chunkInd = chunkInd+1;
        end
        
        %%% Save out median trace
        save([filename(1:end-4) '_medianTrace.mat'], 'medianTrace', '-v7.3');
     
        %%% Plot trace of each channel
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

        % Bar plot of stdev for each channel (noise measure)
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
        
    end % fnum
    fclose(fidOut);
    
    %%% Save .mat with info to separate phy output
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
