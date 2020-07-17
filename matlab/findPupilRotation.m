%%% script to find pupil edge and align over time to calculate cyclotorsion
%%% reads in deinterlated eye video, DLC points, and timestamps
%%% output shiftSmooth = estimated pupil rotation
%%% cmn 2020

clear all


%% filenames
fEyeDLC = dir('J475*eye*Deep*.csv').name; pEyeDLC = '.';
fEyeTS = dir('J475*eye*TS*.csv').name; pEyeTs = '.';
fEyeVid = dir('J475*eye*Deinter*.avi').name; pEyeVid = '.';


%% load eyecam timestamps
eyeTs = dlmread(fullfile(pEyeTs,fEyeTS));
eyeTs= eyeTs(:,1)*60*60 + eyeTs(:,2)*60 + eyeTs(:,3);  %%% data is read as hours, mins, secs
        
%% load eye DLC
eyeDLC = csvread(fullfile(pEyeDLC,fEyeDLC),3,0);
%%% parse columns (x,y,likelihood)
eyeX = eyeDLC(:,2:3:end); eyeY = eyeDLC(:,3:3:end); eyeP = eyeDLC(:,4:3:end);
%%% calculate theta, phi
[eyeTheta,eyePhi, eyeEllipseRaw] = EyeCameraCalc1(length(eyeX), eyeX,eyeY, eyeP);
ellipseDeinter = eyeEllipseRaw;

%% load eye video 
TempVidT = VideoReader(fullfile(pEyeVid,fEyeVid));
frame=1;
display('reading eye vid')
while hasFrame(TempVidT)
    eyeVidRaw(:,:,frame) = mean(readFrame(TempVidT),3);   
    %%% status update
    if mod(frame,500)==0
        fprintf('frame = %d\n',frame)
    end
    frame=frame+1;   
end

eyeVid = eyeVidRaw; clear eyeVidRaw;

eyeCent =nanmean(ellipseDeinter(:,1:2),1);
figure
for i = 1:12
    subplot(3,4,i);
    imagesc(eyeVid(:,:,ceil(rand*size(eyeVid,3))),[0 255]); colormap gray;
    pts = plotEllipse(ellipseDeinter(i,:));
    hold on
    %plot(pts(1,:),pts(2,:),'r')
    axis equal; axis off
    axis([eyeCent(1)-100 eyeCent(1)+100 eyeCent(2)-100 eyeCent(2)+100]); 
end

%%% this is where alignment starts

tic
clear params ci clear pupilEdge
nf = 600; rangeR = 10;

%%% get ellipse params for easy reference
A0= ellipseDeinter(:,3); B0 = ellipseDeinter(:,4);
xcent = ellipseDeinter(:,1); ycent = ellipseDeinter(:,2);
for f = 1:nf
    f
%%% get cross-section of pupil at each angle 1-360 and fit to sigmoid
for th = 1:360;
    meanR = 0.5*(A0(f)+B0(f));   % get the mean radius; really this should be done by computing ellipse point from a,b,phi; x = a*[0.7:1.3]*cosd(th); y = b*[0.7:1.3]*sind(th); multiply x,y, by phi rotmat (see plotEllipse); then circshift all values by phi
    r = (meanR-rangeR):meanR+rangeR;
    
    %%% go out along radius and get pixel values
    for i = 1:length(r)
        try
            pupilEdge(th,i) = eyeVid(round(ycent(f)+r(i)*sind(th)),round( xcent(f) + r(i)*cosd(th)),f);
        catch
            display('out of frame')
        end
    end
    
    %%% fit sigmoind to pupil edge at this theta
    d = squeeze(pupilEdge(th,:));
[params(th,:,f) stat] = sigm_fit(1:length(d), d,[],[100 200 10 0.5],0);
ci(th,:,f) = stat.paramCI(:,2) - stat.paramCI(:,1);
end
% figure
% imagesc(edge(:,:,f)',[0 255]); colormap gray
% hold on
% plot(params(th,3,f),'r');
end %f
toc

%%% clean up fit
fitThresh = 1;
%%% extract radius variable from parameters
rfit = squeeze(params(:,3,:))-1;

%%% if confidence interval in estimate is >fitThresh pix, set to to NaN
rfit(squeeze(ci(:,3,:))>fitThresh) = NaN;

%%% remove if luminance goes the wrong way (e.g. from reflectance)
dIntensity = squeeze(params(:,2,:)-params(:,1,:));
rfit(dIntensity<0)= NaN;

%%% median filter to clear it up a little
rfit = medfilt1(rfit,3,[],1);

figure
plot(rfit)


filtSz = 30;
%%% subtract baseline (boxcar average using conv); this is because our
%%% points aren't perfectly centered on ellipse
for f = 1:size(rfit,2)
    rfitConv(:,f) = rfit(:,f) - nanconv(rfit(:,f),ones(filtSz,1)/filtSz,'same');
end
%%% edges have artifact from conv, so set to NaNs. Could fix this by
%%% padding data with wraparound at 0 and 360deg before conv
rfitConv(1:filtSz/2+1,:) = NaN; rfitConv(end-filtSz/2-1:end,:) = NaN;
figure
plot(rfitConv);


%%% plot data into movie
figure
clear mov
for f = 1:size(rfit,2)
    if ~isnan(ycent(f))
        hold off
        %%% show video frame
        imagesc(eyeVid(:,:,f),[0 255]); colormap gray; axis equal; hold on
        
        %%% calculate points (based on how they were extracted above
        th = 1:360;
        rmin = 0.5*(A0+B0) - rangeR;
        plot( xcent(f) + (rmin(f) + rfit(:,f)).*cosd(th)',ycent(f)+(rmin(f)+rfit(:,f)).*sind(th)','g')
        axis([xcent(f)-100 xcent(f)+100 ycent(f)-100 ycent(f)+100]);
    end
    mov(f) = getframe(gcf);
end

 %%% write movie
[fOut pOut] = uiputfile('*.avi')
if fOut~=0
    vid = VideoWriter(fullfile(pOut,fOut));
    vid.FrameRate = 10;
    vid.Quality = 100;
    open(vid);
    writeVideo(vid,mov);
    close(vid);
end


%%% get correlation across timepoints (only at zero lag, so not great. but
%%% but shows reliability in stable periods
figure
imagesc(corrcoef(rfitConv,'Rows','pairwise'),[0 1])
title('correlation of radius fit across timepoints')

%%% calculate mean as template (not good without some alignment
template = nanmean(rfitConv(:,100:120),2);
figure
plot(template)

figure
plot(nanxcorr(rfitConv(:,110),template,30,'coeff'))
hold on; plot([31 31],[-0.5 0.5])

%%% xcorr of two random timepoints
t(1) = 1; t(2) = 100;
figure
plot(nanxcorr(rfitConv(:,t(1)),rfitConv(:,t(2)),30,'coeff'))
hold on; plot([31 31],[-0.5 0.5])

%%% iterative fit to alignment
%%% start with mean as template
%%% on each iteration, shift individual frames to max xcorr with template
%%% then recalculate mean template

n= size(rfitConv,2);
pupilUpdate = rfitConv;
templateFig = figure; hold on; set(gcf,'Name','template')
histFig = figure; hold on; set(gcf,'Name','correlation histogram');
totalShift = zeros(n,1); c= totalShift;
for rep = 1:12
   %%% calculate and plot template
    template = nanmedian(pupilUpdate,2);
   figure(templateFig)
   subplot(3,4,rep);
   plot(template); title(sprintf('iter %d',rep));
   %%% loop over each frame, take xcorr, and shift accordingly
   for i = 1:n
        [xc lags]= nanxcorr(template,pupilUpdate(:,i),10,'coeff');     
        [c(i) peaklag] = max(xc);
        peak(i) = lags(peaklag);
        totalShift(i) = totalShift(i)+peak(i);
        pupilUpdate(:,i) = circshift(pupilUpdate(:,i),peak(i),1);
   end
    
   %%%histogram of correlations
    figure(histFig)
    subplot(3,4,rep);
    plot(0:0.05:1, hist(c,0:0.05:1)); xlabel('corr')
    title(sprintf('iter %d',rep));
    drawnow
   meanC(rep) =  mean(c)
end

figure
plot(meanC); xlabel('rep'); ylabel('mean xc');

figure
plot(c,totalShift,'.'); xlabel('correlation'); ylabel('shift');


win = 3;
shiftNan = -totalShift;
shiftNan(c<0.4) = NaN;
%shiftSmooth = medfilt1(shiftSmooth,3);
shiftSmooth = nanconv(shiftNan,ones(win,1)/win,'same');
shiftSmooth = shiftSmooth - nanmedian(shiftSmooth);
shiftNan = shiftNan-nanmedian(shiftNan);
figure
plot(shiftNan)
figure
plot(shiftSmooth)


figure
d = -40:40;
clear mov

for f = 1:length(shiftSmooth)
   if ~isnan(ycent(f))
       for sp = 1:4
        subplot(2,2,sp)
        hold off
        imagesc(eyeVid(:,:,f),[0 255]); colormap gray;axis equal
        hold on
        if sp==2 | sp ==4
            plot(xcent(f) + d*cosd(shiftSmooth(f)+90), ycent(f)+d*sind(shiftSmooth(f)+90),'LineWidth',2)
        end
        if sp ==3 | sp==4
            th = 1:360;
            rmin = 0.5*(A0+B0) - rangeR;
            plot( xcent(f) + (rmin(f) + rfit(:,f)).*cosd(th)',ycent(f)+(rmin(f)+rfit(:,f)).*sind(th)','g')
        end
        axis([xcent(f)-100 xcent(f)+100 ycent(f)-100 ycent(f)+100]);
        axis off
       if sp==1
           title(sprintf('corr = %0.2f',c(f)));
       end
        drawnow
       end
   end
    mov(f) = getframe(gcf);
end

 %%% write movie
[fOut pOut] = uiputfile('*.avi')
if fOut~=0
    vid = VideoWriter(fullfile(pOut,fOut));
    vid.FrameRate = 15;
    vid.Quality = 100;
    open(vid);
    writeVideo(vid,mov);
    close(vid);
end
    
figure
plot(nanxcorr(pupilUpdate(:,10),template,30,'coeff'))
