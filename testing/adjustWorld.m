clear all
px_deg = 5.23;
downsamp=0.25;
eyeDownsamp = 0.5;
topDownsamp = 0.25;
eyeInterpMethod = 'linear';  %%% 'linear' or 'nearest'; latter gives sharp images but not direct match

% [fWorldVid pWorldVid] = uigetfile('*.avi','worldcam vid');
% [fWorldTs pWorldTs] = uigetfile('*.csv','worldcam TS');
% [fEyeDLC pEyeDLC]= uigetfile('*.csv','eyecam DLC');
% [fEyeTS pEyeTs] = uigetfile('*.csv','eyecam ts');
% [fEyeVid pEyeVid] = uigetfile('*.avi','eyecam vid');

fWorldVid = dir('*world*.avi').name; pWorldVid = '.';
fWorldTs = dir('*world*.csv').name; pWorldTs = '.';
fEyeDLC = dir('*eye*Deep*.csv').name; pEyeDLC = '.';
fEyeTS = dir('*eye*TS*.csv').name; pEyeTs = '.';
fEyeVid = dir('*eye*.avi').name; pEyeVid = '.';
fTopVid = dir('*top*.avi').name; pTopVid= '.';
fTopTs = dir('*top*.csv').name; pTopTs = '.';

%%% load worldcam timestamps
worldTsRaw = dlmread(fullfile(pWorldTs,fWorldTs));
worldTsRaw= worldTsRaw(:,1)*60*60 + worldTsRaw(:,2)*60 + worldTsRaw(:,3);  %%% data is read as hours, mins, secs

dt = median(diff(worldTsRaw));
worldTs = zeros(size(worldTsRaw,1)*2,1);
worldTs(1:2:end) = worldTsRaw-0.25*dt;
worldTs(2:2:end) = worldTsRaw + 0.25*dt;

%%% load eyecam timestamps
eyeTs = dlmread(fullfile(pEyeTs,fEyeTS));
eyeTs= eyeTs(:,1)*60*60 + eyeTs(:,2)*60 + eyeTs(:,3);  %%% data is read as hours, mins, secs

%%% load top video timestamps
topTs = dlmread(fullfile(pTopTs,fTopTs));
topTs= topTs(:,1)*60*60 + topTs(:,2)*60 + topTs(:,3);  %%% data is read as hours, mins, secs


startT = min(eyeTs(1),worldTs(1));
        
%%% load eye DLC
eyeDLC = csvread(fullfile(pEyeDLC,fEyeDLC),3,0);
%%% parse columns (x,y,likelihood)
eyeX = eyeDLC(:,2:3:end); eyeY = eyeDLC(:,3:3:end); eyeP = eyeDLC(:,4:3:end);
%%% calculate theta, phi
[eyeThetaRaw,eyePhiRaw, eyeEllipseRaw] = EyeCameraCalc1(length(eyeX), eyeX,eyeY, eyeP);
%%% interpolate to world cam timestamps
eyeTheta = interp1(eyeTs,eyeThetaRaw,worldTs); eyePhi = interp1(eyeTs,eyePhiRaw,worldTs);eyeEllipse = interp1(eyeTs,eyeEllipseRaw,worldTs);
eyeEllipse = eyeEllipse*eyeDownsamp;
eyeX = eyeEllipse(:,1); eyeY = eyeEllipse(:,2);

eyePhi = eyePhi-nanmean(eyePhi);
eyeTheta = eyeTheta - nanmean(eyeTheta);

%%% plot as safety check
figure
subplot(2,1,1); plot(eyeTs-startT,eyeThetaRaw);hold on; plot(worldTs-startT,eyeTheta); ylabel('theta');
subplot(2,1,2); plot(eyeTs-startT,eyePhiRaw); hold on; plot(worldTs-startT,eyePhi); ylabel('phi'); legend('raw','interp');
 
%% load world video 
TempVidT = VideoReader(fullfile(pWorldVid,fWorldVid));
frame=1;
display('reading worldvid')
while hasFrame(TempVidT)
    worldVidRaw(:,:,frame) = mean(readFrame(TempVidT),3);   
    %%% status update
    if mod(frame,500)==0
        fprintf('frame = %d\n',frame)
    end
    frame=frame+1;   
end
worldVid = zeros(size(worldVidRaw,1),size(worldVidRaw,2),size(worldVidRaw,3)*2);
worldVid(:,:,1:2:end) = imresize(worldVidRaw(1:2:end,:,:),[size(worldVid,1) size(worldVid,2)]);
worldVid(:,:,2:2:end) = imresize(worldVidRaw(2:2:end,:,:),[size(worldVid,1) size(worldVid,2)]);
clear worldVidRaw

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
eyeVid = zeros(size(eyeVidRaw,1),size(eyeVidRaw,2),size(eyeVidRaw,3)*2);
eyeVid(:,:,1:2:end) = imresize(eyeVidRaw(2:2:end,:,:),[size(eyeVid,1) size(eyeVid,2)]);
eyeVid(:,:,2:2:end) = imresize(eyeVidRaw(1:2:end,:,:),[size(eyeVid,1) size(eyeVid,2)]);
eyeVid = imresize(eyeVid,eyeDownsamp);

dt = median(diff(eyeTs));
eyeTsDeinter = zeros(size(eyeTs,1)*2,1);
eyeTsDeinter(1:2:end) = eyeTs-0.25*dt;
eyeTsDeinter(2:2:end) = eyeTs + 0.25*dt;

d = reshape(eyeVid,size(eyeVid,1)*size(eyeVid,2),size(eyeVid,3));

eyeVidAligned = interp1(eyeTsDeinter,d',worldTs,eyeInterpMethod)';
eyeVidAligned = reshape(eyeVidAligned,size(eyeVid,1),size(eyeVid,2),length(worldTs));
clear eyeVidRaw

%% load top video
TempVidT = VideoReader(fullfile(pTopVid,fTopVid));
frame=1;
display('reading top vid')
while hasFrame(TempVidT)
    topVid(:,:,frame) = imresize(mean(readFrame(TempVidT),3),topDownsamp);   
    %%% status update
    if mod(frame,500)==0
        fprintf('frame = %d\n',frame)
    end
    frame=frame+1;   
end

d = reshape(topVid,size(topVid,1)*size(topVid,2),size(topVid,3));

topVidAligned = interp1(topTs,d',worldTs,'nearest')';
topVidAligned = reshape(topVidAligned,size(topVid,1),size(topVid,2),length(worldTs));

clear topVidRaw


%% plot eye frames
figure
for i = 1:100
    subplot(10,10,i);
    imagesc(eyeVidAligned(:,:,i)); colormap gray;
    title(sprintf('th %0.1f phi %0.1f',eyeTheta(i),eyePhi(i)));
axis equal
axis off
end

pxdeg_resamp=px_deg/downsamp;

%% analyze world vid
worldSm = imresize(worldVid,downsamp);
worldSm = worldSm - mean(worldSm(:));

i=1;
for i = 1:size(worldSm,3)-1;
cc = xcorr2(worldSm(:,:,i),worldSm(:,:,i+1));
[maxval maxind] = max(cc(:));
[ymax(i) xmax(i)] = ind2sub(size(cc),maxind);
i
end

xmax = xmax-160;
ymax= ymax-120; 

xmax(xmax==0)=NaN;
ymax(ymax==0)=NaN;

dth = diff(eyeTheta);
dph = diff(eyePhi);

figure
for i = 1:12
    subplot(3,4,i);
    imagesc(worldSm(:,:,i),[-100 100]); colormap gray;
    if i>1
        title(sprintf('x=%0.0f y=%0.0f dth=%0.0f dph=%0.0f',xmax(i-1),ymax(i-1),eyeTheta(i),eyePhi(i)));
    end
end

%% plot data for xcorr shifts
figure
plot(ymax);
figure
plot(xmax);

yr = [-0.5 0.5];
figure
subplot(2,2,1)
[xc lags]=nanxcorr(dth(1:3000),xmax(1:3000),50,'coeff');
plot(lags,xc); ylim(yr); hold on;plot([0 0],yr);
title('dtheta vs dx');

subplot(2,2,2);
[xc lags]=nanxcorr(dph(1:3000),xmax(1:3000),50,'coeff');
plot(lags,xc); ylim(yr); hold on;plot([0 0],yr);
title('dphi vs dx');

subplot(2,2,3)
[xc lags]=nanxcorr(dth(1:3000),ymax(1:3000),50,'coeff');
plot(lags,xc); ylim(yr); hold on;plot([0 0],yr);
title('dtheta vs dy');

subplot(2,2,4)
[xc lags]=nanxcorr(dph(1:3000),ymax(1:3000),50,'coeff');
plot(lags,xc); ylim(yr); hold on;plot([0 0],yr);
title('dphi vs dy');

figure
plot(dph(1:3000),ymax(1:3000),'.'); ylim([-50 50])
figure
plot(dth(1:3000),ymax(1:3000),'.'); ylim([-50 50])


%%% worldSm(x,y,f) = world camera (downsampled 4x and zero-centered)
%%% eyeVidAligned(x,y,f) = eye camera (downsampled 2x)
%%% topVidAligned(x,y,f) = top camera (downsampled 4x)
%%% eyeTheta(f), eyePhi(f) = theta and phi of eye position
%%% eyeEllipse(f,:) = parameters for eye ellipse fit (x0,y0,a,b,~,~,phi);

%%% create empty matrix for shifted world view
worldShift = zeros(size(worldSm,1)*2,size(worldSm,2)*2,size(worldSm,3));
worldShift(:) = NaN;
pix_deg = 4;

%%% limit range to shift over
phiMax = eyePhi; phiMax(phiMax>15) = 15; phiMax(phiMax<-15)=-15;
thMax = eyeTheta; thMax(thMax>20)=20; thMax(thMax<-20)=-20;

%%% insert worldview into worldShift with offset
for i = 1:length(eyePhi);
    if ~isnan(eyePhi(i)) &  ~isnan(eyeTheta(i))
        worldShift((61:180)-round(phiMax(i)*pix_deg),(81:240)-round(thMax(i)*pix_deg),i) = worldSm(:,:,i);
    end
end

figure; clear mov
offset =120;  %%% to start at arbitrary point within movie
for i = 1:min(3000, size(worldShift,3)+offset);
    f = i+offset;
    
    %%% display eyes
    subplot(2,2,1);
    hold off;
    %%% show eye vid image
    imagesc(eyeVidAligned(:,:,f),[0 300]); colormap gray; axis equal
    hold on;
    %%% plot center and ellipse
    plot(eyeX(f),eyeY(f),'r.');
    pts = plotEllipse(eyeEllipse(f,:));
    plot(pts(1,:),pts(2,:),'LineWidth',2);
    title(sprintf('th %0.1f phi %0.1f',eyeTheta(f), eyePhi(f)));
    axis equal; axis off
    
    %%% display top video
    subplot(2,2,2);
    imagesc(topVidAligned(:,:,f),[0 200]); colormap gray;
    axis equal; axis([100 420 20 250]); axis off
    
    
    %%% display image with target
    subplot(2,2,3)
    hold off
    imagesc(worldSm(:,:,f),[-100 100]); colormap gray;
    hold on;
    plot(81+eyeTheta(f)*pix_deg,61+eyePhi(f)*pix_deg,'ro');
    axis off
    axis equal; axis([1 160 1 120])

    %%% display shifted image
    subplot(2,2,4)
    imagesc(worldShift(:,:,f),[-100 100]); colormap gray;axis equal
    axis off
    
    drawnow
    mov(i)=getframe(gcf);
    
end


[fOut pOut] = uiputfile('*.avi')
if fOut~=0
    vid = VideoWriter(fullfile(pOut,fOut));
    vid.FrameRate = 15;
    open(vid);
    writeVideo(vid,mov);
    close(vid);
end

% figure
% for i = 1:100
%     subplot(10,10,i);
%     f = i+100;
%     imagesc(worldBuff(:,:,f),[-100 100]); colormap gray;
% end

worldBuffSm = imresize(worldShift(:,:,1:1000),0.5);
d = reshape(worldBuffSm,size(worldBuffSm,1)*size(worldBuffSm,2),size(worldBuffSm,3));

c = corr(d,'Rows','Pairwise');
figure
imagesc(c)
title('shifted corr')

clear c
for i = 1:1000
    for j = 1:1000;
        c(i,j) = 128 - sqrt(nanmean((d(:,i)-d(:,j)).^2));
    end
end
figure
imagesc(c)
title('shifted diff');



worldBuffSm = imresize(worldSm(:,:,1:1000),0.5);
d = reshape(worldBuffSm,size(worldBuffSm,1)*size(worldBuffSm,2),size(worldBuffSm,3));
c = corr(d,'Rows','PairWise');
figure
imagesc(c)
title('raw corr')

clear c
for i = 1:1000
    for j = 1:1000;
        c(i,j) = 128 - sqrt(nanmean((d(:,i)-d(:,j)).^2));
    end
end
figure
imagesc(c)
title('raw diff')

d1 = squeeze(worldSm(60,80,:));
figure
plot(nanxcorr(d1,d1,30,'coeff'));

hold on
d2 = squeeze(worldShift(120,160,:));
plot(nanxcorr(d2,d2,30,'coeff'));

maxlag = 120;
clear xc
for lag = -maxlag:maxlag
    xc1(lag+maxlag+1) = 128 - sqrt(nanmean((d1 - circshift(d1,lag)).^2));
end
figure
plot(xc1)

clear xc
for lag = -maxlag:maxlag;
    xc2(lag+maxlag+1) = 128 - sqrt(nanmean((d2 - circshift(d2,lag)).^2));
end
hold on
plot(xc2)

figure
plot(d1(1:end-1),d1(2:end),'.')

figure
plot(d2(1:end-1),d2(2:end),'.')



        

% figure; clear mov
% for i = 1:3000
%   f = i+120;
%     imagesc(worldSm(:,:,f),[-100 100]); colormap gray;axis equal
% axis off
% mov(i)=getframe(gcf);
% drawnow
% end

% vid = VideoWriter('World_J463b_112619_1_2.avi');
% vid.FrameRate = 20;
% open(vid);
% writeVideo(vid,mov);
% close(vid);

% pix_deg=5;
% figure
%  for i = 1:100
%      f = i+200;
%      subplot(10,10,i)
%      imagesc(worldSm(:,:,f),[-100 100]); colormap gray;
%      hold on;
%      plot(81+eyeTheta(f)*pix_deg,61+eyePhi(f)*pix_deg,'go');
%  axis off
%  end
 
%  figure
%  clear mov
%  for i = 1:1000
%      f = i+200;
%      %subplot(10,10,i)
%      hold off
%      imagesc(worldSm(:,:,f),[-100 100]); colormap gray;
%      hold on;
%      plot(81-eyeTheta(f)*pix_deg,61+eyePhi(f)*pix_deg,'go');
%      plot(81-eyeTheta(f)*pix_deg,61+eyePhi(f)*pix_deg,'b*');
%      axis off
%      axis([1 160 1 120])
%      drawnow
%      mov(i) = getframe(gcf);
%  end
 
 pix_deg = 3;
  figure
 clear mov
 for i = 1:500
     f = i+1200;
     
     subplot(1,2,1);    
     imagesc(eyeVidAligned(:,:,f),[0 255]); colormap gray; axis equal 
     axis off
     
     subplot(1,2,2);
     
     hold off
     imagesc(worldSm(:,:,f),[-100 100]); colormap gray;
     hold on;
     plot(81-eyeTheta(f)*pix_deg,61+eyePhi(f)*pix_deg,'go');
     plot(81-eyeTheta(f)*pix_deg,61+eyePhi(f)*pix_deg,'b*');
     axis off
     axis equal
     axis([1 160 1 120])
     drawnow
     mov(i) = getframe(gcf);
 end
 
 pix_deg=4;
 figure
 i = 0; n= 6;
 for f = [1237 1245 1250 1275 1292 1298];
   i = i+1;
    subplot(2,n,i+n);    
     imagesc(eyeVidAligned(:,:,f),[0 300]); colormap gray; axis equal 
    hold on;
    plot(eyeX(f),eyeY(f),'r.');
    pts = plotEllipse(eyeEllipse(f,:)*eyeDownsamp);
    plot(pts(1,:),pts(2,:),'LineWidth',2);
    title(sprintf('th %0.1f phi %0.1f',eyeTheta(f), eyePhi(f)));
     axis([80 240 120 240]);
     axis off
     
     subplot(2,n,i); 
     hold off
     imagesc(worldSm(:,:,f),[-100 100]); colormap gray;
     hold on;
     centx = 81 + (eyeTheta(f))*pix_deg;
     centy = 61+ + (eyePhi(f)-10)*pix_deg;
     plot(centx,centy,'ro','LineWidth',2);
     title(num2str(f))
      axis off;   axis equal;  axis([1 160 1 120])
 end
set(gcf,'Name',fWorldVid);
 
 
figure
plot(eyeTheta,eyePhi)


%%% load world video + ts
%%% deInter + update ts

%%% load de-intered DLC eye points + ts
%%% update ts
%%% compute phi/theta

% interpolate phi/theta to top videos

%%% shift = (alpha *theta) * cos(rot) + (beta*phi) * sin(rot)
%minimize inter-frame shift except on saccades