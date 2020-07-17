%%% example script for free-moving video data
%%% creates montage of eye, top, and 2 world shifts

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