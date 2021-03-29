%%% create a matrix that calculates rms smoothness

nk = [10 10];
nks = prod(nk);
consecutive = ones(nks,1);
consecutive(nk(1):nk(1):end) = 0;
Dxx = spdiags(consecutive*[-1 1],[ 0 1],nks-1,nks);

Dxy = spdiags(ones(nks,1)*[-1 1],[ 0 nk(1)],nks - nk(1) ,nks);

Dx =  Dxx'*Dxx + Dxy'*Dxy    ;

figure
imagesc(Dxx); title('dxx')
figure
imagesc(Dxy); title('dxy')

figure
imagesc(Dx)
colormap jet

data = rand(nk);
rmsx =  sum(sum((data(1:end-1,:) - data(2:end,:)).^2))  
rmsy = sum(sum((data(:,1:end-1) - data(:,2:end)).^2 )) 

rms = rmsx + rmsy

rms_matrix = data(:)' * Dx * data(:)
