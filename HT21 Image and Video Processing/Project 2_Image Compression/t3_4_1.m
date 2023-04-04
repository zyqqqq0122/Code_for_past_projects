clear;
close all;
load coeffs.mat;

im1=double(imread('boats512x512.tif'));
scale=4;
psv=dbwavf('db8');
delta=2;

%% method 1 - matlab functions
% [Lo_D,Hi_D,Lo_R,Hi_R]=orthfilt(psv);
% [cA,cH,cV,cD]=dwt2(im1,Lo_D,Hi_D,'mode','per');
% im_rec1=idwt2(cA,cH,cV,cD,Lo_R,Hi_R,'mode','per');
% cA_q=quantizer(cA,delta);
% cH_q=quantizer(cH,delta);
% cV_q=quantizer(cV,delta);
% cD_q=quantizer(cD,delta);
% im_rec_q1=idwt2(cA_q,cH_q,cV_q,cD_q,Lo_R,Hi_R,'mode','per');
% wacf1=cat(1,[cA cH],[cV cD]);
% wacf_q1=cat(1,[cA_q cH_q],[cV_q cD_q]);

%% method 2 - designed by our own
wacf1=fwt(im1,scale,psv);
im_rec1=ifwt(wacf1,scale,psv);
wacf_q1=quantizer(wacf1,delta);
im_rec_q1=ifwt(wacf_q1,scale,psv);

%% MSE
d=MSE(im1,im_rec_q1);
mse_coeff=MSE(wacf1,wacf_q1);

%% figures
figure;
imshow(uint8(im1));
title('Original image');
figure;
imshow(uint8(wacf1));
title('Original wavelet cofficients');
figure;
imshow(uint8(im_rec1));
title('Reconstructed image without quantization');
figure;
imshow(uint8(wacf_q1));
title('Quantized wavelet cofficients,step-size=2');
figure;
imshow(uint8(im_rec_q1));
title('Reconstructed image with quantization,step-size=2');