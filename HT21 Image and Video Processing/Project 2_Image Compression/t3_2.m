clear;
close all;

load coeffs.mat;
im=imread('harbour512x512.tif');
im=double(im);
scale=4;
psv=dbwavf('db8');

coeff=fwt(im,scale,psv);

im_rec=ifwt(coeff,scale,psv);

figure;
imshow(uint8(im));
title('Original image');
figure;
imshow(uint8(coeff));
title('Wavelet coefficients,scale=4');
figure;
imshow(uint8(im_rec));
title('Reconstructed image');

d=MSE(im,im_rec);
psnr=PSNR(d);