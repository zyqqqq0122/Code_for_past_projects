clear;
close all;

im1=double(imread('boats512x512.tif'));
M=8;
delta=128;
[im_rec1,im_rec_q1,im_dct1,im_dct_q1]=DCT_trans(im1,M,delta);

d=MSE(im1,im_rec_q1);
mse_coeff=MSE(im_dct1,im_dct_q1);

figure;
imshow(uint8(im1));
title('Original image');
figure;
imshow(uint8(im_dct1));
title('Original DCT coeffiecients');
figure;
imshow(uint8(im_rec1));
title('Reconstructed image without quantization');
figure;
imshow(uint8(im_dct_q1));
title('Quantized DCT coeffiecients,step-size=128');
figure;
imshow(uint8(im_rec_q1));
title('Reconstructed image with quantization,step-size=128');