close all;
clear;

%% original image
lena=imread('lena512.bmp');
figure(1);
imshow(lena);
title('The Original Image');

hist1=hist(lena(:),0:255);
histlena=hist1./sum(hist1);
figure(2);
bar(histlena);
xlim([0,255]);
ylim([0,0.01]);
title('Histogram of the Original Image');

%% Gaussian noise
gs=mynoisegen('gaussian',512,512,0,64);
lenand=lena+uint8(round(gs));
figure(3);
imshow(lenand);
title('The Gaussian-noised Image');

hist2=hist(lenand(:),0:255);
histnd=hist2./sum(hist2);
figure(4);
bar(histnd);
xlim([0,255]);
ylim([0,0.01]);
title('Histogram of the Gaussian-noised Image');

%% mean filter
% meanf=fspecial('average',[3,3]);
% lenameanf=imfilter(lenand,meanf);
meanf=ones(3)./9;
lenameanf=uint8(conv2(lenand,meanf,'same'));
figure(5);
imshow(lenameanf);
title('The Mean Filtered Gaussian-noised Image');

hist3=hist(lenameanf(:),0:255);
histmeanf=hist3./sum(hist3);
figure(6);
bar(histmeanf);
xlim([0,255]);
ylim([0,0.01]);
title('Histogram of the Mean Filtered Gaussian-noised Image');

%% median filter
lenamedf=medfilt2(lenand,[3,3]);
figure(7);
imshow(lenamedf);
title('The Median Filtered Gaussian-noised Image');

hist4=hist(lenamedf(:),0:255);
histmedf=hist4./sum(hist4);
figure(8);
bar(histmedf);
xlim([0,255]);
ylim([0,0.01]);
title('Histogram of the Median Filtered Gaussian-noised Image');
