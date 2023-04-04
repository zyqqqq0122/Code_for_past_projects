close all;
clear all;

%% original image
lena=imread('lena512.bmp');
figure(1);
imshow(lena);
title('The Original Image');

hist1=hist(lena(:),0:255);
histlena=hist1./sum(hist1);
figure(2);
bar(histlena);
ylim([0,0.01]);
title('Histogram of the Original Image');

%% Salt&pepper noise
lenand=lena;
sp=mynoisegen('saltpepper',512,512,.05,.05);
lenand(sp==0)=0;
lenand(sp==1)=255;
figure(3);
imshow(lenand);
title('The Salt&pepper-noised Image');

hist2=hist(lenand(:),0:255);
histnd=hist2./sum(hist2);
figure(4);
bar(histnd);
ylim([0,0.01]);
title('Histogram of the Salt&pepper-noised Image');

%% mean filter
% meanf=fspecial('average',[3,3]);
% lenameanf=imfilter(lenand,meanf);
meanf=ones(3)./9;
lenameanf=uint8(conv2(lenand,meanf,'same'));
figure(5);
imshow(lenameanf);
title('The Mean Filtered Salt&pepper-noised Image');

hist3=hist(lenameanf(:),0:255);
histmeanf=hist3./sum(hist3);
figure(6);
bar(histmeanf);
ylim([0,0.01]);
title('Histogram of the Mean Filtered Salt&pepper-noised Image');

%% median filter
lenamedf=medfilt2(lenand,[3,3]);
figure(7);
imshow(lenamedf);
title('The Median Filtered Salt&pepper-noised Image');

hist4=hist(lenamedf(:),0:255);
histmedf=hist4./sum(hist4);
figure(8);
bar(histmedf);
ylim([0,0.01]);
title('Histogram of the Median Filtered Salt&pepper-noised Image');
