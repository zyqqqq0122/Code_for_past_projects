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
ylim([0,0.05]);
title('Histogram of the Original Image');

%% low-contrast image
a=0.2;
b=50;
lenalc=min(max(round(a*lena+b),0),255);
figure(3);
imshow(lenalc);
title('The Low-contrast Image');

hist2=hist(lenalc(:),0:255);
histlc=hist2./sum(hist2);
figure(4);
bar(histlc);
xlim([0,255]);
ylim([0,0.05]);
title('Histogram of the Low-contrast Image');
% figure(5);
% imagesc(lenalc,[50 100]);
% title('The Low-contrast Image');

%% equalization
s=round(255.*cumsum(histlc))+1;
hist3=zeros(1,256);
for ii=1:length(s)
    kk=s(ii);
    hist3(kk)=hist3(kk)+hist2(ii);
end
histeh=hist3./sum(hist3);
figure(6);
bar(histeh);
xlim([0,255]);
ylim([0,0.05]);
title('Histogram of the Enhanced Image');

figure(7);
lenaeh=histeq(lenalc);
imshow(lenaeh);
title('The Enhanced Image');
