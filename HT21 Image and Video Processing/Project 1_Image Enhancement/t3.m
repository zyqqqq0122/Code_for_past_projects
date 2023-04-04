close all;
clear;

%% original image
lena=imread('lena512.bmp');
figure(1);
imshow(lena);
title('The Original Image');

lenaf=t3_spectragen(lena);
figure(2);
imshow(lenaf);
title('Spectrum of the Original Image');

%% blurred image
h=myblurgen('gaussian',8);
lenabld=uint8(min(max(round(conv2(lena,h,'same')),0),255));
figure(3);
imshow(lenabld);
title('The Blurred Image');

lenabf=t3_spectragen(lenabld);
figure(4);
imshow(lenabf);
title('Spectrum of the Blurred Image');

%% deblurred image
diff=min(max(round(conv2(lena,h,'same')),0),255)-conv2(lena,h,'same');
vn=var(diff(:));
lenadblr=t3_deblur(lenabld,h,vn);
figure(5);
imshow(lenadblr);
title('The Deblurred Image');

lenadblrf=t3_spectragen(lenadblr);
figure(6);
imshow(lenadblrf);
title('Spectrum of the Deblurred Image');