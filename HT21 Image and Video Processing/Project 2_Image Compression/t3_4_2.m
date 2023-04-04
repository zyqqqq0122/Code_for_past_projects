clear;
close all;
load coeffs.mat;
load rate_dct.mat;
load psnr_dct.mat;

im1=double(imread('boats512x512.tif'));
im2=double(imread('harbour512x512.tif'));
im3=double(imread('peppers512x512.tif'));

scale=1;
psv=dbwavf('db8');
step_size=[1 2 4 8 16 32 64 128 256 512];

psnr=zeros(1,length(step_size));

for k=1:length(step_size)
    
    delta=step_size(k);
    
    wacf1=fwt(im1,scale,psv);
    im_rec1=ifwt(wacf1,scale,psv);
    wacf_q1=quantizer(wacf1,delta);
    im_rec_q1=ifwt(wacf_q1,scale,psv);
    wacf2=fwt(im2,scale,psv);
    im_rec2=ifwt(wacf2,scale,psv);
    wacf_q2=quantizer(wacf2,delta);
    im_rec_q2=ifwt(wacf_q2,scale,psv);
    wacf3=fwt(im3,scale,psv);
    im_rec3=ifwt(wacf3,scale,psv);
    wacf_q3=quantizer(wacf3,delta);
    im_rec_q3=ifwt(wacf_q3,scale,psv);
    
    % MSE
    d=MSE([im1 im2 im3],[im_rec_q1,im_rec_q2,im_rec_q3]);
    psnr(k)=PSNR(d);
    
end

% bit-rate
H1=entropy_fwt(im1,scale,psv,step_size);
H2=entropy_fwt(im2,scale,psv,step_size);
H3=entropy_fwt(im3,scale,psv,step_size);
rate=(H1+H2+H3)/3;

figure;
plot(rate, psnr,'-*','linewidth',0.75);
hold on;
plot(rate_dct, psnr_dct,'-*','linewidth',0.75);
legend('FWT','DCT','Location','SouthEast');
title('Rate-PSNR curve');
xlabel('Bit-Rate');
ylabel('PSNR[dB]');
grid on;
