clear;
close all;

im1=double(imread('boats512x512.tif'));
im2=double(imread('harbour512x512.tif'));
im3=double(imread('peppers512x512.tif'));

M=8;

step_size=[1 2 4 8 16 32 64 128 256 512];

rate_dct=zeros(1,length(step_size));
psnr_dct=zeros(1,length(step_size));

for k=1:length(step_size)
    delta=step_size(k);
    
    [im_rec1,im_rec_q1,im_dct1,im_dct_q1]=DCT_trans(im1,M,delta);
    [im_rec2,im_rec_q2,im_dct2,im_dct_q2]=DCT_trans(im2,M,delta);
    [im_rec3,im_rec_q3,im_dct3,im_dct_q3]=DCT_trans(im3,M,delta);

    d=MSE([im_dct1,im_dct2,im_dct3],[im_dct_q1,im_dct_q2,im_dct_q3]);
    psnr_dct(k)=PSNR(d);
    
    coeff_q=[im_dct_q1,im_dct_q2,im_dct_q3];
    coeff=zeros(M,M,64*64*3);
    
    for u=1:M
       for v=1:M
          for img=1:3
              temp=1;
              for ii=0:63
                 for jj=0:63
                    coeff(u,v,64*64*(img-1)+temp)=coeff_q(M*ii+u,img*M*jj+v);
                    temp=temp+1;
                 end
              end
          end
       end
    end
    
    H=zeros(M);
    
    for u=1:M
       for v=1:M
          h=squeeze(coeff(u,v,:));
          p=hist(h,min(h):delta:max(h));
          p=p/sum(p);
          H(u,v)=-sum(p.*log2(p+eps));
       end
    end
    rate_dct(k)=mean2(H);
end
% figure;
% surf(H);
% title('Average entropy of 8x8 DCT block coefficients');
figure;
plot(rate_dct, psnr_dct,'-*','linewidth',0.75);
title('Rate-PSNR curve');
xlabel('Bit-Rate');
ylabel('PSNR[dB]');
grid on;

save rate_dct;
save psnr_dct;
