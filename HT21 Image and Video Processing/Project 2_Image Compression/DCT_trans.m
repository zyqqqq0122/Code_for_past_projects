function [im_rec,im_rec_q,im_dct,im_dct_q] = DCT_trans(im,M,delta)

%%  Method 1
%         % DCT coeffiecients
%         im_dct=blkproc(im,[M,M],'dct2');
%         im_rec=blkproc(im_dct,[M,M],'idct2');
%         % quantized coefficients
%         im_dct_q=quantizer(im_dct,delta);
%         im_rec_q=blkproc(im_dct_q,[M,M],'idct2');
%% Method 2
    [a,b]=size(im);
    row=a/M;
    col=b/M;
    for i=1:row
       for j=1:col
          % 8*8 block 
          blk=im((i-1)*M+1:i*M,(j-1)*M+1:j*M);
          % block DCT transform
          blk_dct=dct2(blk);
          % original DCT coefficients
          im_dct((i-1)*M+1:i*M,(j-1)*M+1:j*M)=blk_dct;
          % quantization
          blk_dct_q=quantizer(blk_dct,delta);
          % quantized DCT coefficients
          %im_dct_q((i-1)*M+1:i*M,(j-1)*M+1:j*M)=quantizer(...
          %    im_dct((i-1)*M+1:i*M,(j-1)*M+1:j*M),delta);
          im_dct_q((i-1)*M+1:i*M,(j-1)*M+1:j*M)=blk_dct_q;
          % inverse DCT
          blk_rec=idct2(blk_dct);
          blk_rec_q=idct2(blk_dct_q);
          % reconstructed image
          im_rec((i-1)*M+1:i*M,(j-1)*M+1:j*M)=blk_rec;
          im_rec_q((i-1)*M+1:i*M,(j-1)*M+1:j*M)=blk_rec_q;
       end
    end
end