function [blk_re, rate] = residual_coder(blk_m,blk, q_step)
    
    b_size = size(blk_m,1);
    residual = blk - blk_m;
    residual_DCT = dct(residual,8);
    blockDCTq = quantizer(residual_DCT,q_step);
    resBlock = idct(blockDCTq,8);
    blk_re = resBlock + blk_m;
    
    % DCT coefficients
    vals = reshape(blockDCTq(:,:),[1,size(blockDCTq(:,:),1)*size(blockDCTq(:,:),2)]);

    bins_coefs = min(vals):q_step:max(vals);
    p = hist(vals,bins_coefs)/length(vals);
    
    % compute entropy from pdfs
    H = -sum(p.*log2(p+eps));
    rate = H*(b_size^2);
end