function [re_block, rate] = blk_coder(block, q, b_size)
    blockDCT = dct(block,8);
    blockDCTq = quantizer(blockDCT,q);
    re_block = idct(blockDCTq,8);
    
    % DCT coefficients
    vals = reshape(blockDCTq(:,:),[1,size(blockDCTq(:,:),1)*size(blockDCTq(:,:),2)]);

    bins_coefs = min(vals):q:max(vals);
    p = hist(vals,bins_coefs)/length(vals);
    
    % entropy
    H = -sum(p.*log2(p+eps));
    rate = H*(b_size^2);
end