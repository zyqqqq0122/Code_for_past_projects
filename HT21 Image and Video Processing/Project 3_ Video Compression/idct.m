function J = idct(im, mask)
    [row, col] = size(im);
    M = mask;
    block_x = floor(col/M);
    block_y = floor(row/M);
    
    A = dctmtx(8);
    J = zeros(block_y*M,block_x*M);
    
    for i=1:M:(block_y*M)
        for j=1:M:(block_x*M)
            J(i:i+(M-1),j:j+(M-1)) = A'*im(i:i+(M-1),j:j+(M-1))*A;
        end
    end
end

