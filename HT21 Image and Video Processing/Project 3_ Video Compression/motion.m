function shift = motion(f_pre,f_cur, d, blk_size)

    [height,width] = size(f_pre);

    MSE = zeros(height/blk_size,width/blk_size,length(d));
    shift = zeros(1,height/blk_size*width/blk_size);

    temp = 1;
    for i=1:height/blk_size
        for j=1:width/blk_size
            for u=1:length(d) 
                
                row = 1+(i-1)*blk_size : 1+(i-1)*blk_size + blk_size-1;
                col = 1+(j-1)*blk_size : 1+(j-1)*blk_size + blk_size-1;
               
                %shift
                
                r_s = row + d(1,u);
                c_s = col + d(2,u);

                if(r_s(1) < 1 || c_s(1) < 1 || ...
                        r_s(blk_size) > height || c_s(blk_size) > width)
                    MSE(i,j,u) = 99999999999999;
                    continue;
                end
                
                diff = (f_pre(r_s,c_s)-f_cur(row,col)).^2;
                MSE(i,j,u) = sum(diff(:))/numel(diff);
                
            end
            
            shift1 = find(MSE(i,j,:) == min(MSE(i,j,:)));
            
            if length(shift1) > 1  
                shift1 = shift1(1,1);
            end
            
            shift(1,temp) = shift1;
            temp = temp + 1;
            
        end
    end
end
