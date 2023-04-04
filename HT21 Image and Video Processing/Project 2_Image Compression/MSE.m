function mse = MSE(x,y)
    mse=sum(sum((y-x).^2))/(size(y,1) * size(y,2));
end

