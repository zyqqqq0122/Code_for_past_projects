function d = distortion(img1,img2)

    diff = img1 - img2;
    d = sum(diff(:).^2)/numel(diff(:));
    
end

