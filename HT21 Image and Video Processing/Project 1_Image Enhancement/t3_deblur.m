function [J] = t3_deblur(g,h,v)
    g1=double(g);
    nsr=v/var(g1(:));
    g_taper=edgetaper(g1,h);
    J=uint8(t3_wiener(g_taper,h,nsr));
end

