function quantized = quantizer(x,delta)
    quantized=sign(x)*delta.*floor(abs(x)/delta+1/2);
    % quantized=round(x/delta)*delta;
end