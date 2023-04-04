function [im_rec] = ifwt(wave_coeff,scale,psv)

% input: wavelet coefficients (wave_coeff, which consists of LL, LH, HL, ...
%        HH of different scales), scale, prototype scalimg vector (psv)
% output: reconstructed image (im_rec)(double)
    
    M=length(wave_coeff);

    for i=scale:-1:1
        
       LL{1,i}=wave_coeff(1:M/2^i,1:M/2^i);
       LH{1,i}=wave_coeff(1+M/2^i:M/2^(i-1),1:M/2^i);
       HL{1,i}=wave_coeff(1:M/2^i,1+M/2^i:M/2^(i-1));
       HH{1,i}=wave_coeff(1+M/2^i:M/2^(i-1),1+M/2^i:M/2^(i-1));
       temp=cat(1,[LL{1,i} HL{1,i}],[LH{1,i} HH{1,i}]);
       wave_coeff(1:M/2^(i-1),1:M/2^(i-1))=synthesis_rec(temp,psv);
        
    end
    
    im_rec=wave_coeff;

end

