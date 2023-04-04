function [wave_coeff] = fwt(im,scale,psv)

% input: image (im), scale, prototype scalimg vector (psv)
% output: wave coefficients (wave_coeff, LL, LH, HL, HH)
    
    [a{1,1},b{1,1},c{1,1},d{1,1}]=analysis_fb(im,psv);
    coeff=cat(1,[a{1,1} c{1,1}],[b{1,1} d{1,1}]);
    M=length(coeff);
    wave_coeff=zeros(M);
        
    for i=1:scale
        [a{1,i+1},b{1,i+1},c{1,i+1},d{1,i+1}]=analysis_fb(a{1,i},psv);
        wave_coeff(1:M/2^(i-1),1:M/2^(i-1))=...
            cat(1,[a{1,i} c{1,i}],[b{1,i} d{1,i}]);
    end

end