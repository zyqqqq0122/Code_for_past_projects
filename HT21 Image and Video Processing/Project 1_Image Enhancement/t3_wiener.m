function J = t3_wiener(I,PSF,nsr)
% G(k,l) = (H*(k,l)./(|H(k,l)|^2 + NSR

%% H
H = psf2otf(PSF, size(I));

%% G
denom = (abs(H).^2) + nsr;
G = conj(H) ./ denom;
J = ifft2(G .* fft2(I));

end