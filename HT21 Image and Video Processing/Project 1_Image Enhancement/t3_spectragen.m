function [J] = t3_spectragen(I)
fftI=fft2(I);
sfftI=fftshift(fftI);
A=abs(sfftI);
J=(A-min(min(A)))./(max(max(A)-min(min(A)))).*255;
% J=mat2gray(log(A));
end

