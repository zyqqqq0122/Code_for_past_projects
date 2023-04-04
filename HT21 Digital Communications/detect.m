function bhat = detect(r)
% bhat = detect(r)
%
% Computes the received bits given a received sequence of (phase-corrected)
% QPSK symbols. Gray coding of the individual bits is assumed. Hence, the
% two bits for each symbol can be detected from the real and imaginary
% parts, respectively. The first of the two bits below is output first in
% the bhat-sequence.
%
% Assumed mapping:
%
%  10 x   |   x 00
%         |
%  -------+-------
%         |
%  11 x   |   x 01
%
% Input:
%   r  = sequence of complex-valued QPSK symbols
%
% Output:
%   bhat  = bits {0,1} corresponding to the QPSK symbols

N=length(r);
bhat=zeros(1,N);
a=(r+conj(r))./2;
b=-1i.*(r-conj(r))./2;

for i=1:N
    
    if(a(i)<0)
        bhat(2*i-1)=1;
    else
        bhat(2*i-1)=0;
    end
    
    if(b(i)<0)
        bhat(2*i)=1;
    else
        bhat(2*i)=0;
    end
    
end
