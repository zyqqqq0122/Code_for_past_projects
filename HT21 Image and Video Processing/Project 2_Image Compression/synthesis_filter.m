function [rr] = synthesis_filter(psv,cA,cD)

% synthesis bank function: 1-D, 2-band
% input: wavelet coeffocients (biorthogonal filters) (cA, cD),
%        prototype scaling vector (psv)
% output: reconstructed signal (rr),
%         approximation part (A), detail part(D)

    Lo_R=psv./norm(psv);         % reconstruction LPF
    Hi_R=qmf(Lo_R);              % reconstruction HPF

    % upsampling
    F=upsample(cA,2);
    G=upsample(cD,2);
    
    % periodic extension
    L=length(psv);
    FF=wextend('1D','per',F,L);
    GG=wextend('1D','per',G,L);
    % FF=wextend('1D','sym',F,L-1);
    % GG=wextend('1D','sym',G,L-1);
    
    % filtering
    A1=conv(FF,Lo_R,'same');
    D1=conv(GG,Hi_R,'same');
    % A=conv(FF,Lo_R,'same');
    % D=conv(GG,Hi_R,'same');
    
    % truncating
    A=A1(L+1:end-L);
    D=D1(L+1:end-L);
    
    rr=circshift(A+D,1);
%     r=A+D;
%     rr=r(L:end-L+1);
    
end
