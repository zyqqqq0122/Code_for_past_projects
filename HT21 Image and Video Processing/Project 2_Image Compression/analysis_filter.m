function [cA,cD] = analysis_filter(s,psv)

% analysis bank function: 1-D, 2-band
% input: signal (s), prototype scaling vector (psv)
% output: approximation coefficients (cA1), detail coefficients (cD1)

    Lo_R=psv./norm(psv);         % reconstruction LPF
    Hi_R=qmf(Lo_R);              % reconstruction HPF
    Lo_D=wrev(Lo_R);             % decomposition LPF
    Hi_D=wrev(Hi_R);             % decomposition HPF
    
    % [Lo_D,Hi_D,Lo_R,Hi_R]=orthfilt(psv);
    
    % periodic extension
    L=length(psv);
    f=wextend('1D','per',s,L);
    % f=wextend('1D','sym',s,L-1);
    
    % filtering
    F=conv(f,Lo_D,'same');
    G=conv(f,Hi_D,'same');
    
    % truncating
    FF=F(L+1:end-L);
    GG=G(L+1:end-L);
    % FF=F(L-1:end-L);
    % GG=G(L-1:end-L);
    
    % downsampling
    cA=downsample(FF,2);        % approximation coefficients
    cD=downsample(GG,2);        % detail coefficients



%     % filtering
%     F=conv(f,Lo_D);
%     G=conv(f,Hi_D);
%     
%     % downsampling
%     FF=downsample(F,2);
%     GG=downsample(G,2);
%     
%     p=floor((3*length(s)-2)/4);
%     
%     % truncating
%     cA=FF(p:floor(p+length(s)/2));         % approximation coefficients
%     cD=GG(p:floor(p+length(s)/2));         % detail coefficients

end

