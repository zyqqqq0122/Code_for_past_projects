function [LL,LH,HL,HH] = analysis_fb(im,psv)

% input: image (im), prototype scaling vector (psv)
% output: low-low, low-high, high-low, high-high wavelet coefficients (LL, LH, HL, HH)

    %% method 1
    M=length(im);
    
    % step1
    Lo_D=zeros(M,M/2);
    Hi_D=zeros(M,M/2);
    
    for i=1:M
       % [Lo_D(i,:),Hi_D(i,:)]=analysis_filter(im(i,:),psv);
       [Lo_D(i,:),Hi_D(i,:)]=dwt_analysis(im(i,:),psv);
    end

    % step2
    LL=zeros(M/2,M/2);
    LH=zeros(M/2,M/2);
    HL=zeros(M/2,M/2);
    HH=zeros(M/2,M/2);
    
    for j=1:M/2
        % [LL(:,j),LH(:,j)]=analysis_filter(Lo_D(:,j),psv);
        % [HL(:,j),HH(:,j)]=analysis_filter(Hi_D(:,j),psv);
        [LL(:,j),LH(:,j)]=dwt_analysis(Lo_D(:,j),psv);
        [HL(:,j),HH(:,j)]=dwt_analysis(Hi_D(:,j),psv);
    end
    %% matlab algpoithm
%     Lo_R=psv./norm(psv);         % reconstruction LPF
%     Hi_R=qmf(Lo_R);              % reconstruction HPF
%     Lo_D=wrev(Lo_R);             % decomposition LPF
%     Hi_D=wrev(Hi_R);             % decomposition HPF
%     % [Lo_D,Hi_D,Lo_R,Hi_R]=orthfilt(psv);
%     [LL,HL,LH,HH]=dwt2(im,Lo_D,Hi_D,,'mode','per');
    
end

