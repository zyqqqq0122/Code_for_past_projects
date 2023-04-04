function [im_rec] = synthesis_rec(wave_coeff,psv)

% input: wavelet coefficients (wave_coeff),prototype scaling vector (psv)
% output: reconstructed image (im_rec)

    N=length(wave_coeff);
    LL=wave_coeff(1:N/2,1:N/2);
    LH=wave_coeff(1+N/2:N,1:N/2);
    HL=wave_coeff(1:N/2,1+N/2:N);
    HH=wave_coeff(1+N/2:N,1+N/2:N);

    %% method 1
    M=length(LL);
    im_rec=zeros(2*M);
    
    % step1
    Lo_D=zeros(2*M,M);
    Hi_D=zeros(2*M,M);
    for i=1:M
       Lo_D(:,i)=synthesis_filter(psv,LL(:,i),LH(:,i));     % designed by our own
       Hi_D(:,i)=synthesis_filter(psv,HL(:,i),HH(:,i));
       % Lo_D(:,i)=dwt_synthesis(LL(:,i),LH(:,i),psv);      % according to matlab
       % Hi_D(:,i)=dwt_synthesis(HL(:,i),HH(:,i),psv);
    end
    
    % step2
    for j=1:2*M
        im_rec(j,:)= synthesis_filter(psv,Lo_D(j,:),Hi_D(j,:));
        % im_rec(j,:)= dwt_synthesis(Lo_D(j,:),Hi_D(j,:),psv);
    end

    %% matlab algpoithm
%     im_rec=idwt2(LL,LH,HL,HH,Lo_R,Hi_R,'mode','per');
end
