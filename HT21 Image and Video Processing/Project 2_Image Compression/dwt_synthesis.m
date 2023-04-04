function [s_rec] = dwt_synthesis(cA,cD,psv)

    Lo_R=psv./norm(psv);         % reconstruction LPF
    Hi_R=qmf(Lo_R);              % reconstruction HPF
    Lo_D=wrev(Lo_R);             % decomposition LPF
    Hi_D=wrev(Hi_R);             % decomposition HPF
    % [Lo_D,Hi_D,Lo_R,Hi_R]=orthfilt(psv);
    s_rec=idwt(cA,cD,Lo_R,Hi_R,'mode','per');
    
end

