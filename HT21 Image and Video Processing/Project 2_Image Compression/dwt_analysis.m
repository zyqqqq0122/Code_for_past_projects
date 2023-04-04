function [cA,cD] = dwt_analysis(s,psv)

    Lo_R=psv./norm(psv);         % reconstruction LPF
    Hi_R=qmf(Lo_R);              % reconstruction HPF
    Lo_D=wrev(Lo_R);             % decomposition LPF
    Hi_D=wrev(Hi_R);             % decomposition HPF
    % [Lo_D,Hi_D,Lo_R,Hi_R]=orthfilt(psv);
    [cA,cD]=dwt(s,Lo_D,Hi_D,'mode','per');
    
end

