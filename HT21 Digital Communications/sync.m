function t_samp = sync(mf, b_train, Q, t_start, t_end)
% t_samp = sync(mf, b_train, Q, t_start, t_end)
%
% Determines when to sample the matched filter outputs. The synchronization
% algorithm is based on cross-correlating a replica of the (known)
% transmitted training sequence with the output from the matched filter
% (before sampling). Different starting points in the matched filter output
% are tried and the shift yielding the largest value of the absolute value
% of the cross-correlation is chosen as the sampling instant for the first
% symbol.
%
% Input:
%   mf            = matched filter output, complex baseband, I+jQ
%   b_train       = the training sequence bits
%   Q             = number of samples per symbol
%   t_start       = start of search window
%   t_end         = end of search window
%
% Output:
%   t_samp = sampling instant for first symbol

QPSK_train=qpsk(b_train);
L=length(QPSK_train);
corr=zeros(1,t_end-t_start);

for t_samp=t_start:(t_end-1)
    
    sum=0;
    mf_1=zeros(1,L);
    
    for k=1:L
        mf_1(k)=conj(mf((k-1)*Q+t_samp))*QPSK_train(k);
        sum=sum+mf_1(k);
    end
    
    corr(t_samp-t_start+1)=sum;
    
end

% plot(abs(corr));
[~,loc]=max(abs(corr));
t_samp=loc-1+t_start;

end



