clear all;
close all;

% Initialization
EbN0_db = 0:10;                     % Eb/N0 values to simulate (in dB)
nr_bits_per_symbol = 2;             % Corresponds to k in the report
nr_guard_bits = 10;                 % Size of guard sequence (in nr bits)
                                    % Guard bits are appended to transmitted bits so
                                    % that the transients in the beginning and end
                                    % of received sequence do not affect the samples
                                    % which contain the training and data symbols.
nr_data_bits = 1000;                % Size of each data sequence (in nr bits)
nr_training_bits = 100;             % Size of training sequence (in nr bits)
nr_blocks = 50;                     % The number of blocks to simulate
Q = 8;                              % Number of samples per symbol in baseband
delay=0;
% Define the pulse-shape used in the transmitter. 
% Pick one of the pulse shapes below or experiemnt
% with a pulse of your own.
pulse_shape = ones(1, Q);
%pulse_shape = root_raised_cosine(Q);

% Matched filter impulse response. 
mf_pulse_shape = fliplr(pulse_shape);


% Loop over different values of Eb/No.
nr_errors = zeros(1, length(EbN0_db));   % Error counter
for snr_point = 1:length(EbN0_db)
  
  % Loop over several blocks to get sufficient statistics.
  for blk = 1:nr_blocks

    %%%
    %%% Transmitter
    %%%

    % Generate training sequence.
    b_train = training_sequence(nr_training_bits);
    
    % Generate random source data {0, 1}.
    b_data = random_data(nr_data_bits);

    % Generate guard sequence.
    b_guard = random_data(nr_guard_bits);
 
    % Multiplex training and data into one sequence.
    b = [b_guard b_train b_data b_guard];
sps=8;   
m=length(b);
bits = m;                                 % number of bits
iphase = 0;									% initial phase
% initialize vectors
data = zeros(1,bits);
dd = zeros(1,m/2); qq = zeros(1,m/2); 
theta = zeros(1,m/2);
thetaout = zeros(1,sps*m/2);
% set direct and quadrature bit streams
error_num = 0;
    data = round(rand(1,bits));
    dd = data(1:2:bits-1);
    qq = data(2:2:bits);
    % main programs
    theta(1) = iphase;                        % set initial phase
    thetaout(1:sps) = theta(1)*ones(1,sps);
    for k=2:m/2
       if dd(k) == 1
          phi_k = (2*qq(k)-1)*pi/4;
       else
          phi_k = (2*qq(k)-1)*3*pi/4;
       end   
       theta(k) = phi_k + theta(k-1);
      
    end
    d = cos(theta);
    q = sin(theta);
   
    df= upfirdn(d, pulse_shape, Q, 1);
    qf= upfirdn(q, pulse_shape, Q, 1);
     %%%
    %%% AWGN Channel
    %%%
    
    % Compute variance of complex noise according to report.
    sigma_sqr = norm(pulse_shape)^2 / nr_bits_per_symbol / 10^(EbN0_db(snr_point)/10);
  
    % postprocessor for plotting
    % 通过AWGN信道
        d_awgn = df + sqrt(sigma_sqr/2)*(randn(size(df)));
        q_awgn = qf + sqrt(sigma_sqr/2)*(randn(size(qf)));
        Rx_Sampletime = 1 + (1:1:m/2-1)*sps + delay;
        % 接收取最佳采样点，每sps个点取一个
        % 接收滤波器，进行匹配滤波
        BRx = ones(1,sps); ARx=1;
        % matched filter parameters
        d_filter = filter(BRx,ARx,d_awgn);
        q_filter = filter(BRx,ARx,q_awgn);
        % 差分解码
        W = d_filter(Rx_Sampletime);
        Z = q_filter(Rx_Sampletime);
        U = zeros(1,m/2);
        V = zeros(1,m/2);
        d_demod = zeros(1,length(Rx_Sampletime));
        q_demod = zeros(1,length(Rx_Sampletime));
        % 判决过程
        for k = 2:length(Rx_Sampletime)
            U(k) = W(k)*W(k-1) + Z(k)*Z(k-1);
            V(k) = Z(k)*W(k-1) - W(k)*Z(k-1);
            if U(k)>0
                d_demod(k)=1;
            else
                d_demod(k)=0;
            end
            if V(k)>0
                q_demod(k)=1;
            else
                q_demod(k)=0;
            end
            if d_demod(k)~=dd(k)
                nr_errors(snr_point) = nr_errors(snr_point) + 1;
            end
            if q_demod(k)~=qq(k)
                nr_errors(snr_point) = nr_errors(snr_point) + 1;
            end
        end


    % Next block.
  end

  % Next Eb/No value.
end

% Compute the BER. 
BER = nr_errors / nr_data_bits / nr_blocks;
plot(EbN0_db,BER, 'LineWidth', 2); hold on;
BER_th = qfunc(sqrt(2*10.^(EbN0_db/10)))-0.5.*(qfunc(sqrt(2*10.^(EbN0_db/10))).^2);
plot(EbN0_db,BER_th, 'LineWidth', 2);
set(gca, 'YScale', 'log')
xlabel('Eb/N0 (dB)')
ylabel('BER (dB)')
legend('BER simulation',  'BER theory')


