clear all
close all

EbN0_db = 0:10;                     % Eb/N0 values to simulate (in dB)

%% BER-Time Delay
figure;
t_samp=20:1:70;
BER=zeros(length(EbN0_db),length(t_samp));
for k=1:length(EbN0_db)
	
    for i=1:length(t_samp)
        [BER(k,i),~] = simulation_func_sync(EbN0_db(k),t_samp(i));
    end
    
end

for k=1:length(EbN0_db)
    plot(t_samp,BER(k,:),'.');
    hold on;
end

set(gca, 'YScale', 'log');
xlabel('Time Delay');
ylabel('BER');
title('BER-Time Delay for Different E_{b}/N_{0}');

%% BER-EbN0 for non-ideal synchronization 
BER=zeros(1,length(EbN0_db));
BER_th=zeros(1,length(EbN0_db));

for i=1:11
    [BER(i),BER_th(i)] = simulation_func_sync(EbN0_db(i),47);
end

figure;
plot(EbN0_db,BER);
hold on;
plot(EbN0_db,BER_th);

set(gca, 'YScale', 'log');
xlabel('E_{b}/N_{0}(dB)');
ylabel('BER');
legend('BER Simulation','BER Theory');
title('BER-E_{b}/N_{0} for Non-ideal Synchronization');

%% BER-Phase Error
figure;
phihat=-pi:2*pi/100:pi;
BER=zeros(length(EbN0_db),length(phihat));

for k=1:length(EbN0_db)
	
    for i=1:length(phihat)
        [BER(k,i),~] = simulation_func_phase(EbN0_db(k),phihat(i));
    end
    
end

for k=1:length(EbN0_db)
    plot(phihat,BER(k,:),'.');
    hold on 
end

set(gca, 'YScale', 'log');
xlabel('Phase Estimation');
ylabel('BER');
title('BER-Phase Estimation for Different E_{b}/N_{0}');

%% BER-EbN0 for non-ideal phase estimation
BER=zeros(1,length(EbN0_db));
BER_th=zeros(1,length(EbN0_db));

for i=1:11
    [BER(i),BER_th(i)] = simulation_func_phase(EbN0_db(i),pi*0.1);
end

figure;
plot(EbN0_db,BER);
hold on;
plot(EbN0_db,BER_th);

set(gca, 'YScale', 'log');
xlabel('E_{b}/N_{0}(dB)');
ylabel('BER');
legend('BER Simulation','BER Theory');
title('BER-E_{b}/N_{0} for Non-ideal Phase Estimation');