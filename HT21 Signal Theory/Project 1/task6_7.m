clc;
clear;
close all;

v = -0.5:0.005:0.5;
R = 1./(1+0.25^2-2*0.25*cos(2*v*pi));
figure;subplot(2,1,1);plot(v,R);title('power spectrum of X1(n)');
grid on ;
xlabel('Normalized Frequency');
ylabel('power specturm(dB)');

h=abs(1./(1-0.25*exp(-1i*2*pi*v)));
R2 = h.*h.*R;
subplot(2,1,2);plot(v,R2);title('power spectrum of X2(n)');
grid on ;
xlabel('Normalized Frequency');
ylabel('power specturm(dB)');

n=-100:100;
r2=abs(ifft(R2));
figure;plot(n,R2);title('ACF of X2(n)');grid on;
xlabel('n');ylabel('')