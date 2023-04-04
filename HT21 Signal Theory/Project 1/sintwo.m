load("SinusInNoise2.mat");
[p,w] = periodogram(y);
figure(1);

plot(w/pi/2,10.^(p/10));
grid on;
xlabel('nomalized frequency');
ylabel('power spectral density');

