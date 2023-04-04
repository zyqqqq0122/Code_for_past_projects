load("SinusInNoise1.mat");
[p1,w1] = periodogram(y1);

figure(1);
plot(w1/pi/2,10.^(p1/10));
grid on;
xlabel('nomalized frequency');
ylabel('power spectral density');

[p2,w] = periodogram(y2);

figure(2);
plot(w/2/pi,10.^(p2/10));
grid on;
xlabel('nomalized frequency');
ylabel('power spectral density');

